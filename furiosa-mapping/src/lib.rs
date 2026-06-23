//! TCP mapping expressions.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]

// Re-export these so that users only need to depend on this crate, not the individual crates.
pub use furiosa_mapping_macro::*;
pub use furiosa_mapping_types::*;

use abi_stable::std_types::{RResult, RSlice, RVec, Tuple2};

/// Raw `extern "C-unwind"` decls for the prebuilt impl's exports.
mod sys {
    use super::*;

    #[expect(improper_ctypes, reason = "all types are #[repr(C)] + StableAbi")]
    unsafe extern "C-unwind" {
        pub(super) fn mapping_sequence(
            memories: RSlice<'_, Mapping>,
            streams: RSlice<'_, Mapping>,
            mode: SequencerMode,
        ) -> RResult<RVec<SequencerConfig>, SequencerError>;
        pub(super) fn mapping_normalize(slf: &Mapping) -> Mapping;
        pub(super) fn mapping_split_at(slf: &Mapping, target: usize) -> Tuple2<Mapping, Mapping>;
        pub(super) fn mapping_index(slf: &Mapping, position: usize) -> Index;
        pub(super) fn mapping_indexes(slf: &Mapping) -> RVec<Index>;
        pub(super) fn mapping_axes(slf: &Mapping) -> RVec<Term>;
        pub(super) fn index_finalize(slf: Index) -> RResult<RSortedMap<Ident, usize>, PaddingKind>;
        pub(super) fn index_gen_indexes(slf: &Index, mapping: &Mapping) -> RVec<Index>;
    }
}

/// Methods for [`Mapping`] backed by the impl crate over FFI. Pure `Mapping` operations live as
/// inherent methods in `furiosa-mapping-types`; this trait carries only the ones that need the impl.
pub trait MappingExt: Sized {
    /// Normalizes to canonical form.
    fn normalize(&self) -> Self;
    /// Returns true if `self` is a resize (innermost prefix) of `original`.
    fn is_resize_of(&self, original: &Self) -> bool;
    /// Splits at buffer `target` into `(outer, inner)`: `outer` strides past the first `target`
    /// cells, `inner` keeps them, and `outer.pair(inner)` reads the same buffer. `target` must
    /// divide the size.
    fn split_at(&self, target: usize) -> (Self, Self);
    /// The read at buffer `position`, an [`Index`] of per-axis contributions with composites kept
    /// WHOLE. Call [`IndexExt::finalize`] to decode them (`mapping.index(i).finalize()`).
    fn index(&self, position: usize) -> Index;
    /// The raw read at every buffer position `0..size`, composites kept WHOLE, in one FFI crossing. The
    /// batch form of `index`; call [`IndexExt::finalize`] on a cell to decode it.
    fn indexes(&self) -> Vec<Index>;
    /// The live axis terms (its [`Term`]s), padding excluded.
    fn axes(&self) -> Vec<Term>;
    /// Carves `piece` out of `self` through the matcher, returning the leftover the
    /// scatter/gather/reduce/broadcast sites need. Each matched cell becomes a `Top` pad (the carved
    /// hole), each unmatched cell stays live (a broadcast at `memory_stride` 0). Panics unless `piece`
    /// is contained in `self`.
    fn carve(&self, piece: &Self) -> Self;
}

/// Matches each of `streams` against the `memories`, each its own address space (the fetch engine's
/// `Time` and `Packet`), and returns one coalesced [`SequencerConfig`] per stream — each entry's
/// `memory_index` says which memory it reads. The streams' segments are pooled and carved together
/// (term priority is global across streams), so a `Broadcast` never claims a pad a `Term` needs.
///
/// Coverage is enforced: every memory must end fully consumed for `mode` (a live cell left unread under
/// Read / unwritten under Write is [`SequencerError::Unconsumed`], carrying the carved-down memories so
/// the caller can name the offender). Inputs are read-only; the carving happens on internal copies.
pub fn sequence(
    memories: &[&Mapping],
    streams: &[&Mapping],
    mode: SequencerMode,
) -> Result<Vec<SequencerConfig>, SequencerError> {
    let memories: RVec<Mapping> = memories.iter().map(|m| (*m).clone()).collect();
    let streams: RVec<Mapping> = streams.iter().map(|m| (*m).clone()).collect();
    let configs = unsafe { sys::mapping_sequence(memories.as_rslice(), streams.as_rslice(), mode) }.into_result()?;
    Ok(configs.into_iter().map(SequencerConfigExt::coalesce).collect())
}

impl MappingExt for Mapping {
    fn normalize(&self) -> Self {
        unsafe { sys::mapping_normalize(self) }
    }

    fn is_resize_of(&self, original: &Self) -> bool {
        let n = self.size();
        n <= original.size() && self.normalize() == original.clone().resize(n).normalize()
    }

    fn split_at(&self, target: usize) -> (Self, Self) {
        let Tuple2(outer, inner) = unsafe { sys::mapping_split_at(self, target) };
        (outer, inner)
    }

    fn index(&self, position: usize) -> Index {
        unsafe { sys::mapping_index(self, position) }
    }

    fn indexes(&self) -> Vec<Index> {
        unsafe { sys::mapping_indexes(self) }.into_iter().collect()
    }

    fn axes(&self) -> Vec<Term> {
        unsafe { sys::mapping_axes(self) }.into_iter().collect()
    }

    fn carve(&self, piece: &Self) -> Self {
        let configs = sequence(&[piece], &[self], SequencerMode::Read).expect("carve: piece must be contained in self");
        let mut acc = Mapping::identity();
        for (_key, entry) in configs[0].0.iter() {
            let seg = if entry.memory_stride == 0 {
                // Unbacked stream cell, a live leftover axis (a broadcast vs the carved memory).
                entry.mapping.clone()
            } else {
                // Matched cell, a carved hole kept as Top padding so positions are preserved.
                Mapping::identity().padding(entry.mapping.size(), PaddingKind::Top)
            };
            acc = seg.pair(acc);
        }
        acc
    }
}

/// Methods for [`SequencerConfig`].
pub trait SequencerConfigExt {
    /// Coalesces entries that sit back-to-back in the memory buffer into one, to minimize the
    /// sequencer entry count. Result entries are keyed by cumulative-product stream stride, so
    /// adjacent entries are always stream-contiguous; the merge condition is memory-contiguity
    /// alone (`inner.memory_stride * inner.size == outer.memory_stride`). The merged entry keeps
    /// the inner's key and memory stride, with the run paired into one mapping.
    ///
    /// Apply this PER descriptor (e.g. packet and time separately), never across, or it would
    /// coalesce two entries that must stay distinct.
    fn coalesce(self) -> Self;
}

impl SequencerConfigExt for SequencerConfig {
    fn coalesce(self) -> SequencerConfig {
        let mut out: Vec<(usize, SequencerEntry)> = Vec::new();
        for (key, entry) in self.0 {
            match out.last_mut() {
                // Same memory and memory-contiguous: extend the run. This (outer) entry wraps the
                // inner run; the run keeps the inner's key and memory stride, growing only its paired
                // mapping. Entries in different memories (different spaces) never merge.
                Some((_, run))
                    if run.memory_index == entry.memory_index
                        && run.memory_stride * run.mapping.size() == entry.memory_stride =>
                {
                    let inner = std::mem::replace(&mut run.mapping, Mapping::identity());
                    run.mapping = entry.mapping.pair(inner);
                }
                _ => out.push((key, entry)),
            }
        }
        SequencerConfig(out.into_iter().collect())
    }
}

/// Methods for [`Index`].
pub trait IndexExt: Sized {
    /// Marks this index as invalid.
    fn mark_invalid(&mut self);
    /// Adds a mapping to this index.
    fn add_mapping<I: crate::M>(&mut self, value: usize);
    /// The terminal read: decodes every composite and returns each symbol's absolute coordinate
    /// (`Ident` -> coordinate, coord-0 dropped), or the [`PaddingKind`] a pad cell lands on.
    fn finalize(self) -> RResult<RSortedMap<Ident, usize>, PaddingKind>;
    /// Generates all possible indexes based on the given mapping.
    fn gen_indexes(&self, mapping: Mapping) -> RVec<Self>;
}

impl IndexExt for Index {
    fn mark_invalid(&mut self) {
        self.0 = RResult::RErr(PaddingKind::Bottom);
    }
    fn add_mapping<I: crate::M>(&mut self, value: usize) {
        self.add(I::to_value().index(value));
    }
    fn finalize(self) -> RResult<RSortedMap<Ident, usize>, PaddingKind> {
        unsafe { sys::index_finalize(self) }
    }
    fn gen_indexes(&self, mapping: Mapping) -> RVec<Self> {
        unsafe { sys::index_gen_indexes(self, &mapping) }
    }
}
