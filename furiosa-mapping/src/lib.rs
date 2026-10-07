//! TCP mapping expressions.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]

// Re-export these so that users only need to depend on this crate, not the individual crates.
pub use furiosa_mapping_macro::*;
pub use furiosa_mapping_types::*;

use abi_stable::std_types::{RResult, RSlice, RVec, Tuple2, Tuple3};

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
        pub(super) fn mapping_dma_tails(src: &Mapping, dst: &Mapping) -> Tuple3<usize, usize, usize>;
        pub(super) fn mapping_split_at(slf: &Mapping, target: usize) -> Tuple2<Mapping, Mapping>;
        pub(super) fn mapping_index(slf: &Mapping, position: usize) -> Cell;
        pub(super) fn mapping_indexes(slf: &Mapping) -> RVec<Cell>;
        pub(super) fn mapping_axes(slf: &Mapping) -> RVec<AxisTerm>;
        pub(super) fn mapping_iter(
            slf: &Mapping,
            axes: RSlice<'_, AxisTerm>,
            base: &Index,
            padding: bool,
        ) -> MappingIter;
        pub(super) fn index_finalize(slf: Index) -> RResult<RSortedMap<Ident, usize>, PaddingKind>;
    }
}

/// Methods for [`Mapping`] backed by the impl crate over FFI. Pure `Mapping` operations live as
/// inherent methods in `furiosa-mapping-types`; this trait carries only the ones that need the impl.
pub trait MappingExt: Sized {
    /// Normalizes to canonical form.
    fn normalize(&self) -> Self;
    /// The DMA burst tails for copying source `self` into destination `dst`, as
    /// `(src_packet, dst_packet, common)`.
    ///
    /// `common` is the innermost volume both sides read the same way, which is the volume either side
    /// can SEQUENCE. Nothing is subtracted from it: padding the two sides share is part of the addressing
    /// they share, so it is transferred like any other cell. Only a `Bottom` write hole or a `Zero` that
    /// must read as zero caps it, since no burst may cross one. A DRAM sink pins its tail to it.
    ///
    /// Each packet grows `common` through THAT side's own trailing `Top` padding, so the two may differ (a
    /// padded source against a dense sink) and are reconciled by the tail alignment rather than by a
    /// shared peel size. A packet need not divide its buffer; the caller that peels one checks.
    ///
    /// `(1, 1, 1)` when the two share nothing but the origin cell, a transpose being the plain case.
    fn dma_tails(&self, dst: &Self) -> (usize, usize, usize);
    /// Returns true if `self` is a resize (innermost prefix) of `original`.
    fn is_resize_of(&self, original: &Self) -> bool;
    /// Splits the mapping into `(outer, inner)` at a positive divisor of its size.
    fn split_at(&self, target: usize) -> Result<(Self, Self), SplitAtError>;
    /// The read at buffer `position`, an [`Index`] of per-axis contributions with composites kept
    /// WHOLE. Call [`IndexExt::finalize`] on a live index, or [`CellExt::finalize`] on the cell with
    /// an explicit out-of-bounds fill kind.
    fn index(&self, position: usize) -> Cell;
    /// The raw read at every buffer position `0..size`, composites kept WHOLE, in one FFI crossing. The
    /// batch form of `index`; call [`CellExt::finalize`] with the desired out-of-bounds fill kind.
    fn indexes(&self) -> Vec<Cell>;
    /// The live axis terms (its [`AxisTerm`]s, resolved symbols), padding excluded.
    fn axes(&self) -> Vec<AxisTerm>;
    /// Whether every position reads the same cell across this mapping's extent.
    fn is_broadcast(&self) -> bool;
    /// Whether this mapping contains inaccessible [`PaddingKind::Bottom`] cells.
    fn has_bottom_pad(&self) -> bool;
    /// Whether every position of this mapping holds a cell of its own: no padding of any kind.
    fn has_no_padding(&self) -> bool;
    /// A lazy iterator over where this mapping's cells land in a buffer laid out by `axes`, each offset
    /// shifted by `base`: one `Option<usize>` per physical cell in canonical order (`Some(off)` live,
    /// `None` padding). `padding` picks the traversal: `true` visits every physical cell (the wire order
    /// the buffer seam needs), `false` skips the padding cells (faster on padded mappings, for relayouts).
    ///
    /// `axes` is the *target* buffer's layout (resolved symbol terms, e.g. from [`MappingExt::axes`]),
    /// not necessarily this mapping's own axes: the mapping is projected
    /// onto it. A symbol in both contributes its coordinate; a symbol only in the mapping contributes
    /// nothing (it broadcasts, e.g. a dst axis absent from the src); a symbol only in `axes` stays at
    /// its origin. The offsets land inside the buffer only if `axes` covers the mapping's reach: each
    /// shared axis's extent must hold the mapping's coordinate for it (an out-of-range coordinate
    /// wraps, like an out-of-range `base`).
    ///
    /// The returned [`MappingIter`] is an [`Iterator`]; it is fully owned (borrows nothing), so a
    /// consumer just walks it (`.zip`, `.map`, `.flatten`) without ever naming the type.
    fn iter(&self, axes: &[AxisTerm], base: &Index, padding: bool) -> MappingIter;
    /// The leftover `self − piece` (the scatter/gather/reduce/broadcast residue): `self` is sequenced as
    /// the STREAM against `piece` as memory, so a `self` cell `piece` backs becomes a `Top` pad and a cell
    /// it does not back stays live (the broadcast). Runs in [`SequencerMode::Carve`], so a `Bottom` pad in
    /// `self` (a `view_mut().tile()` hole) is tolerated as padding. Panics unless `piece` is in `self`.
    fn carve(&self, piece: &Self) -> Self;
    /// Narrows the axis at `stride` x `size` to its first `window` positions, restoring the rest as
    /// `hole` padding.
    fn window_axis(
        &self,
        stride: usize,
        size: usize,
        window: usize,
        hole: PaddingKind,
    ) -> Result<Self, WindowAxisError>;
    /// Returns the stride of one consecutive region occupied by `axis`.
    fn find_axis(&self, axis: &Self) -> Result<usize, FindAxisError>;
    /// Removes `axis` as one exact region, leaving the surrounding mapping.
    fn remove_axis(&self, axis: &Self) -> Result<Self, FindAxisError>;
}

/// Why a mapping cannot be split at the requested size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cannot split a mapping of {size} cells at {target}: both sizes must be positive and target must divide size")]
pub struct SplitAtError {
    /// The mapping's extent in cells.
    pub size: usize,
    /// The requested inner extent in cells.
    pub target: usize,
}

/// Why [`MappingExt::window_axis`] could not narrow an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WindowAxisError {
    /// The window is empty, or reaches past the axis it narrows.
    #[error("a window of {window} does not fit an axis of {size} positions")]
    WindowExceedsAxis {
        /// The requested window.
        window: usize,
        /// The axis extent it must fit in.
        size: usize,
    },
    /// The axis's span does not divide the mapping.
    #[error("an axis of {stride} x {size} does not fit a mapping of {cells} cells")]
    SpanDoesNotDivide {
        /// Cells one position of the axis advances.
        stride: usize,
        /// The axis extent, padding included.
        size: usize,
        /// The mapping's extent in cells.
        cells: usize,
    },
}

/// Why [`MappingExt::find_axis`] could not locate an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FindAxisError {
    /// The sequencer matched no term, or matched only part of the axis. Does not prove the axis is
    /// absent.
    #[error("this mapping does not step through the whole axis")]
    NotInMapping,
    /// The mapping holds every cell of the axis, but in pieces at different strides, so no single
    /// stride advances it.
    #[error("this mapping holds the axis in pieces, not in one strided run")]
    ScatteredInMapping,
    /// The axis span is not a region this mapping can be cut at: it reaches past the mapping, or
    /// stops short of it at a span that does not divide it.
    #[error("an axis spanning {} cells does not divide this mapping of {} cells", .0.target, .0.size)]
    SpanDoesNotDivide(#[from] SplitAtError),
}

/// Matches each of `streams` against the `memories`, each its own address space (the fetch engine's
/// `Time` and `Packet`), and returns one coalesced [`SequencerConfig`] per stream — each entry's
/// `memory_index` says which memory it reads. The streams' segments are pooled and carved together
/// (term priority is global across streams), so a `Broadcast` never claims a pad a `Term` needs.
///
/// Every mode but [`SequencerMode::Locate`] enforces coverage: a live cell left unread or unwritten
/// gives [`SequencerError::Unconsumed`]. Inputs are read-only, the carving runs on internal copies.
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

/// Cuts a power-of-two mapping into outermost-first two-position slots.
/// Panics when the mapping size is not a power of two.
pub fn into_slots_of_size_two(mapping: &Mapping) -> Vec<Mapping> {
    let size = mapping.size();
    assert!(
        size.is_power_of_two(),
        "a mapping counted in size-2 slots must have a power-of-two size, got {size}"
    );
    let mut slots = Vec::with_capacity(size.trailing_zeros() as usize);
    let mut rest = mapping.normalize();
    while rest.size() > 1 {
        // Normalize each slot separately so equivalent padded spellings collapse alike.
        slots.push(rest.clone().modulo(2).normalize());
        rest = rest.stride(2).normalize();
    }
    slots.reverse();
    slots
}

impl MappingExt for Mapping {
    fn normalize(&self) -> Self {
        unsafe { sys::mapping_normalize(self) }
    }

    fn dma_tails(&self, dst: &Self) -> (usize, usize, usize) {
        let Tuple3(src_packet, dst_packet, common) = unsafe { sys::mapping_dma_tails(self, dst) };
        (src_packet, dst_packet, common)
    }

    fn is_resize_of(&self, original: &Self) -> bool {
        let n = self.size();
        n <= original.size() && self.normalize() == original.clone().resize(n).normalize()
    }

    fn split_at(&self, target: usize) -> Result<(Self, Self), SplitAtError> {
        let size = self.size();
        if size == 0 || target == 0 || !size.is_multiple_of(target) {
            return Err(SplitAtError { size, target });
        }
        let Tuple2(outer, inner) = unsafe { sys::mapping_split_at(self, target) };
        Ok((outer, inner))
    }

    fn index(&self, position: usize) -> Cell {
        unsafe { sys::mapping_index(self, position) }
    }

    fn indexes(&self) -> Vec<Cell> {
        unsafe { sys::mapping_indexes(self) }.into_iter().collect()
    }

    fn axes(&self) -> Vec<AxisTerm> {
        unsafe { sys::mapping_axes(self) }.into_iter().collect()
    }

    fn is_broadcast(&self) -> bool {
        self.normalize() == Mapping::Broadcast { size: self.size() }
    }

    fn has_bottom_pad(&self) -> bool {
        match self {
            Mapping::Padding { inner, kind, .. } => *kind == PaddingKind::Bottom || inner.has_bottom_pad(),
            Mapping::Stride { inner, .. } | Mapping::Modulo { inner, .. } | Mapping::Resize { inner, .. } => {
                inner.has_bottom_pad()
            }
            Mapping::Pair { left, right } => left.has_bottom_pad() || right.has_bottom_pad(),
            Mapping::Symbol { .. } | Mapping::Broadcast { .. } => false,
        }
    }

    fn has_no_padding(&self) -> bool {
        match self {
            Mapping::Padding { .. } => false,
            Mapping::Stride { inner, .. } | Mapping::Modulo { inner, .. } | Mapping::Resize { inner, .. } => {
                inner.has_no_padding()
            }
            Mapping::Pair { left, right } => left.has_no_padding() && right.has_no_padding(),
            Mapping::Symbol { .. } | Mapping::Broadcast { .. } => true,
        }
    }

    fn iter(&self, axes: &[AxisTerm], base: &Index, padding: bool) -> MappingIter {
        unsafe { sys::mapping_iter(self, RSlice::from(axes), base, padding) }
    }

    fn carve(&self, piece: &Self) -> Self {
        let configs =
            sequence(&[piece], &[self], SequencerMode::Carve).expect("carve: piece must be contained in self");
        let mut acc = Mapping::identity();
        for (_key, entry) in configs[0].0.iter() {
            let seg = if entry.memory_stride == 0 {
                entry.mapping.clone() // unbacked: a live broadcast, or a Top hole left as padding
            } else {
                Mapping::identity().padding(entry.mapping.size(), PaddingKind::Top) // matched: Top pad, positions kept
            };
            acc = seg.pair(acc);
        }
        acc
    }

    fn window_axis(
        &self,
        stride: usize,
        size: usize,
        window: usize,
        hole: PaddingKind,
    ) -> Result<Self, WindowAxisError> {
        if window == 0 || window > size {
            return Err(WindowAxisError::WindowExceedsAxis { window, size });
        }
        // An axis whose span overflows `usize` divides no mapping either, so both refusals below
        // are the same one.
        let error = WindowAxisError::SpanDoesNotDivide {
            stride,
            size,
            cells: self.size(),
        };
        let span = stride.checked_mul(size).ok_or(error)?;
        let (above, axis_and_below) = self.split_at(span).map_err(|_| error)?;
        // `span` is `stride * size`, so the axis divides the region the first split cut out.
        let (axis, below) = axis_and_below.split_at(stride).expect("a stride divides its own span");
        Ok(above
            .pair(axis.resize(window).padding(size, hole))
            .pair(below)
            .normalize())
    }

    fn find_axis(&self, axis: &Self) -> Result<usize, FindAxisError> {
        // The axis is the stream, this mapping the memory, and only part of it is read, hence `Locate`.
        // The stream is the axis alone, so the innermost entry (key `1`) holds its step.
        let config = sequence(&[self], &[axis], SequencerMode::Locate).map_err(|_| FindAxisError::NotInMapping)?;
        let entries = &config.first().ok_or(FindAxisError::NotInMapping)?.0;
        let entry = entries.get(&1).ok_or(FindAxisError::NotInMapping)?;
        // How much of the axis was found, before asking how it is laid out. The entries partition the
        // axis, and one that reads no memory is a piece this mapping does not hold at all.
        let located: usize = entries
            .iter()
            .filter(|(_key, entry)| entry.memory_stride != 0)
            .map(|(_key, entry)| entry.mapping.size())
            .product();
        if located < axis.size() {
            return Err(FindAxisError::NotInMapping);
        }
        // Every cell is there, so it must be there as one strided run: a second entry means the mapping
        // holds the axis in pieces, each at its own stride.
        if entries.len() != 1 {
            return Err(FindAxisError::ScatteredInMapping);
        }
        // A single-position broadcast needs one cell of volume and so passes the check above, and
        // still has to be refused for reading no memory.
        let stride = entry.memory_stride;
        if stride == 0 {
            return Err(FindAxisError::NotInMapping);
        }
        // The sequencer absorbs padding on both sides, so `m![A # 32 / 8]` widened to 8 positions
        // still reports stride 32, twice the 128 cells present.
        let span = stride
            .checked_mul(axis.size())
            .expect("an axis span stays inside usize");
        if span > self.size() {
            return Err(SplitAtError {
                size: self.size(),
                target: span,
            }
            .into());
        }
        Ok(stride)
    }

    fn remove_axis(&self, axis: &Self) -> Result<Self, FindAxisError> {
        if axis.size() == 1 {
            return Ok(self.clone());
        }
        let stride = self.find_axis(axis)?;
        // `find_axis` returns a stride whose span fits the mapping, so this cannot overflow.
        let span = stride * axis.size();
        let (outer, axis_and_inner) = self.split_at(span)?;
        // `span` is `stride * axis.size()`, so the axis divides the region just cut out.
        let (found, inner) = axis_and_inner.split_at(stride).expect("a stride divides its own span");
        // Compare the extracted digit itself; rebuilding a truncated axis can change it.
        if found.normalize() != axis.normalize() {
            return Err(FindAxisError::NotInMapping);
        }
        Ok(outer.pair(inner))
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
    /// Adds a mapping to this index.
    fn add_mapping<I: crate::M>(&mut self, value: usize, oob_fill: PaddingKind) -> Result<(), PaddingKind>;
    /// The terminal read: decodes every composite and returns each symbol's absolute coordinate
    /// (`Ident` -> coordinate, coord-0 dropped), or the [`PaddingKind`] a pad cell lands on.
    fn finalize(self) -> RResult<RSortedMap<Ident, usize>, PaddingKind>;
}

/// Finalization helper for a mapped cell. Padding and out-of-bounds cells remain errors.
pub trait CellExt {
    /// Decode a live cell's logical coordinates.
    /// `oob_fill` supplies the fill kind for [`Cell::OutOfBounds`].
    fn finalize(self, oob_fill: PaddingKind) -> RResult<RSortedMap<Ident, usize>, PaddingKind>;
}

impl CellExt for Cell {
    fn finalize(self, oob_fill: PaddingKind) -> RResult<RSortedMap<Ident, usize>, PaddingKind> {
        match self {
            Cell::Index(index) => index.finalize(),
            Cell::Padding(kind) => RResult::RErr(kind),
            Cell::OutOfBounds => RResult::RErr(oob_fill),
        }
    }
}

impl IndexExt for Index {
    fn add_mapping<I: crate::M>(&mut self, value: usize, oob_fill: PaddingKind) -> Result<(), PaddingKind> {
        match I::to_value().index(value) {
            Cell::Index(index) => self.add(index),
            Cell::Padding(kind) => return Err(kind),
            Cell::OutOfBounds => return Err(oob_fill),
        }
        Ok(())
    }
    fn finalize(self) -> RResult<RSortedMap<Ident, usize>, PaddingKind> {
        unsafe { sys::index_finalize(self) }
    }
}
