use std::marker::PhantomData;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;
use rayon::prelude::*;

use crate::backend::op_prep::{broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::cast::{Cast, ContractionAccumulator, ContractionCast};
use crate::scalar::*;
use crate::storage::par_iters::MappingPositions;
use crate::storage::{PAR_MIN_JOB, min_cells_per_job};

pub trait Buf: From<Vec<u8>> + Into<Vec<u8>> + AsRef<[u8]> + AsMut<[u8]> + Clone + std::fmt::Debug {}

impl<T> Buf for T where T: From<Vec<u8>> + Into<Vec<u8>> + AsRef<[u8]> + AsMut<[u8]> + Clone + std::fmt::Debug {}

/// Cpu / Npu tensor: the dense device image as one packed `Vec<u8>`. Element `i` sits at position
/// `i` (padding included, zero-initialized); a byte-multiple `D` takes its own byte run, a sub-byte `D`
/// (`f4e2m1` / `i4`) packs two per byte. Only [`Scalar::load`] / [`Scalar::store`] know the width, so the
/// generic-over-`D` ops address elements by logical index. The element count is byte-aligned (enforced by
/// [`Self::from_vec`]), so it is recovered from the byte length alone.
///
/// Layout-free: ops take their mapping(s) from type parameters. Reads parallelize via [`Self::par_iter`];
/// a parallel write partitions the byte image into disjoint chunks ([`Self::par_chunks_mut`]).
#[derive(Clone, Debug)]
pub struct BufStorage<D: Scalar, B: crate::storage::Buf> {
    bytes: B,
    _marker: PhantomData<D>,
}

/// Byte-wise equality over the packed device image: two `BufStorage`s are equal iff their raw bytes
/// match. This is the right notion for an opaque storage buffer: identical bit patterns (a `NaN`
/// included) compare equal, which is what tensor-image comparison wants. Byte equality is
/// intentionally STRICTER than `D`'s value equality: `from_vec` stores each code verbatim (it does
/// not canonicalize), so the two `f4e2m1` zero codes (`0x0` = +0, `0x8` = -0) are value-equal yet
/// compare unequal here. (Float `PartialEq`'s `NaN != NaN` is a value-level rule and does not apply
/// to a byte image.)
impl<D: Scalar, B: Buf> PartialEq for BufStorage<D, B> {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}
impl<D: Scalar + Eq, B: Buf> Eq for BufStorage<D, B> {}

impl<D: Scalar, B: Buf> From<B> for BufStorage<D, B> {
    fn from(bytes: B) -> Self {
        Self {
            bytes,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, B: Buf> BufStorage<D, B> {
    /// The host buffer byte length for `n` elements (exact: `n` is byte-aligned by the
    /// [`Self::from_vec`] invariant). A byte-multiple width divides evenly for any `n`; a 4-bit width
    /// needs an even `n`.
    fn byte_len(n: usize) -> usize {
        n * D::STAGING_BITS / 8
    }

    /// Packs a logical `Vec<D>` on `D::STAGING_BITS` via [`Scalar::store`]. The single constructor and sole
    /// byte-alignment enforcement point. No `mapping`-size check here (`BufStorage` is layout-free); the
    /// `data.len() == Mapping::SIZE` gate lives in the `Tensor::from_vec` wrapper, matching `MathStorage`.
    pub(crate) fn from_vec(data: impl IntoIterator<Item = D>) -> Self {
        let vals: Vec<D> = data.into_iter().collect();
        assert!(
            (vals.len() * D::STAGING_BITS).is_multiple_of(8),
            "BufStorage<D>: a sub-byte element count must be byte-aligned (got {} elements at {} bits)",
            vals.len(),
            D::STAGING_BITS,
        );
        // `Scalar::to_buf` is the single packer: a byte-multiple width memcpys the whole slice, a
        // sub-byte width packs two codes per byte (zero-initializing first, since each nibble store is a
        // read-modify-write that reads the byte to preserve its sibling). Wrapping that output is the
        // inverse of `into_buf`.
        Self::from_buf(D::to_buf(&vals))
    }

    /// Wraps a pre-packed device byte image directly (the inverse of [`Self::into_buf`]): the bytes ARE
    /// the packed buffer, so no per-element pack. Peer of [`Self::from_vec`] (which packs logical values);
    /// pre-packed data (fp4 / f4e2m1 weights) comes through here to avoid a decode + re-pack round-trip.
    pub(crate) fn from_buf(bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes.into(),
            _marker: PhantomData,
        }
    }

    pub(crate) fn inner(&self) -> &B {
        &self.bytes
    }

    /// A zeroed packed buffer for `n` elements — a blank canvas a relayout overwrites. `vec![0u8; _]`
    /// lowers to `alloc_zeroed` (calloc), skipping the eager `Vec<D>` memset + pack that `from_vec` does.
    pub(crate) fn zeroed(n: usize) -> Self {
        assert!(
            (n * D::STAGING_BITS).is_multiple_of(8),
            "BufStorage<D>: a sub-byte element count must be byte-aligned (got {} elements at {} bits)",
            n,
            D::STAGING_BITS,
        );
        Self {
            // We only need an uninit alloc here (zero cost): relayouts write only the live cells.
            // This zero-fill instead pays a real O(n) memset, pending a MaybeUninit rework of Buf.
            bytes: vec![0u8; D::buf_bytes(n)].into(),
            _marker: PhantomData,
        }
    }

    /// The element count, recovered from the byte length on `D::STAGING_BITS` (exact: the count is
    /// byte-aligned). `STAGING_BITS`, not `BITS`: a staging type (`i5`/`i9`) sits in its wider backing
    /// integer here, and `BITS` names only the wire width.
    pub(crate) fn len(&self) -> usize {
        self.bytes.as_ref().len() * 8 / D::STAGING_BITS
    }

    /// Reads element `i` via [`Scalar::load`].
    #[inline]
    pub(crate) fn get(&self, i: usize) -> D {
        D::load(self.bytes.as_ref(), i)
    }

    /// Reads element `i` if in range, else `None` (the guarded `get` the fold paths use for a
    /// split-then-padded out-of-range read).
    #[inline]
    pub(crate) fn try_get(&self, i: usize) -> Option<D> {
        (i < self.len()).then(|| self.get(i))
    }

    /// Writes `value` into element `i` via [`Scalar::store`], leaving neighbouring elements untouched.
    #[inline]
    pub(crate) fn set(&mut self, i: usize, value: D) {
        D::store(self.bytes.as_mut(), i, value);
    }

    /// The logical element values as a `Vec<D>`, decoded without consuming the buffer (one `D` per
    /// element). Borrowing readback peer of `from_vec`; consuming callers move through `into_vec` /
    /// `into_buf`.
    pub(crate) fn to_vec(&self) -> Vec<D> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    /// The elements as an indexed parallel iterator, each decoded via [`Scalar::load`]. Unlike a write, a
    /// sub-byte read has no aliasing hazard (two threads decoding sibling nibbles of the same byte never
    /// race), so a plain index range suffices; no byte-chunk partitioning needed here, unlike
    /// [`Self::par_chunks_mut`] (its mutable, byte-owning peer). Backs [`Self::map`] and the zips.
    pub(crate) fn par_iter(&self) -> impl IndexedParallelIterator<Item = D> + '_
    where
        B: Sync,
    {
        (0..self.len()).into_par_iter().map(move |i| self.get(i))
    }

    /// The elements as a parallel iterator of disjoint, byte-aligned mutable chunks: the safe
    /// parallel-write primitive (mutable peer of [`Self::par_iter`]). Each rayon job gets one
    /// [`BufChunkMut`] owning a distinct element range and writes only within it via [`BufChunkMut::set`].
    ///
    /// Keying disjointness on the element index is unsound once `BITS < 8`: two elements share a byte,
    /// so sibling writes from different jobs race on it. Partitioning the byte image makes each chunk
    /// own whole bytes, race-free for every width with no `unsafe`. `min_elems` is the per-chunk floor
    /// (rounded up to the byte-alignment quantum). A sub-window writer intersects [`BufChunkMut::range`]
    /// with its window. Backs transpose / gather.
    pub(crate) fn par_chunks_mut(
        &mut self,
        min_elems: usize,
    ) -> impl IndexedParallelIterator<Item = BufChunkMut<'_, D>> {
        // The alignment quantum `align = lcm(8, STAGING_BITS) / STAGING_BITS` is the fewest elements that fill a whole
        // number of bytes (2 for a 4-bit width, 1 for a byte-multiple width); round `min_elems` up to it so
        // each chunk owns a whole number of elements, byte-aligned on both ends.
        const CACHE_LINE: usize = 64;
        let align = lcm(8, D::STAGING_BITS) / D::STAGING_BITS;
        let elems_per_chunk = min_elems.next_multiple_of(align).max(align);
        // Round the chunk up to a cache line so two adjacent chunks never share one: byte-disjoint already
        // guarantees correctness, but a `bytes_per_chunk` off a 64B boundary leaves neighbours sharing a
        // boundary line, so a write in each false-shares it. 64 is a multiple of every scalar's byte
        // quantum (1 / 2 / 4B), so the whole-byte invariant is preserved.
        let bytes_per_chunk = Self::byte_len(elems_per_chunk).next_multiple_of(CACHE_LINE);
        // Recompute elems from the rounded byte count so `lo = c * elems_per_chunk` stays exact: `64 * 8`
        // is divisible by every `D::STAGING_BITS` (4 / 8 / 16 / 32), so the division is lossless.
        let elems_per_chunk = bytes_per_chunk * 8 / D::STAGING_BITS;
        self.bytes
            .as_mut()
            .par_chunks_mut(bytes_per_chunk)
            .enumerate()
            .map(move |(c, bytes)| {
                // Each chunk owns the element range starting at `c * elems_per_chunk`; the last chunk may be
                // short, so its length is what its own bytes hold.
                let lo = c * elems_per_chunk;
                let hi = lo + bytes.len() * 8 / D::STAGING_BITS;
                BufChunkMut {
                    bytes,
                    lo,
                    hi,
                    _marker: PhantomData,
                }
            })
    }

    /// Element-wise map to a new scalar. Elements are independent, so [`Self::par_iter`] loads / maps each
    /// across the rayon pool and `from_vec` repacks the result on `D2::BITS`. Backs
    /// [`crate::backend::Backend::map`].
    pub(crate) fn map<D2: Scalar>(&self, f: impl Fn(D) -> D2 + Sync) -> BufStorage<D2, B>
    where
        B: Sync,
    {
        // Pass `&f` (a moved `f` would demand `f: Send`); this keeps `f` `Sync`-only with no closure.
        BufStorage::from_vec(self.par_iter().map(&f).collect::<Vec<_>>())
    }

    /// Element-wise zip of two same-layout physical buffers, bare `D`, offset-aligned. Backs
    /// [`crate::backend::Backend::zip_with`].
    pub(crate) fn zip_with<D2: MaterializableScalar, D3: Scalar>(
        &self,
        other: &BufStorage<D2, B>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> BufStorage<D3, B>
    where
        D: MaterializableScalar,
        B: Sync,
    {
        // Reads never race, so zip the two buffers' parallel element iterators (the physical packing is
        // transparent through `par_iter`). The output repacks on `D3::BITS` via `from_vec`.
        let data = self
            .par_iter()
            .zip(other.par_iter())
            .map(|(a, b)| f(a, b))
            .collect::<Vec<_>>();
        BufStorage::from_vec(data)
    }

    /// Element-wise ternary zip over the physical buffer. Ternary peer of [`Self::zip_with`].
    pub(crate) fn zip3_with<D2: MaterializableScalar, D3: MaterializableScalar, D4: Scalar>(
        &self,
        b: &BufStorage<D2, B>,
        c: &BufStorage<D3, B>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> BufStorage<D4, B>
    where
        D: MaterializableScalar,
        B: Sync,
    {
        let data = self
            .par_iter()
            .zip(b.par_iter())
            .zip(c.par_iter())
            .map(|((a, b), c)| f(a, b, c))
            .collect::<Vec<_>>();
        BufStorage::from_vec(data)
    }

    /// Writes a transposed/broadcast view of `src` into `self` (the destination) via a sequencer
    /// walk over the physical buffer. `src_map` / `dst_map` are the two storages' base (live-axis)
    /// mappings, used only to resolve a partial-view offset's wire base (see [`window_base`]). Backs
    /// [`crate::backend::Backend::transpose`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transpose<Src: M, Mapping: M>(
        &mut self,
        src: &BufStorage<D, B>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &MappingValue,
        dst_map: &MappingValue,
        allow_broadcast: bool,
    ) where
        B: Sync,
    {
        // Structural check (also asserts `Src` is contained in `Mapping`); `!allow_broadcast`
        // rejects a non-padding leftover.
        let _ = transpose_broadcast::<Src, Mapping>(allow_broadcast);
        let src_view = &Src::to_value();
        let dst_view = &Mapping::to_value();
        // Relayout: sequence the `Src` buffer (memory) against the `Mapping` (dst) layout (stream)
        // under `Carve`. Each dst buffer position reads one Src element; a broadcast axis (in dst, not Src)
        // gets `memory_stride` 0, so one Src element feeds the whole broadcast run. `Carve` (not `Read`)
        // tolerates a `Bottom` pad in the dst stream, a `view_mut().tile()` write hole, read as a `Top`
        // don't-care, which the dst liveness walk then gates out of the write below.
        let config = &sequence(&[src_view], &[dst_view], SequencerMode::Carve)
            .expect("transpose: Src must be covered by the dst stream")[0];
        // The src window's physical base and the dst window's base. Both default to 0 for an empty
        // (whole-tensor) offset. The base resolves the offset against the *base* (live-axis) map, where
        // a tiled offset axis the view carries as padding is still a live `Symbol` (the view shares the
        // base's wire layout, so the wire position carries over).
        let src_base = window_base(src_map, src_offset);
        let dst_base = window_base(dst_map, dst_offset);
        // Walk the dst window, zipping three position-aligned streams of the dst layout: the dst elements
        // (each position writes its own, `dst_base + pos`), the src offset to read (`config`, splits via
        // seek), and the dst liveness (`iter_positions()` yields `None` for a padding / write-hole cell), so
        // liveness is read inline, not materialized into a mask. A live position reads `src`'s element and
        // writes its own dst element; the stream's trailing `Bottom`-pad positions, which would address past
        // the window, are clamped away below. The write is range-partitioned on the dst (each rayon job
        // owns a byte-aligned element range), so the scatter is safe for every width, no branch on packing.
        // The per-chunk position range is bounded by both stream lengths so a seek never runs past either.
        let stream = config.stream_size().min(dst_view.iter_positions().len());
        self.par_chunks_mut(PAR_MIN_JOB).for_each(|mut chunk| {
            // The chunk owns the dst elements in `chunk.range()`; the positions it writes are those whose
            // `dst_base + pos` lands in the chunk (and in range of the stream). Seek both the src offsets
            // and the liveness to that contiguous position range and walk them position-aligned.
            let elems = chunk.range();
            let pos_lo = elems.start.saturating_sub(dst_base);
            let pos_hi = elems.end.saturating_sub(dst_base).min(stream);
            if pos_lo >= pos_hi {
                return;
            }
            let src_pos = config.iter_range(pos_lo, pos_hi);
            let live = dst_view.iter_positions().range(pos_lo, pos_hi);
            for (i, (src_pos, live)) in src_pos.zip(live).enumerate() {
                if live.is_some() {
                    chunk.set(dst_base + pos_lo + i, src.get(src_base + src_pos));
                }
            }
        });
    }

    /// Reduces the factors of `self`'s mapping that are absent in `Dst`. `Dst` must be a factor of
    /// the source mapping (only reduce away existing axes; no broadcast). Backs [`crate::backend::Backend::reduce`].
    ///
    /// Per-output-cell: each output cell independently folds its block of
    /// the reduced (residue) axes into its own slot (`par_iter_mut`), no shared accumulator. The output
    /// cell's source read base is the cell's coordinate projected onto the source (a broadcast axis `Dst`
    /// adds projects to 0, so its fan shares one base); the residue deltas are the reduced axes, shared
    /// across cells. So a read is the additive `base + residue_delta` over the physical buffer.
    ///
    /// REASSOCIATES: one output cell's residue block is folded left-to-right in `residue` order, which is
    /// the physical (wire) order of the reduced axes, not necessarily the serial-`iter` stream order, so
    /// `reduce_fn` runs in an implementation-defined order. For an associative `reduce_fn` the result
    /// equals the serial one; for a non-associative one (e.g. `f32` add/mul) the Cpu result can
    /// differ from serial. Accepted: Cpu is not bit-reproducible for non-associative reductions.
    /// The order is still deterministic across runs regardless of rayon's split, since each cell folds
    /// its whole block alone.
    pub(crate) fn reduce<Src: M, Dst: M, R: Fn(D, D) -> D + Sync>(
        &self,
        reduce_fn: R,
        identity: D,
        allow_broadcast: bool,
    ) -> Self
    where
        D: MaterializableScalar,
        B: Sync,
    {
        let src = Src::to_value();
        let dst = Dst::to_value();
        // Axes `Dst` adds beyond the reduced source are broadcast; `!allow_broadcast` rejects them.
        let broadcast = broadcast_axes(&src, &dst);
        assert!(
            allow_broadcast || broadcast.axes().is_empty(),
            "reduce: Dst adds axes absent from the source; pass allow_broadcast=true for reduce-then-broadcast"
        );
        // The kept axes (`Dst` minus the broadcast) sit in both source and Dst; the reduced (residue)
        // axes are the source's beyond them. A broadcast axis (in `dst`, not the source) reads nothing,
        // so its fan copies share one source base and recompute the fold.
        let inter = dst.carve(&broadcast);
        let reduce_residue = src.carve(&inter);
        let out_size = dst.size();
        let plan = ReadPlan::new(&src, &dst, &reduce_residue);
        // Fold each independent output cell across the rayon pool into a logical `Vec<D>`, then pack the
        // result on `D::STAGING_BITS` via `from_vec`. `with_min_len` holds each job at >= `PAR_MIN_JOB` work
        // (cells x live residue) so small reduces stay one job. A dead (padding) cell has no reads and
        // keeps its `identity`. A split-then-padded reduced axis can land a read past the physical
        // buffer, so `try_get` skips an out-of-range one.
        let mut out = vec![identity; out_size];
        out.par_iter_mut()
            .with_min_len(min_cells_per_job(plan.cell_work()))
            .enumerate()
            .for_each(|(o, slot)| {
                let Some(reads) = plan.reads(o) else { return };
                let mut acc = identity;
                for p in reads {
                    if let Some(v) = self.try_get(p) {
                        acc = reduce_fn(acc, v);
                    }
                }
                *slot = acc;
            });
        Self::from_vec(out)
    }

    /// Fused contraction (generalized matmul) for the physical buffer: each output cell independently
    /// sums `lhs * rhs` over the contracted axes (those in `Union` but absent from `Out`). O(`Out`)
    /// memory, no `Union` outer product. The one fold behind [`Self::contraction`] and
    /// [`Self::contraction_prewidened`], which pick `Acc` so no caller names it.
    ///
    /// Per-output-cell: each operand offset splits into the output
    /// cell's read base plus a contracted-coordinate delta, and each output cell folds that block into
    /// its own slot (`par_iter_mut`), no shared accumulator. An operand axis absent from `Out` (a
    /// contracted axis) is a residue delta; an axis the operand lacks broadcasts (its base / delta is 0).
    ///
    /// Widens each operand on load (`D -> Acc`), sums in `Acc`, and narrows once into the slot, so an
    /// `i8` matmul accumulates in `i32` without a widened copy of either operand. `D` may be a staging
    /// type: an `i9` stream folds here without ever reaching memory.
    ///
    /// One `Backend::contraction` is a single within-slice contraction (its `Union`/`Out` carry no
    /// `Slice` axis), so its narrow is the per-slice Lane Folder narrow. A cross-slice reduction is a
    /// separate downstream Vector Engine `reduce` over the already-narrowed per-slice results, so the
    /// narrow here never spans slices (pinned by `cross_slice_reduce_narrows_per_lane_fold_not_globally`).
    ///
    /// REASSOCIATES, like [`Self::reduce`]: one output cell's products are summed in physical (wire)
    /// order of the contracted axes, not the serial-`iter` stream order. Exact for an associative
    /// combine; for `f32` add/mul the Cpu result can differ from serial. Accepted; the order is
    /// still deterministic across runs regardless of rayon's split.
    fn contraction_in<Acc>(
        lhs: &BufStorage<D, B>,
        rhs: &BufStorage<D, B>,
        lhs_map: &MappingValue,
        rhs_map: &MappingValue,
        pre_reduce: &MappingValue,
        out_map: &MappingValue,
    ) -> BufStorage<D, B>
    where
        D: Cast<Acc>,
        Acc: ContractionAccumulator + Cast<D>,
        B: Sync,
    {
        // A bare `BufStorage` carries no axes of its own, so the operand layouts arrive as
        // `lhs_map`/`rhs_map` rather than being read off the storage (contrast `MathStorage`).
        // The contracted axes are `pre_reduce` beyond the kept output, exactly `reduce`'s residue.
        let contracted = pre_reduce.carve(out_map);
        let out_size = out_map.size();
        // One read base per output cell into each operand (the output coordinate, contracted at 0), plus
        // the per-contracted-coordinate operand offset deltas shared across cells; the residue is the
        // contracted axes (an operand axis absent from `Out` reduces, one it lacks broadcasts).
        let lhs_plan = ReadPlan::new(lhs_map, out_map, &contracted);
        let rhs_plan = ReadPlan::new(rhs_map, out_map, &contracted);
        let mut out = vec![num_traits::Zero::zero(); out_size];
        // Each output cell sums the products over its contracted block (the two plans' reads zip
        // position-aligned). A dead (padding) cell has no reads and keeps its zero. A split-then-padded
        // contracted coordinate can land past a buffer, so skip an out-of-range read.
        out.par_iter_mut()
            .with_min_len(min_cells_per_job(lhs_plan.cell_work()))
            .enumerate()
            .for_each(|(o, slot)| {
                let (Some(lhs_reads), Some(rhs_reads)) = (lhs_plan.reads(o), rhs_plan.reads(o)) else {
                    return;
                };
                // The narrow sits outside the fold: one per output cell, as the Lane Folder does.
                // `filter_map` drops a split-then-padded out-of-range read.
                let acc = lhs_reads
                    .zip(rhs_reads)
                    .filter_map(|(lp, rp)| match (lhs.try_get(lp), rhs.try_get(rp)) {
                        (Some(l), Some(r)) => Some(Cast::cast(l) * Cast::cast(r)),
                        _ => None,
                    })
                    .fold(<Acc as num_traits::Zero>::zero(), |acc, prod| acc + prod);
                *slot = Cast::cast(acc);
            });
        Self::from_vec(out)
    }

    /// [`Self::contraction_in`] accumulating in [`ContractionCast::Output`], the width the engine's
    /// Multiplier widens this operand type to. Peer of [`crate::backend::Backend::contraction`].
    pub(crate) fn contraction(
        lhs: &BufStorage<D, B>,
        rhs: &BufStorage<D, B>,
        lhs_map: &MappingValue,
        rhs_map: &MappingValue,
        pre_reduce: &MappingValue,
        out_map: &MappingValue,
    ) -> BufStorage<D, B>
    where
        D: ContractionCast,
        B: Sync,
    {
        Self::contraction_in::<<D as ContractionCast>::Output>(lhs, rhs, lhs_map, rhs_map, pre_reduce, out_map)
    }

    /// [`Self::contraction_in`] over operands already at accumulator width, so its casts are the
    /// identity. Peer of [`crate::backend::Backend::contraction_prewidened`].
    pub(crate) fn contraction_prewidened(
        lhs: &BufStorage<D, B>,
        rhs: &BufStorage<D, B>,
        lhs_map: &MappingValue,
        rhs_map: &MappingValue,
        pre_reduce: &MappingValue,
        out_map: &MappingValue,
    ) -> BufStorage<D, B>
    where
        D: ContractionAccumulator,
        B: Sync,
    {
        Self::contraction_in::<D>(lhs, rhs, lhs_map, rhs_map, pre_reduce, out_map)
    }

    /// Scatters `self` into `dst` at positions read from the `i32` index tensor. Backs
    /// [`crate::backend::Backend::scatter`].
    pub(crate) fn scatter<Src: M, Key: M, Dst: M, Idx: M>(
        &self,
        dst: &mut BufStorage<D, B>,
        index: &BufStorage<i32, B>,
        scaled: bool,
    ) {
        let key = Key::to_value();
        let (payload, dst_term) = scatter_params(&Src::to_value(), &Dst::to_value(), &key);
        let payload = payload.remove_padding();
        let scatter_axis = MappingValue::from_terms(std::iter::once(dst_term.to_term()));
        // The index tensor's buffer holds exactly one element per `Idx` position, which is why
        // `decode_indices` reads it whole.
        debug_assert_eq!(index.len(), Idx::SIZE);
        let indices = decode_indices(index, decode_stride::<D>(&payload, scaled));

        let key_size = key.size();
        let axis_size = scatter_axis.size();
        let src_off: Vec<usize> = sequence(&[&Src::to_value()], &[&payload.clone().pair(key)], SequencerMode::Read)
            .expect("scatter: Src factors into payload x key")[0]
            .iter()
            .collect();
        let dst_off: Vec<usize> = sequence(&[&Dst::to_value()], &[&payload.pair(scatter_axis)], SequencerMode::Read)
            .expect("scatter: Dst factors into payload x scatter-axis")[0]
            .iter()
            .collect();
        // Serial element copy through the accessor (scatter is inherently serial: distinct keys can map to
        // the same dst element, so the last write wins). The accessor keeps it correct for a packed dst.
        for payload_pos in 0..src_off.len() / key_size {
            for key_pos in 0..key_size {
                let src_elem = src_off[payload_pos * key_size + key_pos];
                let dst_elem = dst_off[payload_pos * axis_size + indices[key_pos]];
                let v = self.get(src_elem);
                dst.set(dst_elem, v);
            }
        }
    }

    /// Gathers from `self` (table) into `dst` at positions read from the index tensor. Backs
    /// [`crate::backend::Backend::gather`].
    pub(crate) fn gather<Src: M, Dst: M, Idx: M>(
        &self,
        dst: &mut BufStorage<D, B>,
        index: &BufStorage<i32, B>,
        scaled: bool,
    ) where
        D: MaterializableScalar,
        B: Sync,
    {
        let params = gather_params(&Src::to_value(), &Dst::to_value(), &Idx::to_value());
        let payload = params.payload.remove_padding();
        let gather_axis = MappingValue::from_terms(std::iter::once(params.src_term.to_term()));
        // The compact residue axes are exactly the index tensor's mapping.
        let idx_residue = Idx::to_value();
        let indices = decode_indices(index, decode_stride::<D>(&payload, scaled));

        let axis_size = gather_axis.size();
        let residue_size = idx_residue.size();
        let src_off: Vec<usize> = sequence(
            &[&Src::to_value()],
            &[&payload.clone().pair(gather_axis)],
            SequencerMode::Read,
        )
        .expect("gather: Src table factors into payload x gather-axis")[0]
            .iter()
            .collect();
        let dst_off: Vec<usize> = sequence(&[&Dst::to_value()], &[&payload.pair(idx_residue)], SequencerMode::Read)
            .expect("gather: Dst factors into payload x idx-residue")[0]
            .iter()
            .collect();
        // Each payload block writes its `residue_size` dst elements, and `dst_off` is a permutation of dst
        // positions (distinct), so the blocks scatter into disjoint slots. The index only repeats *reads*
        // (`src_off[... + indices[r]]`), never a write. The dst positions `dst_off` names are permuted, not
        // a contiguous range, so range-partitioning the dst (the uniform safe scatter) needs the inverse:
        // `writer[dst_elem]` is the `(payload_pos, r)` flat index that writes `dst_elem`, or `None` for a
        // dst position the gather does not touch (which then keeps its prior value).
        let mut writer: Vec<Option<usize>> = vec![None; dst.len()];
        for (flat, &dst_elem) in dst_off.iter().enumerate() {
            writer[dst_elem] = Some(flat);
        }
        // Range-partition the dst: each rayon job owns a contiguous, byte-aligned element range, reads its
        // permuted source through the shared `&` source (a race-free read), and writes only positions it
        // owns. Safe for every width, no branch on packing.
        dst.par_chunks_mut(min_cells_per_job(residue_size).saturating_mul(residue_size))
            .for_each(|mut chunk| {
                for dst_elem in chunk.range() {
                    let Some(flat) = writer[dst_elem] else { continue };
                    let (payload_pos, r) = (flat / residue_size, flat % residue_size);
                    let src_elem = src_off[payload_pos * axis_size + indices[r]];
                    chunk.set(dst_elem, self.get(src_elem));
                }
            });
    }

    /// Reinterprets the physical buffer from `Mapping`-shaped to `Mapping2`-shaped, returning the
    /// result. Backs [`crate::backend::Backend::reshape`].
    pub(crate) fn reshape<Mapping: M, Mapping2: M>(&self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        // Same physical buffer; only the type-level mapping changes. The packed buffer clones as-is
        // (its bit layout is layout-independent).
        self.clone()
    }

    /// Relabels the physical buffer from `src_map` to `dst_map`, compacting away any padding the
    /// relabel drops, so later offset-addressed ops see `dst_map`'s layout. Backs
    /// [`crate::backend::Backend::transmute`]. The vector-engine relayout is
    /// `…tile(k).read().transmute()`: the read leaves the buffer padded in `src_map`, the transmute
    /// narrows it to the compact `dst_map`.
    pub(crate) fn transmute(self, src_map: &MappingValue, dst_map: &MappingValue) -> Self
    where
        D: MaterializableScalar,
    {
        // Same layout drops no padding.
        if src_map == dst_map {
            return self;
        }
        Self::from_vec(place_live_elems(dst_map, self.live_elems_in_wire_order(src_map)))
    }

    /// The source's live elements in wire order, the real data, padding excluded.
    fn live_elems_in_wire_order(self, src_map: &MappingValue) -> impl Iterator<Item = D>
    where
        D: MaterializableScalar,
    {
        // Decode to logical elements first, then drop the padding positions the mapping marks dead.
        src_map
            .iter_positions()
            .zip(self.to_vec())
            .filter_map(|(offset, elem)| offset.is_some().then_some(elem))
    }

    /// Serialize to a flat logical `Vec<D>` in `mapping`-order (one `D` per element), decoding each packed
    /// element. The mapping is unused (the buffer is already wire-order).
    pub(crate) fn into_vec(self, _mapping: &MappingValue) -> Vec<D>
    where
        D: MaterializableScalar,
    {
        self.to_vec()
    }

    /// The dense physical device byte image: the exact packed bytes for the DMA / LIR boundary. The packed
    /// buffer already IS this image (one `Vec<u8>` packed on `D::BITS`), so this is a direct move, no
    /// re-pack.
    pub(crate) fn into_buf(self, _mapping: &MappingValue) -> Vec<u8> {
        self.bytes.into()
    }

    /// Borrow the packed device byte image (the borrowing peer of [`Self::into_buf`]), the bridge the
    /// NPU storage uses to move bytes in and out of this host-eval buffer.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

/// A byte-aligned element range of a [`BufStorage`] owned exclusively by one rayon job, handed out by
/// [`BufStorage::par_chunks_mut`] (which establishes disjointness). Writes go by global index
/// ([`Self::set`]) within [`Self::range`]; it touches only its own `&mut [u8]` slice.
pub(crate) struct BufChunkMut<'a, D: Scalar> {
    bytes: &'a mut [u8],
    lo: usize,
    hi: usize,
    _marker: PhantomData<D>,
}

impl<D: Scalar> BufChunkMut<'_, D> {
    /// The global element range `[lo, hi)` this chunk owns, byte-aligned on both ends (the last chunk may
    /// be short).
    pub(crate) fn range(&self) -> std::ops::Range<usize> {
        self.lo..self.hi
    }

    /// Writes `value` into element `i` (a global index, which must lie in [`Self::range`]) via
    /// [`Scalar::store`], leaving the chunk's other elements untouched.
    #[inline]
    pub(crate) fn set(&mut self, i: usize, value: D) {
        debug_assert!(
            (self.lo..self.hi).contains(&i),
            "BufChunkMut::set: element {i} escapes chunk [{}, {})",
            self.lo,
            self.hi
        );
        D::store(self.bytes, i - self.lo, value);
    }
}

/// The per-output-cell read plan for `reduce` / `contraction` over the physical buffer: how to find,
/// for each output cell, the operand buffer positions whose values fold into it.
///
/// Built from one sequencer config over the `out`-outer / `residue`-inner stream (which fully covers the
/// operand: `out` carries the kept axes, `residue` the reduced ones; a broadcast axis the operand lacks
/// contributes nothing). The operand wire offset is additive in the two coordinates
/// (`offset(o, r) = base(o) + delta(r)`), the same model the serial per-cell fold uses. The shared
/// `delta(r)` (one residue row, small) is precomputed; each cell's `base(o)` is seeked lazily by
/// [`Self::reads`], so there is no `out_size` base table and the fold is a single parallel pass (the
/// seek that would fill such a table happens inside it instead). [`Self::reads`] `.get()`-guards an
/// out-of-range sum (a split-then-padded coordinate).
struct ReadPlan {
    /// The operand wire offsets over the `out`-outer / `residue`-inner stream; `base(o)` is seeked from it.
    config: SequencerConfig,
    /// Live residue deltas added to a cell's base, in `residue` walk order (padding positions dropped).
    delta: Vec<usize>,
    /// The padded residue width: the stream stride between consecutive output cells.
    rwidth: usize,
    /// The first live residue column within a row (the base read's offset into the row).
    r0: usize,
    /// Total stream length; an output cell whose base position is `>= stream_size` is dead (padding).
    stream_size: usize,
}

impl ReadPlan {
    /// `memory` is the operand mapping, `out` the output mapping (its cell index `o` is the stream row),
    /// `residue` the operand's reduced / contracted axes (the `carve` leftover, which marks the kept /
    /// broadcast axes as `Top` pads).
    fn new(memory: &MappingValue, out: &MappingValue, residue: &MappingValue) -> Self {
        // The residue carries `Top` pads (the kept / broadcast axes `carve` marked, and any padded
        // operand axis): a padded position re-reads its live sibling (`memory_stride` 0), so folding it
        // would double-count. Walk the residue once to mask the live positions; only those become deltas.
        let live_r: Vec<usize> = residue
            .iter_positions()
            .enumerate()
            .filter_map(|(r, off)| off.map(|_| r))
            .collect();
        let rwidth = residue.size(); // the padded residue width: one full stream row below
        // A reduce / contraction always folds at least one cell (an empty residue is degenerate), so the
        // residue has a first live position; index it after this guard rather than panicking opaquely.
        let &r0 = live_r
            .first()
            .expect("reduce/contraction: residue must have a live cell to fold");
        // Operand wire offsets over the `out` x `residue` stream (`out` outer, residue inner). `out`
        // carries the kept axes (a broadcast axis the operand lacks reads stride 0, so its fan shares one
        // base) and `residue` the reduced axes, so together they consume the operand. `out` sequenced as
        // the outer stream means stream row `o` IS output-buffer offset `o`: `offset(o, r)` sits at stream
        // position `o * rwidth + r`.
        let config = sequence(&[memory], &[&out.clone().pair(residue.clone())], SequencerMode::Read)
            .expect("reduce/contraction: out x residue must factor the operand")
            .swap_remove(0);
        let stream_size = config.stream_size();
        // Deltas: the live residue cells of the first reachable output cell's row, each minus that row's
        // first live read. The first reachable cell is `o == 0` when `r0` is in range (the base position
        // `o * rwidth + r0` grows with `o`); otherwise nothing is reachable and the deltas are unused.
        let delta: Vec<usize> = if r0 < stream_size {
            let row: Vec<usize> = config.iter_range(0, rwidth.min(stream_size)).collect();
            live_r.iter().map(|&r| row[r] - row[r0]).collect()
        } else {
            Vec::new()
        };
        Self {
            config,
            delta,
            rwidth,
            r0,
            stream_size,
        }
    }

    /// The operand buffer positions folding into output cell `o`: its base (seeked from the stream) plus
    /// each live residue delta. `None` for a dead (padding) output cell the stream never reaches. A
    /// split-then-padded residue can still push an individual read past the buffer, so the caller
    /// `.get()`-guards each position.
    fn reads(&self, o: usize) -> Option<impl Iterator<Item = usize> + '_> {
        let pos = o * self.rwidth + self.r0;
        let base = (pos < self.stream_size).then(|| self.config.iter_range(pos, pos + 1).next())??;
        Some(self.delta.iter().map(move |&d| base + d))
    }

    /// Work per output cell (the live residue width), for the rayon `with_min_len` job-size hint.
    fn cell_work(&self) -> usize {
        self.delta.len()
    }
}

/// The physical wire base where the `offset` window starts in `base_map` (0 for an empty offset): the
/// wire position whose canonical offset equals the offset's. `base_map` is the storage's live-axis
/// mapping, so a tiled offset axis the relayout's `Src` / `Dst` *type* carries as padding is still a
/// live `Symbol` here and matches; the view shares the base's wire layout, so this position is also the
/// partial view's start in the buffer.
pub(crate) fn window_base(base_map: &MappingValue, offset: &Index) -> usize {
    if *offset == Index::new() {
        return 0;
    }
    // The offset's dense (canonical) offset under `base_map`: each axis's coordinate times its
    // canonical weight (innermost axis = 1). One `finalize` (inside `axis_coords`).
    let axes = base_map.axes();
    let coords = crate::storage::axis_coords(&axes, offset.clone())
        .expect("transpose: partial-view offset must be a live coordinate of its base mapping");
    let mut weight = 1;
    let mut target = 0;
    for (axis, &coord) in axes.iter().rev().zip(coords.iter().rev()) {
        target += coord * weight;
        weight *= axis.modulo;
    }
    // The wire position whose canonical offset is `target`, found by the lazy wire walk (`Some` = live,
    // padding never matches), replacing a `finalize` of every position with the cheap odometer.
    base_map
        .iter(&axes, &Index::new(), true)
        .position(|offset| offset == Some(target))
        .expect("transpose: partial-view offset must land on a live cell of its base mapping")
}

/// Lays `live` elements into a fresh `dst_map`-sized buffer at its live wire positions (padding stays
/// zero).
fn place_live_elems<D: Scalar>(dst_map: &MappingValue, mut live: impl Iterator<Item = D>) -> Vec<D> {
    let mut data = vec![D::zero(); dst_map.size()];
    for (elem, offset) in data.iter_mut().zip(dst_map.iter_positions()) {
        if offset.is_some() {
            *elem = live
                .next()
                .expect("transmute: dst_map has more live elements than src_map");
        }
    }
    data
}

/// Decodes a `BufStorage`-backed `i32` index tensor to element positions: each value divided by the
/// stride from [`decode_stride`] (a byte stride when scaled, else 1). `BufStorage` offsets are
/// physical and `data` already holds exactly the index mapping's cells, so the whole buffer is read.
/// Shared by scatter and gather.
fn decode_indices<B: Buf>(index: &BufStorage<i32, B>, index_stride: usize) -> Vec<usize> {
    // The index buffer holds exactly the index mapping's elements; read them in order (an `i32` decode is
    // a whole-value read, so this is the cheap byte-multiple path).
    (0..index.len())
        .map(|i| {
            let raw = index.get(i);
            let pos = usize::try_from(raw)
                .unwrap_or_else(|_| panic!("scatter/gather index at cell {i} must be non-negative, got {raw}"));
            pos / index_stride
        })
        .collect()
}

/// The divisor [`decode_indices`] applies: a scaled index tensor holds byte offsets into
/// payload-sized blocks (divide by the block's byte size); an unscaled one holds element indices.
fn decode_stride<D>(payload: &MappingValue, scaled: bool) -> usize {
    if scaled {
        payload.size() * std::mem::size_of::<D>()
    } else {
        1
    }
}

/// Least common multiple, for the byte-alignment quantum of a [`BufStorage::par_chunks_mut`] chunk.
fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `zeroed(n)` must allocate the same byte length as the `from_vec` path it replaced, for every
    /// dtype -- i.e. the host image size `load`/`store` address, not the `BITS`-based wire size. A
    /// staging type (`i5` an `i8`, `i9` an `i16`) has a host image wider than `n * BITS / 8`, so a
    /// `BITS`-based `zeroed` under-allocates it (the `collect` i9-transpose out-of-range this pins).
    #[test]
    fn zeroed_byte_len_matches_from_vec_for_every_dtype() {
        fn check<D: Scalar>(n: usize) {
            let zeroed = BufStorage::<D, Vec<u8>>::zeroed(n).as_bytes().len();
            let packed = BufStorage::<D, Vec<u8>>::from_vec(std::iter::repeat_n(<D as num_traits::Zero>::zero(), n))
                .as_bytes()
                .len();
            assert_eq!(
                zeroed,
                packed,
                "{}: zeroed({n}) allocated {zeroed} B but from_vec packs {packed} B",
                std::any::type_name::<D>(),
            );
        }
        for n in [2usize, 8, 128] {
            for check in [
                check::<i8>,
                check::<i16>,
                check::<i32>,
                check::<bf16>,
                check::<f32>,
                check::<i4>,
                check::<f4e2m1>,
                check::<i5>,
                check::<i9>,
            ] {
                check(n);
            }
        }
    }

    /// `len()` must recover the count that went in, for every dtype: it divides the byte length by
    /// `STAGING_BITS`, so a staging type whose wire `BITS` is narrower than its host form (`i5` an `i8`,
    /// `i9` an `i16`) has to round-trip too. Every whole-buffer walk (`par_iter` / `map` / `to_vec`)
    /// derives its element count from this, and a `BITS`-based count walked past the buffer.
    #[test]
    fn len_round_trips_the_element_count_for_every_dtype() {
        fn check<D: Scalar>(n: usize) {
            let buf = BufStorage::<D, Vec<u8>>::from_vec(std::iter::repeat_n(<D as num_traits::Zero>::zero(), n));
            let name = std::any::type_name::<D>();
            assert_eq!(buf.len(), n, "{name}: len() lost the count");
            assert_eq!(
                D::buf_bytes(n),
                buf.as_bytes().len(),
                "{name}: buf_bytes disagrees with the packer"
            );
        }
        for n in [2usize, 8, 128] {
            for check in [
                check::<i8>,
                check::<i16>,
                check::<i32>,
                check::<bf16>,
                check::<f32>,
                check::<i4>,
                check::<f4e2m1>,
                check::<i5>,
                check::<i9>,
            ] {
                check(n);
            }
        }
    }

    /// The walk a staging type could not take before: `map` over an `i9` buffer visits its four
    /// elements, where a `BITS`-derived count would have claimed eight and read past the buffer. This
    /// is the widen `contract_outer` runs on a zero-point-subtracted stream.
    #[test]
    fn map_walks_a_staging_type_within_its_buffer() {
        let buf = BufStorage::<_, Vec<u8>>::from_vec([1, -1, 255, -256].map(i9::from_i32));
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.map(|v| v.to_i32()).to_vec(), vec![1, -1, 255, -256]);
    }

    /// `BufStorage::contraction` reproduces a plain matmul. `[M,K] · [K,N] -> [M,N]` with no padding,
    /// so the wire buffers are row-major and the result is hand-checkable.
    #[test]
    fn contraction_matches_matmul() {
        axes![M = 2, K = 3, N = 2];
        // lhs row-major: [[1,2,3],[4,5,6]]; rhs row-major: [[1,2],[3,4],[5,6]].
        let lhs = BufStorage::<_, Vec<u8>>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = BufStorage::<_, Vec<u8>>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = BufStorage::<_, Vec<u8>>::contraction_prewidened(
            &lhs,
            &rhs,
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        // [[1·1+2·3+3·5, 1·2+2·4+3·6], [4·1+5·3+6·5, 4·2+5·4+6·6]] = [[22,28],[49,64]].
        assert_eq!(out.to_vec(), vec![22.0f32, 28.0, 49.0, 64.0]);
    }

    /// A `SequencerConfig`'s walk (the parallel-`contraction`/`reduce` offset source): range pieces
    /// concatenate to the full `iter`, and `next_back` reverses it. Guards the seek / double-ended paths.
    #[test]
    fn sequencer_range_and_back_match_full() {
        axes![M = 4, K = 6, N = 4];
        let cfg = sequence(
            &[&<m![M, N]>::to_value()],
            &[&<m![M, K, N]>::to_value()],
            SequencerMode::Read,
        )
        .unwrap();
        let config = &cfg[0];
        let full: Vec<usize> = config.iter().collect();
        let n = config.stream_size();
        assert_eq!(n, full.len());
        for split in [0, 1, n / 2, n] {
            let mid = split.min(n);
            let mut concat: Vec<usize> = config.iter_range(0, mid).collect();
            concat.extend(config.iter_range(mid, n));
            assert_eq!(concat, full, "iter_range split at {mid}");
        }
        let mut reversed = full.clone();
        reversed.reverse();
        assert_eq!(config.iter().rev().collect::<Vec<_>>(), reversed, "next_back");
    }

    /// A reduce big enough (> `PAR_MIN_JOB`) that rayon actually splits the walk across workers. `i32`
    /// add is associative, so the regrouped parallel sum equals the serial one and is deterministic —
    /// the sub-`PAR_MIN_JOB` unit tests never exercise the split itself.
    #[test]
    fn reduce_large_split_matches_serial() {
        axes![R = 256, C = 512]; // 131072 stream positions > PAR_MIN_JOB (65536)
        let data: Vec<i32> = (0..256 * 512).map(|i| i % 5).collect();
        let out = BufStorage::<_, Vec<u8>>::from_vec(data.clone()).reduce::<m![R, C], m![C], _>(|a, b| a + b, 0, false);
        let mut expected = vec![0i32; 512];
        for r in 0..256 {
            for c in 0..512 {
                expected[c] += data[r * 512 + c];
            }
        }
        assert_eq!(out.to_vec(), expected);
    }

    /// The `out_size ≫ contracted` corner the per-output-cell rewrite targets: a huge output (> 1M cells)
    /// reduced over a tiny (4-cell) axis. The old per-worker-accumulator fold allocated and combined a
    /// `vec![identity; out_size]` per leaf here, `O(out_size × leaves)` — far more than the real work.
    /// The per-cell fold writes each output cell once. Pins that the unified path stays correct (and the
    /// bench `bench/storage.rs::reduce_wide_thin` pins it is not slower than serial).
    #[test]
    fn reduce_wide_out_thin_contracted_matches_serial() {
        axes![Big = 1048576, Small = 4]; // out = 2^20 cells, contracted = 4
        let data: Vec<i32> = (0..(1 << 20) * 4).map(|i| i % 7).collect();
        let out = BufStorage::<_, Vec<u8>>::from_vec(data.clone()).reduce::<m![Big, Small], m![Big], _>(
            |a, b| a + b,
            0,
            false,
        );
        let mut expected = vec![0i32; 1 << 20];
        for big in 0..(1 << 20) {
            for small in 0..4 {
                expected[big] += data[big * 4 + small];
            }
        }
        assert_eq!(out.to_vec(), expected);
    }

    /// A contraction whose output-cell count comfortably exceeds the per-job floor, so rayon splits the
    /// per-cell fold across workers (the floor is `min_cells_per_job(K=4) = PAR_MIN_JOB/4 = 16384`, so
    /// `M*N = 128*512 = 65536` cells give ~4 jobs — not a single serial run). `i32` products / sums are
    /// associative, so the per-cell result equals the serial reference exactly.
    #[test]
    fn contraction_large_split_matches_serial() {
        axes![M = 128, K = 4, N = 512]; // 65536 output cells >> 16384 floor
        let lhs: Vec<i32> = (0..128 * 4).map(|i| i % 5).collect();
        let rhs: Vec<i32> = (0..4 * 512).map(|i| i % 3).collect();
        let out = BufStorage::<_, Vec<u8>>::contraction_prewidened(
            &BufStorage::<_, Vec<u8>>::from_vec(lhs.clone()),
            &BufStorage::<_, Vec<u8>>::from_vec(rhs.clone()),
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        let mut expected = vec![0i32; 128 * 512];
        for m in 0..128 {
            for n in 0..512 {
                for k in 0..4 {
                    expected[m * 512 + n] += lhs[m * 4 + k] * rhs[k * 512 + n];
                }
            }
        }
        assert_eq!(out.to_vec(), expected);
    }

    /// A byte-multiple `transpose` over a buffer larger than `PAR_MIN_JOB`, so rayon actually splits the
    /// dst walk and many workers run the byte-partitioned `BufStorage::par_chunks_mut` concurrently (the small
    /// concrete_tests transposes stay one chunk, never exercising the
    /// concurrency the disjoint chunks rest on). `[R=512, C=256]` -> `[C, R]` is 131072 `i32` elements >
    /// `PAR_MIN_JOB` (65536); the parallel result must equal the serial transpose oracle.
    #[test]
    fn transpose_large_split_matches_serial() {
        axes![R = 512, C = 256]; // 131072 cells > PAR_MIN_JOB (65536)
        let src_data: Vec<i32> = (0..512 * 256).collect();
        let src = BufStorage::<_, Vec<u8>>::from_vec(src_data.clone());
        let mut dst = BufStorage::<_, Vec<u8>>::from_vec(vec![0i32; 512 * 256]);
        dst.transpose::<m![R, C], m![C, R]>(
            &src,
            &Index::new(),
            &Index::new(),
            &<m![R, C]>::to_value(),
            &<m![C, R]>::to_value(),
            false,
        );
        // Serial oracle: out[c*R + r] = src[r*C + c].
        let mut expected = vec![0i32; 512 * 256];
        for r in 0..512 {
            for c in 0..256 {
                expected[c * 512 + r] = src_data[r * 256 + c];
            }
        }
        assert_eq!(dst.to_vec(), expected);
    }

    /// `BufStorage::contraction` with a REORDERED output (`Out = m![N, M]`, not the canonical `[M, N]`):
    /// the per-cell `ReadPlan` indexes the base table by `Out`'s own buffer order, so the result must
    /// read back in `[N, M]` order. Guards that `ReadPlan` does not assume canonical out order.
    #[test]
    fn contraction_reordered_out_matches_serial() {
        axes![M = 2, K = 3, N = 2];
        let lhs = BufStorage::<_, Vec<u8>>::from_vec(vec![1i32, 2, 3, 4, 5, 6]); // [M,K] row-major
        let rhs = BufStorage::<_, Vec<u8>>::from_vec(vec![1i32, 2, 3, 4, 5, 6]); // [K,N] row-major
        let out = BufStorage::<_, Vec<u8>>::contraction_prewidened(
            &lhs,
            &rhs,
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![N, M]>::to_value(),
        );
        // The (M,N) product [[22,28],[49,64]] read back N-major: [22,49,28,64].
        assert_eq!(out.to_vec(), vec![22i32, 49, 28, 64]);
    }

    /// `BufStorage::contraction` with a PADDED operand axis (`K = 2 # 4`: K live 2, wire extent 4). The
    /// contracted residue carries the `# 4` Top pad; the live-residue mask must fold only the 2 live K
    /// cells (the padded positions, if folded, would read stale wire cells). Pins the padding guard on
    /// the contraction path (the reduce peer is `reduce_padded_source_math_and_buf`).
    #[test]
    fn contraction_padded_contracted_axis_matches_serial() {
        axes![M = 2, K = 2, N = 2];
        // lhs m![M, K = 2 # 4]: each M row is [k0, k1, pad, pad] in wire order (size 2*4 = 8).
        let lhs = BufStorage::<_, Vec<u8>>::from_vec(vec![1i32, 2, 0, 0, 3, 4, 0, 0]);
        // rhs m![K = 2 # 4, N]: each of the 4 K wire rows holds [n0, n1]; rows 2,3 are pad (size 4*2 = 8).
        let rhs = BufStorage::<_, Vec<u8>>::from_vec(vec![1i32, 2, 3, 4, 0, 0, 0, 0]);
        let out = BufStorage::<_, Vec<u8>>::contraction_prewidened(
            &lhs,
            &rhs,
            &<m![M, K = 2 # 4]>::to_value(),
            &<m![K = 2 # 4, N]>::to_value(),
            &<m![M, K = 2 # 4, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        // Only the 2 live K cells fold: out[m,n] = sum_{k<2} lhs[m,k] * rhs[k,n].
        // [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7,10],[15,22]].
        assert_eq!(out.to_vec(), vec![7i32, 10, 15, 22]);
    }

    /// `BufStorage::contraction` with a BROADCAST operand: `rhs` (`m![N]`) lacks `M`, so it broadcasts
    /// across the M output rows (its base / M-stride is 0, one rhs cell feeds every M). Pins the
    /// broadcast branch described in the contraction doc (an axis the operand lacks).
    #[test]
    fn contraction_broadcast_operand_matches_serial() {
        axes![M = 2, K = 3, N = 2];
        let lhs_data = vec![1i32, 2, 3, 4, 5, 6]; // [M,K] row-major
        let rhs_data = vec![1i32, 2, 3, 4, 5, 6]; // [K,N] row-major, broadcasts over M
        let out = BufStorage::<_, Vec<u8>>::contraction_prewidened(
            &BufStorage::<_, Vec<u8>>::from_vec(lhs_data.clone()),
            &BufStorage::<_, Vec<u8>>::from_vec(rhs_data.clone()),
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        let mut expected = vec![0i32; 4];
        for m in 0..2 {
            for n in 0..2 {
                for k in 0..3 {
                    expected[m * 2 + n] += lhs_data[m * 3 + k] * rhs_data[k * 2 + n];
                }
            }
        }
        assert_eq!(out.to_vec(), expected);
    }

    /// The integer half: an `i8` `K = 100` dot of `100 * 100` holds `1_000_000` in `i32` and wraps
    /// `as i8` only at the final narrow, where an `i8`-wide fold would overflow on the first product.
    #[test]
    fn contraction_i8_accumulates_wide_then_wraps_once() {
        axes![M = 1, K = 100, N = 1];
        let out = BufStorage::<_, Vec<u8>>::contraction(
            &BufStorage::<_, Vec<u8>>::from_vec(vec![100i8; 100]),
            &BufStorage::<_, Vec<u8>>::from_vec(vec![100i8; 100]),
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        assert_eq!(out.get(0), 1_000_000i32 as i8);
    }

    /// A `bf16` `K = 256` contraction accumulates in `f32`, recovering small addends a narrow per-step
    /// accumulator drops: `lhs = [1.0, 1/128, .., 1/128]` against `rhs = ones` (both exact in bf16, so the
    /// only error is accumulation rounding) sums to `1.0 + 255*(1/128)`. A bf16 running sum saturates near
    /// `2.0` and swallows every later `1/128`; the `f32` accumulator keeps them. No padding, so the wire
    /// buffers are the raw vecs.
    #[test]
    fn contraction_bf16_recovers_small_addends_via_wide_accumulator() {
        axes![M = 1, K = 256, N = 1];
        let lhs: Vec<bf16> = std::iter::once(1.0)
            .chain(std::iter::repeat_n(1.0 / 128.0, 255))
            .map(bf16::from_f32)
            .collect();
        let rhs = vec![bf16::from_f32(1.0); 256];
        let out = BufStorage::<_, Vec<u8>>::contraction(
            &BufStorage::<_, Vec<u8>>::from_vec(lhs),
            &BufStorage::<_, Vec<u8>>::from_vec(rhs),
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        let got = out.get(0).to_f32();
        // Exact f32 reference 1.0 + 255*(1/128); bf16 has 7 mantissa bits so 1 ulp = reference * 2^-7.
        let reference = 2.9921875_f32;
        assert!(
            (got - reference).abs() <= reference * f32::exp2(-7.0),
            "f32 accumulator should recover small addends bf16 drops: got={got}"
        );
    }

    /// The `Acc -> D` narrow happens once per `contraction` (the Lane Folder narrow), so a cross-slice
    /// reduction over the narrowed per-slice partials differs from one global wide fold. The hardware
    /// narrows each per-slice partial at its own Lane Folder and the Vector Engine sums the narrowed
    /// partials; the sim composes that as three per-slice `BufStorage::contraction`s feeding a downstream
    /// `BufStorage::reduce`.
    ///
    /// Each per-slice dot is `[1.0, 2^-8] . [1.0, 1.0] = 1.0 + 2^-8`, which the per-slice narrow rounds
    /// to `1.0` (`2^-8` is half a bf16 ulp at magnitude 1, ties-to-even). The cross-slice `reduce` over
    /// the three `1.0`s gives `3.0`. A global wide fold is the same six MACs as one `K=6` contraction: it
    /// keeps `3.0 + 3·2^-8` in `f32` and narrows once to `bf16(3.015625)`. The two outputs differ, so the
    /// narrow is per slice, not held wide across the cross-slice reduce.
    #[test]
    fn cross_slice_reduce_narrows_per_lane_fold_not_globally() {
        axes![Slice = 3, M = 1, K = 2, N = 1, GlobalK = 6];
        // Per-slice path: each slice is a within-slice contraction that narrows to bf16 at its lane fold.
        let lhs_slice =
            || BufStorage::<_, Vec<u8>>::from_vec(vec![bf16::from_f32(1.0), bf16::from_f32(2.0f32.powi(-8))]);
        let rhs_slice = || BufStorage::<_, Vec<u8>>::from_vec(vec![bf16::from_f32(1.0); 2]);
        let partial = BufStorage::<_, Vec<u8>>::contraction(
            &lhs_slice(),
            &rhs_slice(),
            &<m![M, K]>::to_value(),
            &<m![K, N]>::to_value(),
            &<m![M, K, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        let partial0 = partial.get(0);
        assert_eq!(
            partial0.to_f32(),
            1.0,
            "per-slice narrow rounds 1 + 2^-8 to 1.0 in bf16"
        );
        // The Vector Engine sums the three narrowed per-slice partials downstream (a `reduce`).
        let three_narrowed = BufStorage::<_, Vec<u8>>::from_vec(vec![partial0; 3]);
        let cross_slice = three_narrowed
            .reduce::<m![Slice, M, N], m![M, N], _>(|a, b| a + b, bf16::from_f32(0.0), false)
            .get(0)
            .to_f32();
        assert_eq!(cross_slice, 3.0);
        // One global wide fold: the same six MACs as a single K=6 contraction, narrowing once at the end.
        let global = BufStorage::<_, Vec<u8>>::contraction(
            &BufStorage::<_, Vec<u8>>::from_vec([1.0, 2.0f32.powi(-8)].into_iter().cycle().take(6).map(bf16::from_f32)),
            &BufStorage::<_, Vec<u8>>::from_vec(vec![bf16::from_f32(1.0); 6]),
            &<m![M, GlobalK]>::to_value(),
            &<m![GlobalK, N]>::to_value(),
            &<m![M, GlobalK, N]>::to_value(),
            &<m![M, N]>::to_value(),
        );
        let global0 = global.get(0).to_f32();
        assert_eq!(global0, 3.015625, "global wide fold keeps 3 + 3·2^-8, narrows once");
        assert_ne!(
            cross_slice, global0,
            "per-slice Lane Folder narrow must differ from a global narrow across the cross-slice reduce"
        );
    }

    /// f32 reduce over a `> PAR_MIN_JOB` split: `f32` add is non-associative, so the reassociated parallel
    /// fold need only AGREE WITH SERIAL TO TOLERANCE, not bit-exactly (the documented Cpu
    /// non-reproducibility). The tolerance is tight (`1e-4` relative): reassociating 256 well-conditioned
    /// positive adds drifts only ~`n·eps ≈ 1.5e-5`, so `1e-4` passes genuine reassociation yet still
    /// catches a dropped / double-counted term (~0.4%+), which the associative `i32` tests cannot.
    #[test]
    fn reduce_large_split_f32_close_to_serial() {
        axes![R = 256, C = 512];
        let data: Vec<f32> = (0..256 * 512).map(|i| (i % 13) as f32 * 0.5 + 0.25).collect();
        let out =
            BufStorage::<_, Vec<u8>>::from_vec(data.clone()).reduce::<m![R, C], m![C], _>(|a, b| a + b, 0.0, false);
        // f64-accumulated reference, so the tight f32 bound measures only the parallel fold's drift.
        let mut serial = vec![0.0f64; 512];
        for r in 0..256 {
            for c in 0..512 {
                serial[c] += data[r * 512 + c] as f64;
            }
        }
        for (got, want) in out.to_vec().iter().zip(&serial) {
            assert!(
                (*got as f64 - want).abs() <= 1e-4 * want.abs().max(1.0),
                "got {got}, serial {want}"
            );
        }
    }
}
