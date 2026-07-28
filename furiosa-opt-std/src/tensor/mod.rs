use std::fmt;

use furiosa_mapping::*;
use num_traits::Zero;
use rand::Rng;
use rand::distr::StandardUniform;

use self::view::*;
use crate::scalar::*;

use crate::backend::Backend;
use crate::cast::ContractionCast;
use crate::runtime::CurrentBackend;

pub(crate) mod memory;
pub mod pseudo;
pub(crate) mod tu;
pub(crate) mod view;

/// Tensor with scalar type `D`, axes determined by `Mapping`, and backend determined by `B`.
///
/// A thin newtype over the backend's concrete tensor type ([`Backend::Storage`]). `B` defaults to
/// [`CurrentBackend`], a cfg-aliased type that picks
/// Emulation / Npu / Typecheck. User code writes `Tensor<D, M>` and gets the
/// backend-appropriate storage automatically. Explicit `Tensor<D, M, SomeBackend>` overrides for
/// testing / cross-backend code.
pub struct Tensor<D: Scalar, Mapping: M, B: Backend = CurrentBackend> {
    pub(crate) inner: B::Storage<D>,
    // `B::Storage<D>` drops `Mapping` at the type level (storage-layer erasure); this phantom keeps
    // the layout traveling at the `Tensor` level so memory-tier structs stay generic and inferable.
    pub(crate) _marker: std::marker::PhantomData<Mapping>,
}

impl<D: Scalar, Mapping: M, B: Backend> fmt::Debug for Tensor<D, Mapping, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor").field("inner", &self.inner).finish()
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Clone for Tensor<D, Mapping, B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D: Scalar, Mapping: M, B: Backend> PartialEq for Tensor<D, Mapping, B> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Wraps a backend-native concrete tensor in `Tensor`.
    pub(crate) fn from_inner(inner: B::Storage<D>) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }

    /// Consumes the tensor and returns its underlying backend-native concrete tensor.
    pub fn into_inner(self) -> B::Storage<D> {
        self.inner
    }

    /// Creates a new tensor from an initialized buffer. Panics if the buffer length does not match
    /// `Mapping::SIZE`.
    pub fn from_vec(data: impl IntoIterator<Item = D>) -> Self {
        // The length check is backend-independent `Mapping::SIZE` arithmetic, so it lives here in the
        // wrapper rather than on the construct-only `Backend` seam.
        let data: Vec<D> = data.into_iter().collect();
        assert_eq!(
            data.len(),
            Mapping::SIZE,
            "from_vec: buffer length does not match mapping size"
        );
        Self::from_inner(B::from_vec(&Mapping::to_value(), data))
    }

    /// Creates a tensor from a pre-packed device byte image (the packed buffer, [`Self::into_buf`]'s
    /// inverse). Contrast [`Self::from_vec`], which packs logical values: pre-packed data (fp4 / f4e2m1
    /// weights) comes through here so the bytes are stored as-is, not decoded to a logical `Vec` and
    /// re-packed. Panics if the byte length does not match the mapping's packed size.
    pub fn from_buf(buf: Vec<u8>) -> Self {
        assert_eq!(
            buf.len() * 8,
            Mapping::SIZE * D::BITS,
            "from_buf: byte length does not match the mapping's packed size"
        );
        Self::from_inner(B::from_buf(&Mapping::to_value(), buf))
    }

    /// Returns the tensor data as a flat `Vec<D>`, consuming the tensor.
    pub fn into_vec(self) -> Vec<D>
    where
        D: MaterializableScalar,
    {
        B::into_vec(self.inner, &Mapping::to_value())
    }

    /// Returns the tensor's dense physical device byte image (packed on `D::BITS`), consuming the
    /// tensor. Delegates to the backend's [`Backend::into_buf`]; Emulation / Npu move the packed
    /// `BufStorage` bytes out directly.
    pub fn into_buf(self) -> Vec<u8>
    where
        D: MaterializableScalar,
    {
        B::into_buf(self.inner, &Mapping::to_value())
    }

    /// Maps the tensor element-wise to a new scalar; same layout. The bare-scalar closure is lifted
    /// by the backend's own [`Backend::map`] strategy (Emulation/Npu a bare buffer walk, Typecheck a
    /// no-op), no offset-uniform primitive imposed across storages.
    ///
    /// Only valid for a `D` whose storage-native length recovery is exact -- every
    /// [`crate::scalar::MaterializableScalar`] type, where `Scalar::BITS` always agrees with what
    /// `to_buf`/`load`/`store` actually pack per element. A non-materializable staging type (`i5`/`i9`)
    /// does NOT hold that invariant (`BITS` names a hardware-only wire width, disconnected from its host
    /// `size_of`-based default `load`/`store`) and must use [`Self::map_bounded`] instead -- see that
    /// method's doc for why this one is unsound for it.
    pub fn map<D2: Scalar, F>(&self, f: F) -> Tensor<D2, Mapping, B>
    where
        F: Fn(D) -> D2 + Sync,
        D: MaterializableScalar,
    {
        Tensor::from_inner(B::map(&self.inner, f))
    }

    /// [`Self::map`], but bounded by this tensor's own `Mapping::SIZE` instead of trusting the backend
    /// storage's own length recovery.
    ///
    /// For most `D` this is equivalent to [`Self::map`] (redundant, so prefer the simpler `map`). It
    /// exists for a non-`MaterializableScalar` staging type (`i5`/`i9`, the zero-point-subtracted
    /// contraction operands): such a `D` is deliberately never routed through any device image, HBM, or
    /// DM (`MaterializableScalar` gates those entry points), so its `Scalar::BITS` describes only a
    /// real-hardware wire width -- for `i9` that's 8 bits, one physical byte, because the extra bit is
    /// produced inside the contraction engine at consumption time and never actually crosses a wire or
    /// sits in device memory. Host-side (Emulation), `i5`/`i9` still need a REAL, full-precision value
    /// (`fetch_zero_point_sub` computes the exact zero-point-subtracted result eagerly), so they stay in
    /// their full in-memory representation (`i9` is `i16`-backed) -- neither overrides `load`/`store`/
    /// `to_buf`, so those default to `size_of::<Self>()`, twice `BITS`'s 8-bit claim. `self.len()`
    /// (used by [`Self::map`] via [`crate::storage::buf::BufStorage::par_iter`]) derives the element
    /// count from the buffer's byte length using `BITS`, so for `i9` it silently reports DOUBLE the
    /// true count, and an unpaired whole-buffer walk (this widen has no shorter sibling to truncate it,
    /// unlike a `zip_with`) reads/writes past the buffer once the count crosses the true half.
    /// `Mapping::SIZE` is the type-level truth regardless of any of that, so bound the walk on it here.
    pub(crate) fn map_bounded<D2: Scalar, F>(&self, f: F) -> Tensor<D2, Mapping, B>
    where
        F: Fn(D) -> D2 + Sync,
    {
        Tensor::from_inner(B::map_bounded(&self.inner, Mapping::SIZE, f))
    }

    /// Zips two same-layout tensors element-wise. Delegates to the backend's [`Backend::zip_with`];
    /// the result is defined only where both inputs are.
    pub fn zip_with<D2: MaterializableScalar, D3: Scalar, F>(
        &self,
        other: &Tensor<D2, Mapping, B>,
        f: F,
    ) -> Tensor<D3, Mapping, B>
    where
        F: Fn(D, D2) -> D3 + Sync,
        D: MaterializableScalar,
    {
        Tensor::from_inner(B::zip_with(&self.inner, &other.inner, f))
    }

    /// Performs reduction (sum) over axes not present in `Dst` (strict: no broadcast).
    pub fn reduce_add<Dst: M>(&self) -> Tensor<D, Dst, B>
    where
        D: Zero + MaterializableScalar,
    {
        self.reduce::<Dst>(|a, b| a + b, D::zero(), false)
    }

    /// Reduces the factors of `self`'s mapping that are absent in `Dst`. Axes that `Dst` adds beyond
    /// the reduced source are broadcast (the reduced value repeats across them); `allow_broadcast`
    /// gates that, `false` is a plain reduce that requires `Dst` to be a pure factor and rejects an
    /// unintended extra axis, `true` permits the broadcast (like [`Self::transpose`]'s flag).
    ///
    /// The reduce fn works over bare scalars; the backend lifts it to its cell representation.
    /// Emulation/Npu have no undefined cells.
    pub fn reduce<Dst: M>(
        &self,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Tensor<D, Dst, B>
    where
        D: MaterializableScalar,
    {
        Tensor::from_inner(B::reduce::<D, Mapping, Dst>(
            &self.inner,
            reduce_fn,
            identity,
            allow_broadcast,
        ))
    }

    /// Creates a tensor with every logical cell set to `value`, a scalar broadcast ("splat").
    /// Padding cells stay uninitialized.
    pub fn splat(value: D) -> Self {
        Self::from_vec(std::iter::repeat_n(value, Mapping::SIZE))
    }

    /// Creates a tensor with every logical cell set to zero (every [`Scalar`] is `Zero` via `Num`).
    pub fn zero() -> Self {
        Self::splat(num_traits::Zero::zero())
    }

    /// Creates a random tensor.
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Self::from_vec((0..Mapping::SIZE).map(|_| rng.random::<D>()))
    }

    /// Performs contraction between two tensors: a generalized matmul that multiplies `lhs` and `rhs`
    /// over the shared `Union` index space and reduces the axes absent from the output `Mapping`.
    /// Delegates to [`Backend::contraction`], which the host backend fuses into a single accumulating
    /// pass (Emulation does, so no full `Union` outer product is materialized).
    pub fn contraction<Union: M, Lhs: M, Rhs: M>(lhs: &Tensor<D, Lhs, B>, rhs: &Tensor<D, Rhs, B>) -> Self
    where
        D: ContractionCast + MaterializableScalar,
    {
        Tensor::from_inner(B::contraction(
            &lhs.inner,
            &rhs.inner,
            &Lhs::to_value(),
            &Rhs::to_value(),
            &Union::to_value(),
            &Mapping::to_value(),
        ))
    }
}

/// Host-side tensor methods.
impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// The empty destination canvas for the relayout ops (transpose / scatter / gather / reduce /
    /// reshape), which fill only the live cells; cells no op writes (padding and any gaps) are
    /// don't-care. `pub(crate)`: no public constructor hands out an undefined-value tensor (an
    /// external caller that wants zeros uses `splat`/`zero`).
    pub(crate) fn zeroed() -> Self {
        // Replace zeroed with uninit (zero-cost) once MaybeUninit rework of Buf lands.
        Self::from_inner(B::zeroed(&Mapping::to_value()))
    }

    /// Creates a mutable view of the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Mapping, B> {
        TensorViewMut::new(&mut self.inner)
    }

    /// Creates an immutable view of the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Mapping, B> {
        TensorView::new(&self.inner)
    }

    /// Rewraps the tensor under a different mapping.
    ///
    /// After the `Mapping`-type-parameter erasure of the storage layer, `B::Storage<D>` no longer
    /// carries the mapping at the type level, so the inner storage of `Tensor<D, Mapping, B>` and
    /// `Tensor<D, Mapping2, B>` is literally the same type, rewrapping it is a plain, safe move.
    ///
    /// Deliberately NOT size-checked: `transmute` is used to relabel a padded read down to its
    /// compact slot (the vector engine does `…tile(k).read().transmute()`, shrinking the mapping),
    /// so `Mapping::SIZE != Mapping2::SIZE` is a legitimate, intended case. Delegates to
    /// [`Backend::transmute`], which COMPACTS the physical `BufStorage` buffer (drops the padding
    /// the relabel removes).
    pub fn transmute<Mapping2: M>(self) -> Tensor<D, Mapping2, B>
    where
        D: MaterializableScalar,
    {
        Tensor::from_inner(B::transmute::<D, Mapping, Mapping2>(
            self.inner,
            &Mapping::to_value(),
            &Mapping2::to_value(),
        ))
    }

    /// Reshapes the tensor to a different mapping by reinterpreting its buffer under `Mapping2`
    /// (delegates to [`Backend::reshape`]). Relabels axes over an unchanged buffer; the element count
    /// is checked (`Mapping::SIZE == Mapping2::SIZE`, asserted below).
    ///
    /// Kept a *runtime* `assert_eq!`, not a `const` block: every
    /// `{Dm,Hbm}Tensor(View/ViewMut)::reshape` delegates its `Element`-inclusive combined mapping here
    /// (their own per-axis checks stop at `Chip`/`Cluster`/`Slice` -- see
    /// [`constraints::assert_reshape_dimension_preserved`]'s TODO: some current examples reshape with
    /// a mismatched `Element`). A `const` block is evaluated at monomorphization regardless of whether
    /// the call is ever reached at runtime, so making this check static would trip on those dead-but-
    /// compiled paths too, not just the ones actually exercised.
    ///
    /// # Safety
    ///
    /// `Mapping` and `Mapping2` must lay the elements out in the SAME physical (wire) order, so the
    /// element at each wire position is unchanged by the relabel. Axis regrouping (merge/split)
    /// preserves wire order and is valid; a permutation like `m![A, B]` → `m![B, A]` does NOT, that
    /// reorders elements and must use `transpose`. Equal sizes alone do not guarantee this.
    pub unsafe fn reshape<Mapping2: M>(&self) -> Tensor<D, Mapping2, B> {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        Tensor::from_inner(B::reshape::<D, Mapping, Mapping2>(&self.inner))
    }

    /// Performs transpose. The mapping division is validated inside
    /// [`Backend::transpose`] (via `transpose_broadcast`), so this generic body works
    /// for every backend.
    pub fn transpose<Dst: M>(&self, allow_broadcast: bool) -> Tensor<D, Dst, B> {
        let mut dst = Tensor::zeroed();
        dst.view_mut().transpose(self.view(), allow_broadcast);
        dst
    }

    /// Scatter elements from self into dst at positions given by index tensor. Delegates to
    /// [`Backend::scatter`].
    pub fn scatter<Key: M, Dst: M, Idx: M>(
        &self,
        dst: &mut Tensor<D, Dst, B>,
        index: &Tensor<i32, Idx, B>,
        scaled: bool,
    ) {
        B::scatter::<D, Mapping, Key, Dst, Idx>(&self.inner, &mut dst.inner, &index.inner, scaled);
    }

    /// Gather elements from self (table) into dst at positions given by index tensor. Delegates
    /// to [`Backend::gather`].
    pub fn gather<Dst: M, Idx: M>(&self, dst: &mut Tensor<D, Dst, B>, index: &Tensor<i32, Idx, B>, scaled: bool)
    where
        D: MaterializableScalar,
    {
        B::gather::<D, Mapping, Dst, Idx>(&self.inner, &mut dst.inner, &index.inner, scaled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Emulation, Typecheck};

    /// Owned `Tensor::reshape` REBUILDS the storage under the new axes (`MathStorage::reshape` /
    /// `BufStorage::reshape`), so it supports an arbitrary wire-order-preserving RELABEL — even to
    /// fresh symbols a borrowed view could not resolve against its base `axes`. This is exactly the
    /// mechanism `HbmTensor::reshape` delegates to. `m![A, B]` -> `m![C]` (flatten to a fresh symbol)
    /// -> `m![B, A]` (regroup); each step preserves wire order, so the flat buffer is unchanged.
    #[test]
    fn owned_reshape_relabels_to_fresh_symbols_preserving_wire_order() {
        axes![A = 4, B = 6, C = 24];

        let buf: Vec<i32> = (0..24).collect();
        let t = Tensor::<i32, m![A, B], Emulation>::from_vec(buf.clone());

        let flat: Tensor<i32, m![C], Emulation> = unsafe { t.reshape::<m![C]>() };
        assert_eq!(flat.clone().into_vec(), buf);

        let regrouped: Tensor<i32, m![B, A], Emulation> = unsafe { flat.reshape::<m![B, A]>() };
        assert_eq!(regrouped.into_vec(), buf);
    }

    /// The one runtime guard the `unsafe` reshape enforces is `Mapping::SIZE == Mapping2::SIZE`; an
    /// element-count change must panic on that `assert_eq!`, not silently reinterpret a mismatched
    /// buffer. Pins the guard so a refactor cannot drop it.

    #[test]
    fn typecheck_from_vec_validates_length() {
        axes![A = 2];

        // The wrapper length-validates on every backend, Typecheck included.
        let tensor = Tensor::<i32, m![A], Typecheck>::from_vec(vec![1, 2]);
        assert!(tensor.clone().into_vec().is_empty());
    }

    #[test]
    fn typecheck_to_buf_is_empty() {
        axes![A = 2];

        let tensor = Tensor::<i32, m![A], Typecheck>::empty();

        assert!(tensor.clone().into_vec().is_empty());
    }

    /// Composite `MappingIter` over a *reordering* composite: `[B, A]` packs B-major (wire position
    /// `p` decodes to `b = p / 3`, `a = p % 3`), but the canonical layout sorts A-major, so the
    /// composite table is genuinely non-identity, unlike `[A, B] # N`, where wire order already
    /// equals canonical. The expected literal is hand-derived: wire `p = 3b + a` lands at canonical
    /// `4a + b`.
    #[test]
    fn emulation_composite_transpose_reorders() {
        axes![A = 3, B = 4];
        let data: Vec<i32> = (0..12).collect();
        let expected = vec![0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];

        let emu = Tensor::<i32, m![[B, A] # 12], Emulation>::from_vec(data);
        let emu_t: Tensor<i32, m![A, B], Emulation> = emu.transpose(false);
        assert_eq!(emu_t.into_vec(), expected, "Emulation (sequencer) composite reorder");
    }

    /// `write_gather` round-trip, `scaled=false` (raw row-position indices): gather a small table
    /// by a list of indices and confirm we get table rows back in indexed order. Mirrors the
    /// inverse-of-scatter contract documented on `HbmTensor::dma_gather`. The `scaled=true` path is
    /// covered by [`emulation_write_gather_roundtrip_scaled`].
    #[test]
    fn emulation_write_gather_roundtrip_unscaled() {
        // table: [W=3, V=2] = [[10,11], [20,21], [30,31]]. gather-key = W.
        // indices select rows 0, 2, 1, 0. output: [K=4, V=2].
        axes![W = 3, V = 2, K = 4];

        let table = Tensor::<i32, m![W, V], Emulation>::from_vec(vec![10, 11, 20, 21, 30, 31]);
        let index = Tensor::<i32, m![K], Emulation>::from_vec(vec![0, 2, 1, 0]);
        let mut output = Tensor::<i32, m![K, V], Emulation>::zeroed();
        table.gather::<_, _>(&mut output, &index, false);

        assert_eq!(output.into_vec(), vec![10, 11, 30, 31, 20, 21, 10, 11]);
    }

    /// Typecheck-backend smoke: gather should propagate mapping checks (via
    /// `gather_params`) without iterating any buffer. The output tensor under Typecheck
    /// has no values; we only verify that the call doesn't panic for a well-formed shape.
    #[test]
    fn typecheck_write_gather_runs_assertion_only() {
        axes![W = 3, V = 2, K = 4];

        let table = Tensor::<i32, m![W, V], Typecheck>::empty();
        let index = Tensor::<i32, m![K], Typecheck>::empty();
        let mut output = Tensor::<i32, m![K, V], Typecheck>::empty();
        table.gather::<_, _>(&mut output, &index, true);
    }

    #[test]
    fn emulation_into_vec_round_trips() {
        axes![A = 2];

        let tensor = Tensor::<i32, m![A], Emulation>::from_vec(vec![1, 2]);

        assert_eq!(tensor.into_vec(), vec![1, 2]);
    }

    // ---- Emulation backend: physical `BufStorage` buffer semantics ----
    //
    // These pin `B = Emulation` via the type parameter, so they run under the default
    // (emulation-cfg) `cargo test` without a backend rebuild, computed over the physical
    // `Mapping::SIZE` staging buffer.

    /// `from_vec`/`to_vec` round-trip: the staging buffer is stored verbatim in `Mapping`-order.
    #[test]
    fn emulation_from_vec_round_trips() {
        axes![A = 4];

        let tensor = Tensor::<i32, m![A], Emulation>::from_vec(vec![5, 6, 7, 8]);

        assert_eq!(tensor.into_vec(), vec![5, 6, 7, 8]);
    }

    /// `from_buf` stores a pre-packed byte image verbatim, no decode + re-pack: an `f4e2m1` packed buffer
    /// (two codes per byte) round-trips through `into_buf` BIT-IDENTICALLY and decodes back to the
    /// original codes. This is the fp4 weight-load path; pre-packed nibbles must not detour through a
    /// logical `Vec` (which would repack to canonical bits and could differ from the source image).
    #[test]
    fn emulation_from_buf_fp4_is_bit_identical() {
        axes![C = 4];
        // Four e2m1 codes [1, 2, 3, 4] pack low-nibble-first to bytes [0x21, 0x43].
        let packed = vec![0x21u8, 0x43];
        let tensor = Tensor::<f4e2m1, m![C], Emulation>::from_buf(packed.clone());
        // `into_buf` returns the same bytes it was given (stored as-is, not re-packed).
        assert_eq!(tensor.clone().into_buf(), packed);
        // and the codes decode back.
        let codes: Vec<u8> = tensor.into_vec().iter().map(|c| c.to_bits()).collect();
        assert_eq!(codes, vec![0x1, 0x2, 0x3, 0x4]);
    }

    /// `from_buf` on a byte-multiple `D` is the LE byte image and equals the `from_vec` result (the
    /// `from_safetensors` path). Pins that the pre-packed and logical constructors agree for byte widths.
    #[test]
    fn emulation_from_buf_byte_multiple_matches_from_vec() {
        axes![A = 3];
        let vals = vec![1i32, -1, 0x0102_0304];
        let packed: Vec<u8> = [1i32.to_le_bytes(), (-1i32).to_le_bytes(), 0x0102_0304i32.to_le_bytes()].concat();
        let from_buf = Tensor::<i32, m![A], Emulation>::from_buf(packed);
        assert_eq!(from_buf.clone().into_vec(), vals.clone());
        assert_eq!(
            from_buf.into_vec(),
            Tensor::<i32, m![A], Emulation>::from_vec(vals).into_vec()
        );
    }

    /// The buffer is `Mapping::SIZE` long, padding cells included. `m![A # 8]` with `A = 6` is a
    /// 6-live + 2-pad buffer of length 8; `BufStorage` keeps all 8 cells.
    #[test]
    fn emulation_buffer_is_padded_size_and_round_trips() {
        axes![A = 6];

        let data: Vec<i32> = (0..8).collect();
        let tensor = Tensor::<i32, m![A # 8], Emulation>::from_vec(data.clone());

        assert_eq!(tensor.into_vec(), data);
    }

    /// Element-wise `map` over the physical buffer.
    #[test]
    fn emulation_map() {
        axes![A = 4];

        let tensor = Tensor::<i32, m![A], Emulation>::from_vec(vec![0, 1, 2, 3]);
        let mapped = tensor.map(|v| v * 10);

        assert_eq!(mapped.into_vec(), vec![0, 10, 20, 30]);
    }

    /// Element-wise `zip_with` over two physical buffers of the same mapping.
    #[test]
    fn emulation_zip_with() {
        axes![A = 4];

        let lhs = Tensor::<i32, m![A], Emulation>::from_vec(vec![1, 2, 3, 4]);
        let rhs = Tensor::<i32, m![A], Emulation>::from_vec(vec![10, 20, 30, 40]);
        let sum = lhs.zip_with(&rhs, |a, b| a + b);

        assert_eq!(sum.into_vec(), vec![11, 22, 33, 44]);
    }

    /// Multi-axis layout: `m![K, M]` (K outer / M inner) exercises per-axis buffer strides
    /// (`stride_K = M`, `stride_M = 1`). Round-trip confirms `read_index`/`write_index` address the
    /// physical buffer correctly through `Σ coordₖ · strideₖ`.
    #[test]
    fn emulation_multi_axis_strides_round_trip() {
        axes![K = 8, M = 4];

        let buf: Vec<i32> = (0..8).flat_map(|k| (0..4).map(move |m| k + 100 * m)).collect();
        let tensor = Tensor::<i32, m![K, M], Emulation>::from_vec(buf.clone());

        // Re-map (identity transpose) forces a write_index/read_index round-trip over the strides.
        let same: Tensor<i32, m![K, M], Emulation> = tensor.transpose(false);
        assert_eq!(same.into_vec(), buf);
    }

    /// Real relayout via the sequencer walk: `m![K,N]` -> `m![N,K]` permutes the buffer.
    #[test]
    fn emulation_transpose_relayout() {
        axes![K = 2, N = 3];

        // m![K,N]: K outer (stride 3), N inner. buf[k*3+n].  k=0:{0,1,2}, k=1:{10,11,12}
        let src = Tensor::<i32, m![K, N], Emulation>::from_vec(vec![0, 1, 2, 10, 11, 12]);

        // m![N,K]: N outer (stride 2), K inner. out[n*2+k] = src[k,n].
        let dst: Tensor<i32, m![N, K], Emulation> = src.transpose(false);

        // n=0:{k0,k1}={0,10}, n=1:{1,11}, n=2:{2,12}
        assert_eq!(dst.into_vec(), vec![0, 10, 1, 11, 2, 12]);
    }

    /// `reduce` over a partially-split shared symbol: `Src = m![K, M]` consolidates K into a single
    /// `K(stride 1, mod 8)` term in `self.axes`, while `Dst = m![K / 2, M]` carries only K's
    /// `(stride 2, mod 4)` sub-factor: the `(stride 1, mod 2)` piece must be reduced.
    #[test]
    fn emulation_reduce_keeps_partial_axis_when_only_a_sub_factor_remains() {
        axes![K = 8, M = 4];

        let source_buf: Vec<i32> = (0..8).flat_map(|k| (0..4).map(move |m| k + 100 * m)).collect();
        let source = Tensor::<i32, m![K, M], Emulation>::from_vec(source_buf);

        let reduced: Tensor<i32, m![K / 2, M], Emulation> = source.reduce_add();

        let expected: Vec<i32> = (0..4).flat_map(|j| (0..4).map(move |m| 4 * j + 1 + 200 * m)).collect();
        assert_eq!(reduced.into_vec(), expected);
    }

    /// `reduce` with broadcast, covering the same partial-sub-factor hazard as the test above, *plus*
    /// a real broadcast axis that exists only in `Dst`.
    #[test]
    fn emulation_reduce_keeps_partial_axis_and_broadcasts_extra_dst_axis() {
        axes![A = 4, B = 2];

        let source = Tensor::<i32, m![A], Emulation>::from_vec(vec![0, 1, 2, 3]);

        let result: Tensor<i32, m![A / 2, B], Emulation> = source.reduce(|a, b| a + b, 0, true);

        assert_eq!(result.into_vec(), vec![1, 1, 5, 5]);
    }

    /// `write_gather` round-trip, `scaled=true`: indices are in byte-offset units (e.g. row 1 of
    /// `[W, V=2]` of i32 = byte offset `1*2*4 = 8`). The `scaled=false` path is covered by
    /// [`emulation_write_gather_roundtrip_unscaled`].
    #[test]
    fn emulation_write_gather_roundtrip_scaled() {
        axes![W = 3, V = 2, K = 4];

        let table = Tensor::<i32, m![W, V], Emulation>::from_vec(vec![10, 11, 20, 21, 30, 31]);
        let index = Tensor::<i32, m![K], Emulation>::from_vec(vec![0, 16, 8, 0]);
        let mut output = Tensor::<i32, m![K, V], Emulation>::zeroed();
        table.gather::<_, _>(&mut output, &index, true);

        assert_eq!(output.into_vec(), vec![10, 11, 30, 31, 20, 21, 10, 11]);
    }

    /// `write_scatter` round-trip: scatter rows of a `[K, V]` source into a `[W, V]` destination at
    /// indexed positions (the inverse of the gather above).
    #[test]
    fn emulation_write_scatter_roundtrip_unscaled() {
        axes![W = 3, V = 2, K = 3];

        let source = Tensor::<i32, m![K, V], Emulation>::from_vec(vec![10, 11, 20, 21, 30, 31]);
        let index = Tensor::<i32, m![K], Emulation>::from_vec(vec![2, 0, 1]);
        let mut output = Tensor::<i32, m![W, V], Emulation>::zeroed();
        source.scatter::<m![K], _, _>(&mut output, &index, false);

        // row k of source lands at row index[k]: 0→W2, 1→W0, 2→W1.
        assert_eq!(output.into_vec(), vec![20, 21, 30, 31, 10, 11]);
    }
}
