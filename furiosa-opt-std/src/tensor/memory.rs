//! Tensors placed on memory.

mod dma_layout;
mod redistribute;

pub use redistribute::*;

use dma_layout::{assert_dm_dma_layout, assert_dma_layout};
use rand::Rng;
use rand::distr::StandardUniform;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;

use furiosa_opt_rt::Buffer;

use furiosa_mapping::*;
use furiosa_opt_lower::{DM_WRITE_ALIGN_BYTES, PadInput, TileInput, config_pad, config_tile};
use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::storage::BufStorage;
use crate::tensor::view::Tiled;
use crate::tensor::*;

/// Address.
///
/// TODO: check that every address is 64-bit.
pub type Address = u64;

/// Tensor stored in host memory.
///
/// `D: MaterializableScalar`, not just `Scalar`: a `HostTensor` is exactly the "host... carrier" the
/// bound's own doc names, so a non-materializable staging type (`i5`/`i9`) can never even be
/// CONSTRUCTED as one -- rejected at the type's own definition, not left to be caught only if some
/// particular method (`into_vec`, etc.) happens to touch the unsound whole-buffer path. `i5`/`i9`'s
/// entire legitimate lifetime is between `fetch_zero_point_sub` and `contract_outer`, both
/// engine-internal (`TuTensor`/bare `Tensor`), never through this public host-facing wrapper.
#[primitive(HostTensor)]
#[derive(Debug, Clone)]
pub struct HostTensor<D: MaterializableScalar, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Element, B>,
}

impl<D: MaterializableScalar, Element: M, B: Backend> From<Tensor<D, Element, B>> for HostTensor<D, Element, B> {
    fn from(inner: Tensor<D, Element, B>) -> Self {
        Self { inner }
    }
}

impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = Element;

    pub(crate) fn inner(&self) -> &Tensor<D, Element, B> {
        &self.inner
    }

    pub(crate) fn storage(&self) -> &B::Storage<D> {
        &self.inner.inner
    }

    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage<D> {
        &mut self.inner.inner
    }

    /// The same tensor in page-locked host memory, which a transfer reads or writes directly
    /// instead of pinning pages on every call. Already pinned memory stays where it is.
    pub fn pinned(self) -> Result<Self, Error> {
        Ok(Tensor::from_inner(B::pin(self.inner.inner)?).into())
    }

    /// Creates a tensor from an initialized buffer. Panics if the buffer length does not match the
    /// mapping size.
    pub fn from_vec(data: impl IntoIterator<Item = D>) -> Self {
        Tensor::from_vec(data).into()
    }

    /// Creates a tensor from a pre-packed device byte image ([`crate::scalar::Scalar::to_buf`]'s inverse), stored as-is.
    /// Contrast [`Self::from_vec`], which packs logical values; pre-packed fp4 weights come through here
    /// to avoid a decode + re-pack round-trip. Panics on a byte-length mismatch.
    pub fn from_buf(buf: Vec<u8>) -> Self {
        Tensor::from_buf(buf).into()
    }

    /// Prepares a transfer of this tensor to HBM: `.await` stages it into a fresh device
    /// allocation, `.output(&mut hbm).await` writes into an existing one.
    pub fn to_hbm<'t, Chip: M, Element2: M>(
        &'t self,
        dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    ) -> ToHbm<'t, D, Element, Chip, Element2, B> {
        ToHbm {
            host: self,
            dma,
            _marker: PhantomData,
        }
    }

    /// Consumes self and returns the inner tensor.
    pub fn into_inner(self) -> Tensor<D, Self::Mapping, B> {
        self.inner
    }

    /// Returns the tensor data as a flat `Vec<D>`, consuming the tensor.
    pub fn into_vec(self) -> Vec<D> {
        self.inner.into_vec()
    }
}

/// Host-side `HostTensor` constructors. Bound to `Backend`; the value-iterating methods (`zero`,
/// `rand`) and `from_vec` / `from_safetensors` all bottom out in `Tensor::from_vec`, which
/// `BufStorage` implements as a real `Vec<D>` fill, so those work on Npu / Cpu host-side
/// staging too.
impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Creates a tensor filled with zeros.
    pub fn zero() -> Self
    where
        D: num_traits::Zero,
    {
        Tensor::splat(D::zero()).into()
    }

    /// Creates a tensor filled with random values.
    #[primitive(HostTensor::rand)]
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Tensor::rand(rng).into()
    }

    /// Creates a tensor from a `safetensors` tensor view.
    ///
    /// The view's per-axis shape must match `Element`'s pair-flattened size list (e.g.
    /// `m![H, X]` expects safetensors shape `[H.size, X.size]`) and its bytes are decoded as
    /// little-endian `D` values — LE is mandated by the safetensors format spec, not our
    /// choice. Returns [`safetensors::SafeTensorError::TensorInvalidInfo`] on any mismatch.
    pub fn from_safetensors(view: &safetensors::tensor::TensorView<'_>) -> Result<Self, safetensors::SafeTensorError>
    where
        D: ScalarBytes,
    {
        fn flat_shape(mapping: &Mapping, out: &mut Vec<usize>) {
            match mapping {
                Mapping::Pair { left, right } => {
                    flat_shape(left, out);
                    flat_shape(right, out);
                }
                _ => out.push(mapping.size()),
            }
        }
        let mut expected_shape = Vec::new();
        flat_shape(&Element::to_value(), &mut expected_shape);
        if view.shape() != expected_shape.as_slice() {
            return Err(safetensors::SafeTensorError::TensorInvalidInfo);
        }
        let stride = D::BITS / 8;
        if view.data().len() != Element::SIZE * stride {
            return Err(safetensors::SafeTensorError::TensorInvalidInfo);
        }
        // The view's LE bytes ARE the packed device image for a byte-multiple `D`, so store them directly
        // through `from_buf` rather than decoding each element and re-packing.
        Ok(Tensor::from_buf(view.data().to_vec()).into())
    }
}

/// Tensor stored in HBM memory.
#[primitive(HbmTensor)]
pub struct HbmTensor<D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Pair<Chip, Element>, B>,
    // The device allocation this tensor names and keeps live, one chip's share of the bytes.
    // `None` until a host-side transfer places it; a device function's own tensors are placed by
    // the compiled program and never reach the host this way.
    buffer: Option<Buffer>,
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> std::fmt::Debug for HbmTensor<D, Chip, Element, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HbmTensor")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Element }];

    pub(crate) fn from_parts(inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self { inner, buffer: None }
    }

    /// A fresh HBM tensor, for a device function's output.
    ///
    /// The backend allocates it: `Npu` takes a device allocation, which the handle owns and frees on
    /// drop, and `Cpu` holds the bytes on the host. A kernel compiled for `Npu` never runs this,
    /// because the compiled program places its own buffers.
    // `new()` builds an uninitialized handle, so a `Default` impl (which would look zero-cost and
    // safe) is deliberately not provided, matching `DmTensor::new`.
    #[allow(clippy::new_without_default)]
    #[primitive(HbmTensor::new)]
    pub fn new() -> Self {
        B::alloc_hbm::<D, Chip, Element>()
    }

    /// An HBM tensor whose bytes are on the device: its host storage is empty, and allocates nothing.
    pub(crate) fn unbacked<Buf: crate::storage::Buf>() -> Self
    where
        B: Backend<Storage<D> = BufStorage<D, Buf>>,
    {
        Self::from_parts(Tensor::from_inner(BufStorage::from_buf(Vec::new())))
    }

    /// Names the device allocation this tensor lives in and keeps it live for the tensor's lifetime.
    pub(crate) fn place(&mut self, buffer: Buffer) {
        self.buffer = Some(buffer);
    }

    pub(crate) fn placed(mut self, buffer: Buffer) -> Self {
        self.place(buffer);
        self
    }

    pub(crate) fn inner(&self) -> &Tensor<D, Self::Mapping, B> {
        &self.inner
    }

    /// The device allocation this tensor is placed in, or `None` while the compiled program owns
    /// its placement.
    pub(crate) fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Size of the packed device image in bytes, the measure [`Self::to_buf`] produces.
    ///
    /// The wire width, not the staging width: the types staged wider (`i5` / `i9`) are not
    /// `MaterializableScalar`, so no host-facing buffer ever holds one.
    pub fn size() -> usize {
        D::size_in_bytes_from_length(Pair::<Chip, Element>::SIZE)
    }

    /// Prepares a transfer of this tensor to the host: `.await` reads it into fresh host memory,
    /// `.output(&mut host).await` reads into an existing tensor, directly when that tensor is
    /// [`HostTensor::pinned`].
    pub fn to_host<'t, Element2: M>(
        &'t self,
        dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    ) -> ToHost<'t, D, Chip, Element, Element2, B>
    where
        D: MaterializableScalar,
    {
        ToHost {
            hbm: self,
            dma,
            _marker: PhantomData,
        }
    }

    /// Returns the tensor data as a flat logical `Vec<D>` in `m![Chip, Element]` axis order, one `D`
    /// per logical element. For a sub-byte scalar this is *not* the packed byte image (see
    /// [`Self::to_buf`]).
    pub fn into_vec(self) -> Vec<D>
    where
        D: MaterializableScalar,
    {
        self.inner.into_vec()
    }

    /// Returns the dense packed byte image in `m![Chip, Element]` axis order, sized
    /// [`Scalar::size_in_bytes_from_length`]. A 4-bit scalar ([`f4e2m1`] / [`i4`]) packs two codes per
    /// byte, so the result is half the element count; byte-aligned scalars pass through unchanged. This
    /// is the buffer the LIR executor consumes and the `compare_edf!` harness feeds as the LIR input.
    pub fn to_buf(&self) -> Vec<u8>
    where
        D: MaterializableScalar,
    {
        // Cpu / Npu move the already-packed bytes out; Simulation re-packs. Both agree because
        // they share the `Scalar::store` packing primitive.
        let packed = self.inner.clone().into_buf();
        debug_assert_eq!(packed.len(), D::size_in_bytes_from_length(Pair::<Chip, Element>::SIZE));
        packed
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Creates an immutable view of the tensor.
    #[primitive(HbmTensor::view)]
    pub fn view<'l>(&'l self) -> HbmTensorView<'l, D, Chip, Element, B> {
        HbmTensorView {
            inner: self.inner.view(),
            buffer: self.buffer.clone(),
            chip_tiled: false,
        }
    }

    /// Creates a mutable view of the tensor.
    #[primitive(HbmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> HbmTensorViewMut<'l, D, Chip, Element, B> {
        HbmTensorViewMut {
            inner: self.inner.view_mut(),
            buffer: self.buffer.clone(),
            chip_tiled: false,
        }
    }

    /// Redistributes HBM chip slots by DMA: `shuffle_pattern[target] = source`.
    #[primitive(HbmTensor::hbm_chip_shuffle)]
    pub fn hbm_chip_shuffle<const CHIP_DIM: usize, const DMA: Dma>(
        &self,
        dma: &mut DmaContext<{ DMA }, B>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> HbmTensor<D, Chip, Element, B> {
        self.view().hbm_chip_shuffle(dma, shuffle_pattern)
    }

    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(HbmTensor::to_hbm)]
    pub fn to_hbm<const DMA: Dma, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ DMA }, B>,
    ) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::from_parts(self.inner.transpose(true))
    }

    /// Gather DRAM rows into SRAM at positions given by index tensor.
    ///
    /// Implements `index_select` along the table's gather-key axis (the axis present in
    /// `Element` but not in the output's `Element2`). The output's indices axes (in
    /// `Element2`, mirroring `Element3` from the index tensor) replace that gather-key axis:
    /// `output[..pre, k, ..post] = self[..pre, index[k], ..post]`.
    ///
    /// Inverse of [`DmTensor::dma_scatter`]. Index values are byte offsets along the gather
    /// axis: to gather row `r`, pass `r` times one row's byte size (its element count times
    /// `size_of::<D>()`; e.g. `128 * 2 = 256` for a 128-wide `bf16` row). Gathering with a raw,
    /// SPM-resident index is [`Self::dma_gather_unscaled`].
    #[primitive(HbmTensor::dma_gather_scaled)]
    pub fn dma_gather_scaled<Cluster2: M, Slice2: M, Element2: M, Element3: M>(
        &self,
        index: &HbmTensor<i32, Chip, Element3, B>,
    ) -> DmTensor<D, Chip, Cluster2, Slice2, Element2, B> {
        let mut output: DmTensor<D, Chip, Cluster2, Slice2, Element2, B> = DmTensor::from_parts(Tensor::zeroed(), None);
        self.inner.gather::<_, _>(&mut output.inner, &index.inner, true);
        output
    }

    /// Gather DRAM rows into SRAM at positions given by an SPM-resident (on-chip) index,
    /// interpreting index values as raw row positions.
    ///
    /// Complements [`Self::dma_gather_scaled`] for indices computed on-chip (paged-attention block
    /// tables, unscaled embedding lookups): the index is an SPM-resident `DmTensor` rather than
    /// an `HbmTensor` in DRAM, and its values are raw row positions rather than the byte offsets
    /// [`Self::dma_gather_scaled`] expects.
    #[primitive(HbmTensor::dma_gather_unscaled)]
    pub fn dma_gather_unscaled<IdxCluster: M, IdxSlice: M, IdxElement: M, Cluster2: M, Slice2: M, Element2: M>(
        &self,
        index: &DmTensor<i32, Chip, IdxCluster, IdxSlice, IdxElement, B>,
    ) -> DmTensor<D, Chip, Cluster2, Slice2, Element2, B> {
        let mut output: DmTensor<D, Chip, Cluster2, Slice2, Element2, B> = DmTensor::from_parts(Tensor::zeroed(), None);
        self.inner.gather::<_, _>(&mut output.inner, &index.inner, false);
        output
    }
}

// ANCHOR: dma_impl
impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensor::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        DmTensor::from_parts(self.inner.transpose(true), None)
    }

    /// Reshapes the tensor to a different mapping at the same HBM address, consuming `self`.
    /// The HBM analogue of [`DmTensor::reshape`]; both delegate to [`Tensor::reshape`].
    ///
    /// # Safety
    ///
    /// The per-level sizes (`Chip::SIZE == Chip2::SIZE`, `Element`) are asserted at compile time below
    /// (see [`constraints::assert_hbm_reshape_dimension_preserved`]); the genuine precondition is
    /// [`Tensor::reshape`]'s: the old and new mappings must lay the elements out in the SAME physical
    /// (wire) order, so the relabel moves no data. Axis regrouping (merge/split) preserves wire order
    /// and is valid; a permutation is not (use a transpose). Equal sizes do not guarantee this.
    /// Consuming `self` is the safety contract made explicit: the old-shaped handle cannot survive to
    /// alias the same HBM bytes under a conflicting mapping.
    #[primitive(HbmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensor<D, Chip2, Element2, B> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        let reshaped = unsafe { self.inner.reshape::<m![{ Chip2 }, { Element2 }]>() };
        HbmTensor {
            inner: reshaped,
            buffer: self.buffer,
        }
    }
}
// ANCHOR_END: dma_impl

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Shuffles data across clusters on HBM (HBM ↔ HBM DMA). Not yet implemented: an `HbmTensor`
    /// has no cluster dimension (only Chip and Element); clusters are assigned later, at `to_dm`.
    pub fn hbm_cluster_shuffle<const DMA: Dma>(
        &self,
        _dma: &mut DmaContext<{ DMA }, B>,
        _shuffle_pattern: &[usize],
    ) -> Self {
        todo!(
            "hbm_cluster_shuffle is Under Construction. HbmTensor has no Cluster axis \
             (only Chip + Element); Cluster distribution is decided at .to_dm() time. \
             No current callers. Either the Element axis is meant to encode a Cluster \
             sub-axis (API needs to take that axis explicitly) or the operation belongs \
             on DmTensorView::cluster_swap. Pending design review; see the doc \
             comment on hbm_cluster_shuffle."
        )
    }
}

/// View of an HBM tensor.
#[primitive(HbmTensorView)]
#[derive(Debug, Clone)]
pub struct HbmTensorView<'l, D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    inner: TensorView<'l, D, Pair<Chip, Element>, B>,
    // The base tensor's allocation, absent for Cpu and compiler-placed tensors. A tile's offset
    // lives in `inner`; `buffer()` applies it once, so repeated tiles cannot double-count it.
    buffer: Option<Buffer>,
    chip_tiled: bool,
}

/// The per-chip byte range a view covers: from the window starting at element `window_base` to the
/// end of a base tensor of `base_len` elements spread over `Chip` chips.
fn hbm_window<D: Scalar, Chip: M>(base_len: usize, window_base: usize) -> std::ops::Range<usize> {
    assert_ne!(Chip::SIZE, 0, "an HBM view must span at least one chip");
    assert!(
        base_len.is_multiple_of(Chip::SIZE),
        "HBM view elements must divide evenly across chips"
    );
    let elements = base_len / Chip::SIZE;
    assert_ne!(elements, 0, "an HBM view must span at least one element per chip");
    D::size_in_bytes_from_length(window_base % elements)..D::size_in_bytes_from_length(elements)
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Element }];

    /// The device bytes this view covers: the base tensor's allocation from the window
    /// [`Self::tile`] selected to its end. It is the whole of what a view tells the Npu backend.
    pub(crate) fn buffer(&self) -> Option<Buffer> {
        Some(self.buffer.as_ref()?.slice(self.window()))
    }

    /// The per-chip byte range of the base allocation this view covers.
    pub(crate) fn window(&self) -> std::ops::Range<usize> {
        assert!(!self.chip_tiled, "a host-bound HBM view cannot select individual chips");
        hbm_window::<D, Chip>(self.inner.base_len(), self.inner.window_base())
    }

    /// Writes to HBM tensor view. The destination's `Chip2` is free of the
    /// source's `Chip`: a read source (Top padding) may target a `view_mut`
    /// destination (Bottom padding). `transpose` validates the live layout.
    #[primitive(HbmTensorView::to_hbm_view)]
    pub fn to_hbm_view<const DMA: Dma, Chip2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ DMA }, B>,
        mut dst: HbmTensorViewMut<'l, D, Chip2, Element2, B>,
    ) {
        dst.inner.transpose(self.inner, true);
    }

    /// Writes to data memory tensor view.
    #[primitive(HbmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element2, B>,
    ) {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip2 }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        dst.inner.transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip2, Element, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        HbmTensorView {
            inner,
            buffer: self.buffer.clone(),
            chip_tiled: true,
        }
    }

    /// Creates immutable views by splitting along a tile expression.
    #[primitive(HbmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip, Element2, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        HbmTensorView {
            inner,
            buffer: self.buffer.clone(),
            chip_tiled: self.chip_tiled,
        }
    }

    /// Reshapes the view at the same HBM address, consuming `self`. A reshape is a MOVE: a zero-copy
    /// rewrap of the same borrow (see [`TensorView::reshape`]). Sound on every backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Element` sizes asserted at compile time (see
    /// [`constraints::assert_hbm_reshape_dimension_preserved`]); precondition is same-wire-order
    /// (regroup valid, permutation not). Consuming `self` makes the move explicit.
    #[primitive(HbmTensorView::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensorView<'l, D, Chip2, Element2, B> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        HbmTensorView {
            inner: unsafe { self.inner.reshape::<m![{ Chip2 }, { Element2 }]>() },
            buffer: self.buffer.clone(),
            chip_tiled: self.chip_tiled,
        }
    }

    /// Views the same cells inside the wider buffer they sit in, restating `Element` with one
    /// outermost padding factor: `m![L1]` as `m![L1 # 256]`. See [`DmTensorView::pad`]; on an HBM
    /// SOURCE this is how a DMA states the extent its destination is staged at, so the two sides carve
    /// the same tail. The read over-reaches the live cells into declared don't-care, which is what a
    /// read packet is allowed to do -- `Pad::extent` trims to the live cells on the SINK side only.
    #[primitive(HbmTensorView::pad)]
    pub fn pad<Element2: M>(self) -> HbmTensorView<'l, D, Chip, Element2, B> {
        config_pad(PadInput {
            element: Element::to_value(),
            expected: Element2::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        HbmTensorView {
            inner: self.inner.redeclare::<m![{ Chip }, { Element2 }]>(),
            buffer: self.buffer.clone(),
            chip_tiled: self.chip_tiled,
        }
    }

    /// Returns the view data as a flat `Vec<D>` in `m![Chip, Element]` axis order. Reads the view
    /// into a temporary tensor and serializes that. Borrows, so the view stays usable (e.g. read as
    /// a LIR input, then passed to `launch`).
    pub fn to_vec(&self) -> Vec<D>
    where
        D: MaterializableScalar,
    {
        self.inner.clone().read().into_vec()
    }

    /// [`Self::to_vec`] for owned callers; moves the view straight into `read` (no view clone).
    pub fn into_vec(self) -> Vec<D>
    where
        D: MaterializableScalar,
    {
        self.inner.read().into_vec()
    }

    /// Dense packed byte image of the view; see [`HbmTensor::to_buf`].
    pub fn to_buf(&self) -> Vec<u8>
    where
        D: MaterializableScalar,
    {
        // The `read` tensor's buffer already IS the packed device image on Cpu / Npu, so
        // `into_buf` moves it out directly; Simulation re-packs. Both share `Scalar::store`.
        self.inner.clone().read().into_buf()
    }
}

impl<'l, D: MaterializableScalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensorView::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        DmTensor::from_parts(self.inner.read().transpose(true), None)
    }

    /// Redistributes HBM chip slots by DMA: `shuffle_pattern[target] = source`.
    /// Panics unless the pattern is a permutation of every position in `Chip`.
    #[primitive(HbmTensorView::hbm_chip_shuffle)]
    pub fn hbm_chip_shuffle<const CHIP_DIM: usize, const DMA: Dma>(
        self,
        dma: &mut DmaContext<{ DMA }, B>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> HbmTensor<D, Chip, Element, B> {
        assert_chip_shuffle_pattern(shuffle_pattern, Chip::SIZE);
        let mut shuffled: HbmTensor<D, Chip, Element, B> = HbmTensor::new();

        for (target_chip_idx, source_chip_idx) in shuffle_pattern.iter().enumerate() {
            self.chip_tile::<Chip, 1, Padding<Identity, Broadcast<CHIP_DIM>>>(*source_chip_idx)
                .to_hbm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, Broadcast<CHIP_DIM>, { PaddingKind::Bottom }>>(
                            target_chip_idx,
                        ),
                );
        }

        shuffled
    }
}

/// Mutable view of an HBM tensor.
#[primitive(HbmTensorViewMut)]
#[derive(Debug)]
pub struct HbmTensorViewMut<'l, D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    inner: TensorViewMut<'l, D, Pair<Chip, Element>, B>,
    // The base tensor's allocation; see [`HbmTensorView`]'s field of the same name.
    buffer: Option<Buffer>,
    chip_tiled: bool,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorViewMut<'l, D, Chip, Element, B> {
    /// The device bytes this view covers; see [`HbmTensorView::buffer`].
    pub(crate) fn buffer(&self) -> Option<Buffer> {
        Some(self.buffer.as_ref()?.slice(self.window()))
    }

    /// The per-chip byte range of the base allocation this view covers.
    pub(crate) fn window(&self) -> std::ops::Range<usize> {
        assert!(!self.chip_tiled, "a host-bound HBM view cannot select individual chips");
        hbm_window::<D, Chip>(self.inner.base_len(), self.inner.window_base())
    }

    /// Returns a dense packed byte snapshot of the view without consuming its mutable handle.
    ///
    /// In-place execution harnesses use this to give their reference executors the same
    /// pre-launch contents that the device kernel receives.
    pub fn to_buf(&self) -> Vec<u8>
    where
        D: MaterializableScalar,
    {
        self.inner.read().into_buf()
    }

    /// Creates mutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorViewMut::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip2, Element, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        HbmTensorViewMut {
            inner,
            buffer: self.buffer.clone(),
            chip_tiled: true,
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip, Element2, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        HbmTensorViewMut {
            inner,
            buffer: self.buffer.clone(),
            chip_tiled: self.chip_tiled,
        }
    }

    /// Reshapes the mutable view at the same HBM address, consuming `self`. A reshape is a MOVE: a
    /// zero-copy rewrap of the same `&mut` borrow (see [`TensorViewMut::reshape`]). Sound on every
    /// backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Element` sizes asserted at compile time (see
    /// [`constraints::assert_hbm_reshape_dimension_preserved`]); precondition is same-wire-order
    /// (regroup valid, permutation not). Consuming `self` makes the move explicit.
    #[primitive(HbmTensorViewMut::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensorViewMut<'l, D, Chip2, Element2, B> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        HbmTensorViewMut {
            inner: unsafe { self.inner.reshape::<m![{ Chip2 }, { Element2 }]>() },
            buffer: self.buffer.clone(),
            chip_tiled: self.chip_tiled,
        }
    }
}

/// Tensor stored in data memory.
#[primitive(DmTensor)]
#[derive(Debug)]
pub struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
    address: Option<Address>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// `Cluster` / `Slice` map to physical SRAM partitions; checked in `from_parts` so every DM
    /// tensor constructor validates them at compile time. One `const` block per check so a bad
    /// `Cluster` and a bad `Slice` each report their own error (a single block stops at the first
    /// panic).
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    pub(crate) fn from_parts(inner: Tensor<D, Self::Mapping, B>, address: Option<Address>) -> Self {
        Self::check_constraints();

        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Creates a fresh DM tensor handle with no assigned address. The backend places it.
    ///
    /// `Cluster` / `Slice` are validated at compile time (see [`Self::from_parts`]). A bad partition
    /// is rejected before codegen; because each check sits in its own `const` block, every violated
    /// check reports its own error (here `Cluster = 3` and `Slice = 5` both do, in one compile):
    ///
    /// ```compile_fail
    /// use furiosa_opt_std::prelude::*;
    /// // Cluster must be 1 | 2 and Slice must be 64 | 128 | 256.
    /// let _ = DmTensor::<i32, m![1], m![3], m![5], m![8]>::new();
    /// ```
    // `new()` builds an uninitialized handle, so a `Default` impl (which would look zero-cost and
    // safe) is deliberately not provided.
    #[allow(clippy::new_without_default)]
    #[primitive(DmTensor::new)]
    pub fn new() -> Self {
        Self::from_parts(Tensor::zeroed(), None)
    }
}

/// Determines which hardware dimension keys an asymmetric slice's indices.
#[derive(Clone, Copy)]
enum SliceIndexing {
    PerChip,
    PerCluster,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Creates immutable views by splitting along a tile expression.
    #[primitive(DmTensor::view)]
    pub fn view<'l>(&'l self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element, B> {
        DmTensorView {
            inner: self.inner.view(),
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(DmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.view_mut(),
        }
    }

    /// Selects one live position of an outermost, 8-byte-aligned axis per chip.
    #[primitive(DmTensor::asymmetric_chip_slice)]
    pub fn asymmetric_chip_slice<AxisToSlice: M, Element2: M>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
        slice_indices: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        self.asymmetric_slice::<AxisToSlice, Element2>(sub, slice_indices, SliceIndexing::PerChip)
    }

    /// Selects one live position of an outermost, 8-byte-aligned axis per cluster.
    #[primitive(DmTensor::asymmetric_cluster_slice)]
    pub fn asymmetric_cluster_slice<AxisToSlice: M, Element2: M>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
        slice_indices: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        self.asymmetric_slice::<AxisToSlice, Element2>(sub, slice_indices, SliceIndexing::PerCluster)
    }

    fn asymmetric_slice<AxisToSlice: M, Element2: M>(
        &self,
        _sub: &mut TuContext<{ Tu::Sub }>,
        slice_indices: &[usize],
        indexing: SliceIndexing,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_asymmetric_slice::<D, AxisToSlice, Element, Element2>();
        let targets = match indexing {
            SliceIndexing::PerChip => Chip::SIZE,
            SliceIndexing::PerCluster => Cluster::SIZE,
        };
        assert_slice_indices::<AxisToSlice>(slice_indices, targets);
        let tensor = self.view();
        let mut sliced = DmTensor::new();

        for (target, slice_idx) in slice_indices.iter().enumerate() {
            match indexing {
                SliceIndexing::PerChip => {
                    let selected = tensor
                        .chip_tile_derived::<Chip, 1>(target)
                        .slice_axis::<AxisToSlice>(*slice_idx);
                    sliced
                        .view_mut()
                        .chip_tile_derived::<Chip, 1>(target)
                        .inner
                        .transpose(selected.inner, false);
                }
                SliceIndexing::PerCluster => {
                    let selected = tensor
                        .cluster_tile_derived::<Cluster, 1>(target)
                        .slice_axis::<AxisToSlice>(*slice_idx);
                    sliced
                        .view_mut()
                        .cluster_tile_derived::<Cluster, 1>(target)
                        .inner
                        .transpose(selected.inner, false);
                }
            }
        }

        sliced
    }

    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(DmTensor::to_hbm)]
    pub fn to_hbm<Element2: M>(&self, _dma: &mut DmaContext<{ Dma::Tensor }, B>) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::from_parts(self.inner.transpose(true))
    }

    /// Scatter SRAM values to DRAM at positions given by index tensor.
    ///
    /// ```text
    /// data:   [N, K, V]
    /// index:  [N, K]
    /// output: [N, X, V]
    ///
    /// (data - Chip).divide(K) = [N, V]
    /// ```
    ///
    /// Index values are byte offsets along the scatter axis (the dual of [`HbmTensor::dma_gather_scaled`]):
    /// to write row `r`, pass `r` times one row's byte size (its element count times
    /// `size_of::<D>()`; e.g. `128 * 2 = 256` for a 128-wide `bf16` row). Scattering with a raw,
    /// SPM-resident index is [`Self::dma_scatter_unscaled`].
    #[primitive(DmTensor::dma_scatter)]
    pub fn dma_scatter<Key: M, Element2: M, Element3: M>(
        &self,
        index: &HbmTensor<i32, Chip, Element3, B>,
        output: &mut HbmTensor<D, Chip, Element2, B>,
    ) {
        let src = Pair::<Slice, Element>::to_value();
        let key = Key::to_value();
        // The key must be fully contained in the source: carving it out of `src` with the matcher
        // must consume every key cell (the matcher dual of `divide(..).exact_checked()`).
        assert!(
            sequence(&[&key], &[&src], SequencerMode::Read).is_ok(),
            "scatter key `{key}` must be fully contained in source `{src}`. \
             If the key axis is split across Chip and Element, indirect DMA cannot address it.",
        );

        self.inner.scatter::<Key, _, _>(&mut output.inner, &index.inner, true);
    }

    /// Scatter SRAM values to DRAM at positions given by an SPM-resident (on-chip) index,
    /// interpreting index values as raw row positions.
    ///
    /// Complements [`Self::dma_scatter`]'s DRAM byte-offset index, for indices computed
    /// on-chip. `Key` names the scatter-key axis, exactly as in [`Self::dma_scatter`]: the
    /// unscaled path scatters along the same key, so the caller must still specify it.
    /// Not yet implemented.
    // TODO: register the `DmTensor` index as the unscaled indirect-DMA SPM index tensor.
    pub fn dma_scatter_unscaled<Key: M, IdxCluster: M, IdxSlice: M, IdxElement: M, Element2: M>(
        &self,
        _index: &DmTensor<i32, Chip, IdxCluster, IdxSlice, IdxElement, B>,
        _output: &mut HbmTensor<D, Chip, Element2, B>,
    ) {
        // Same key-containment contract as `dma_scatter`.
        let src = Pair::<Slice, Element>::to_value();
        let key = Key::to_value();
        assert!(
            sequence(&[&key], &[&src], SequencerMode::Read).is_ok(),
            "scatter key `{key}` must be fully contained in source `{src}`. \
             If the key axis is split across Chip and Element, indirect DMA cannot address it.",
        );
        todo!("unscaled dma_scatter (SPM-resident raw index) is not implemented yet")
    }

    /// Converts to a data-memory tensor with the requested dimension mappings.
    #[primitive(DmTensor::to_dm)]
    pub fn to_dm<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        assert_dm_dma_layout::<D, Chip, Cluster, Slice, Element, Chip2, Cluster2, Slice2, Element2>(
            DM_WRITE_ALIGN_BYTES,
        );
        DmTensor::from_parts(self.inner.transpose(true), None)
    }

    /// Reshapes the tensor to a different mapping at the same address, consuming `self`. Delegates to
    /// [`Tensor::reshape`].
    ///
    /// # Safety
    ///
    /// The per-level sizes (`Chip::SIZE == Chip2::SIZE`, `Cluster`, `Slice`, `Element`) are asserted
    /// below; the genuine precondition is [`Tensor::reshape`]'s: the old and new DM mappings must lay
    /// the elements out in the SAME physical (wire) order, so the relabel moves no data. Axis
    /// regrouping is valid; a permutation is not (use a transpose). Equal sizes do not guarantee this.
    /// Consuming `self` makes the move explicit: reshape is a MOVE (neither alias nor copy), so no
    /// old-shaped handle survives to alias the same bytes under a conflicting mapping.
    #[primitive(DmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        let reshaped = unsafe {
            self.inner
                .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
        };
        DmTensor::from_parts(reshaped, self.address)
    }
}

/// Mutable view of a data memory tensor.
#[primitive(DmTensorViewMut)]
#[derive(Debug)]
pub struct DmTensorViewMut<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: TensorViewMut<'l, D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
}

/// View of a data memory tensor.
#[primitive(DmTensorView)]
#[derive(Debug, Clone)]
pub struct DmTensorView<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: TensorView<'l, D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    From<DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>>
    for DmTensorView<'l, D, Chip, Cluster, Slice, Element, B>
{
    fn from(view: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>) -> Self {
        Self {
            inner: view.inner.into(),
        }
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorView<'l, D, Chip, Cluster, Slice, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// Writes data to a mutable tensor view for HBM. `Chip2` is free of the
    /// source's `Chip`: a read source (Top) may target a `view_mut` destination
    /// (Bottom). `transpose` validates the live layout.
    #[primitive(DmTensorView::to_hbm_view)]
    pub fn to_hbm_view<Chip2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: HbmTensorViewMut<'l, D, Chip2, Element2, B>,
    ) {
        dst.inner.transpose(self.inner, true);
    }

    /// Writes data to a mutable tensor view for data memory. `Chip2`/`Cluster2`
    /// are free of the source's: a read source (Top) may target a `view_mut`
    /// destination (Bottom). `transpose` validates the live layout.
    #[primitive(DmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B>,
    ) {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Slice }, { Element }],
            Element,
            m![{ Cluster2 }, { Slice2 }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        dst.inner.transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorView { inner }
    }

    pub(crate) fn chip_tile_derived<Index: M, const LEN: usize>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Tiled<Index, Chip, LEN, { PaddingKind::Top }>, Cluster, Slice, Element, B> {
        DmTensorView {
            inner: self.inner.retile_derived::<Index, _>(start),
        }
    }

    pub(crate) fn slice_axis<AxisToSlice: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Tiled<AxisToSlice, Element, 1, { PaddingKind::Top }>, B> {
        DmTensorView {
            inner: self.inner.retile::<AxisToSlice, _>(start),
        }
    }

    /// Creates immutable views by splitting along a tile expression over Cluster.
    #[primitive(DmTensorView::cluster_tile)]
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster2, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster2, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Cluster::to_value(),
            expected: Cluster2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorView { inner }
    }

    pub(crate) fn cluster_tile_derived<Index: M, const LEN: usize>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Tiled<Index, Cluster, LEN, { PaddingKind::Top }>, Slice, Element, B> {
        DmTensorView {
            inner: self.inner.retile_derived::<Index, _>(start),
        }
    }

    /// Creates immutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorView::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice2, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Slice::to_value(),
            expected: Slice2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorView { inner }
    }

    /// Reshapes the view over the same borrow, consuming `self`. A reshape is a MOVE: a zero-copy
    /// rewrap of the same borrow (see [`TensorView::reshape`]). Sound on every backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Cluster`/`Slice` asserted at compile time, matching [`DmTensor::reshape`] (see
    /// [`constraints::assert_reshape_dimension_preserved`] -- `Element` isn't checked there either,
    /// per its own TODO: some current examples reshape with a mismatched `Element`). Precondition is
    /// same-wire-order (regroup within `Element` valid; a permutation is not, use a transpose).
    #[primitive(DmTensorView::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensorView<'l, D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        DmTensorView {
            inner: unsafe {
                self.inner
                    .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
            },
        }
    }

    /// Views the same cells inside the wider buffer they sit in, restating `Element` with one
    /// outermost padding factor: `m![L4, B % 64]` as `m![L4 # 512, B % 64]`. The inverse of the
    /// [`Self::tile`] that reads a padded tensor's live rows (see [`TensorView::pad`] for why it is
    /// safe), and the way to give an interleave's two operands one `Element` without narrowing the
    /// wider one and losing its extent.
    #[primitive(DmTensorView::pad)]
    pub fn pad<Element2: M>(self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        // `Element` alone: the buffer a pad re-declares is the per-slice region, so the padding is
        // outermost within `Element`. The distribution classes ride through untouched.
        config_pad(PadInput {
            element: Element::to_value(),
            expected: Element2::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        DmTensorView {
            inner: self
                .inner
                .redeclare::<m![{ Chip }, { Cluster }, { Slice }, { Element2 }]>(),
        }
    }

    /// Views a wider-staged producer's live cells alone, dropping one outermost padding factor from
    /// `Element`: `m![L1 # 256, B % 64]` as `m![L1, B % 64]`. The inverse of [`Self::pad`]; see
    /// [`TensorView::unpad`] for why it is address-preserving and why a `tile` is not a substitute.
    #[primitive(DmTensorView::unpad)]
    pub fn unpad<Element2: M>(self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        config_pad(PadInput {
            element: Element2::to_value(),
            expected: Element::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        DmTensorView {
            inner: self
                .inner
                .redeclare::<m![{ Chip }, { Cluster }, { Slice }, { Element2 }]>(),
        }
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>
{
    fn reborrow(&mut self) -> DmTensorViewMut<'_, D, Chip, Cluster, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.reborrow(),
        }
    }

    /// Creates mutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorViewMut::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorViewMut { inner }
    }

    pub(crate) fn chip_tile_derived<Index: M, const LEN: usize>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Tiled<Index, Chip, LEN, { PaddingKind::Bottom }>, Cluster, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.retile_derived::<Index, _>(start),
        }
    }

    /// Creates mutable views by splitting along a tile expression over Cluster.
    #[primitive(DmTensorViewMut::cluster_tile)]
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster2, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster2, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Cluster::to_value(),
            expected: Cluster2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorViewMut { inner }
    }

    pub(crate) fn cluster_tile_derived<Index: M, const LEN: usize>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Tiled<Index, Cluster, LEN, { PaddingKind::Bottom }>, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.retile_derived::<Index, _>(start),
        }
    }

    /// Creates mutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorViewMut::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice2, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Slice::to_value(),
            expected: Slice2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start);
        DmTensorViewMut { inner }
    }

    /// Reshapes the mutable view over the same `&mut` borrow, consuming `self`. A reshape is a MOVE: a
    /// zero-copy rewrap of the same borrow (see [`TensorViewMut::reshape`]). Sound on every backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Cluster`/`Slice` asserted at compile time, matching [`DmTensor::reshape`] (see
    /// [`constraints::assert_reshape_dimension_preserved`] -- `Element` isn't checked there either,
    /// per its own TODO: some current examples reshape with a mismatched `Element`). Precondition is
    /// same-wire-order (regroup within `Element` valid; a permutation is not, use a transpose).
    #[primitive(DmTensorViewMut::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        DmTensorViewMut {
            inner: unsafe {
                self.inner
                    .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
            },
        }
    }

    /// Fills this view's region with the typed value-domain `value` (`bf16::from_f32(1.0)`, not a
    /// `0x3f80` bit pattern), lowering to one on-device `Command::ParallelMemSet` (`itos`).
    ///
    /// `value` must be compile-time constant so it const-folds to the fill's element bits: a plain
    /// literal folds directly, a computed value needs a `const` block, e.g.
    /// `memset(const { bf16::from_f32(1.0) }, ..)`. A non-constant `value` is rejected at translation.
    ///
    /// Only a whole-region fill (a bare `view_mut()`) is supported today; a sub-view fill
    /// (`view_mut().tile(..).memset(..)`) is rejected at translation, pending a ranged `ParallelMemSet`
    /// (see `memset::lower`'s TODO). Supports every [`Scalar`] this branch translates to vISA: the byte+
    /// RNGD scalars (`i8`, `i16`, `i32`, `f32`, `bf16`, `f8e4m3`) and sub-byte `i4` (fill materialized
    /// from the low `D::BITS`; a sub-byte region must be byte-aligned or it is rejected). `f4e2m1` is a
    /// follow-up: a valid DSL `Scalar`, but its vISA `mir_ast::Scalar` variant lands with the separate
    /// fetch/table-lookup work, so a `memset(f4e2m1_value)` is rejected rather than mis-lowered.
    #[primitive(DmTensorViewMut::memset)]
    pub fn memset(&mut self, value: D, _sub: &mut TuContext<{ Tu::Sub }>) {
        // Reference backends fill this view's live cells (padding stays untouched, matching the device
        // write-back); the device path emits ParallelMemSet into the viewed region.
        let fill: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B> = Tensor::splat(value);
        self.inner.transpose(fill.view(), true);
    }
}

// ANCHOR: trf_tensor_def
/// Tensor stored in the tensor register file.
#[primitive(TrfTensor)]
#[derive(Debug)]
pub struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Element>>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
// ANCHOR_END: trf_tensor_def

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// A fresh TRF tensor, zero-filled. Where in the register file it lands is the compiler's to
    /// decide, so the handle carries no address of its own.
    pub fn zero() -> Self {
        Self::new(Tensor::zeroed())
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping, B> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping, B> {
        self.inner.view()
    }
}

// ANCHOR: vrf_tensor_def
/// Tensor stored in the vector register file (VRF).
#[primitive(VrfTensor)]
#[derive(Debug, Clone)]
pub struct VrfTensor<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
// ANCHOR_END: vrf_tensor_def

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// Every VRF handle is built here, which is where one slice's `Element` answers to the file it
    /// has to fit.
    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>) -> Self {
        constraints::assert_vrf_capacity::<D, Element>();
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// A fresh VRF tensor, zero-filled. Where in the register file it lands is the compiler's to
    /// decide, so the handle carries no address of its own.
    pub fn zero() -> Self {
        Self::new(Tensor::zeroed())
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Restates the operand under a different mapping over the same register, consuming `self`. Its
    /// use is naming the slices a broadcast operand already holds a copy in, so it can feed a stream
    /// partitioned by a named axis (the operand rule is
    /// [`IntoBranchedOperand`](crate::engine::vector::operand::IntoBranchedOperand)'s).
    ///
    /// The VRF peer of [`DmTensor::reshape`]; both delegate to [`Tensor::reshape`] and relabel rather
    /// than relayout, so no element moves.
    ///
    /// # Safety
    ///
    /// Every physical position (chip, cluster, slice, in-slice element) must already hold what the
    /// new mapping claims of it. Two forms do that, and this call must be one of them:
    ///
    /// 1. **Regrouping axes**, `m![W]` -> `m![W / 2, W % 2]`. This is [`Tensor::reshape`]'s own
    ///    precondition: both mappings lay the SAME live elements out in the SAME wire order.
    /// 2. **Naming a broadcast distribution axis**, `m![256]` -> `m![W]`. Here the live elements do
    ///    NOT line up, since a `Broadcast<256>` carries one of them and `m![W]` carries 256. It is
    ///    sound for the other reason: a broadcast at `Chip` / `Cluster` / `Slice` means those units
    ///    each hold the same copy, so naming the copies reads no other unit's data. `Element` has no
    ///    such reading, and a broadcast one relabelled to a named axis invents data.
    ///
    /// Permuting axes (`m![A, B]` -> `m![B, A]`) is neither; that is [`Tensor::transpose`]'s job.
    ///
    /// Only per-level `SIZE` equality is asserted, so which of the two forms this is stays the
    /// caller's to know. A violation is not UB but silence: the backend relabel is a volume
    /// `assert_eq!` plus a copy, so it yields wrong values under emulation and a wrong EDF once
    /// compiled, with no diagnostic on either path.
    #[primitive(VrfTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> VrfTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        let reshaped = unsafe {
            self.inner
                .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
        };
        VrfTensor::new(reshaped)
    }

    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping, B> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping, B> {
        self.inner.view()
    }
}

/// Tensor stored in dot product engine
#[derive(Debug)]
pub struct DpeTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Lane: M, Packet: M, B: Backend = CurrentBackend>
{
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Pair<Lane, Packet>>>>>, B>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Lane: M, Packet: M, B: Backend>
    DpeTensor<D, Chip, Cluster, Slice, Time, Lane, Packet, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Lane }, { Packet }];
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Lane: M, Packet: M, B: Backend>
    DpeTensor<D, Chip, Cluster, Slice, Time, Lane, Packet, B>
{
    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping, B> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping, B> {
        self.inner.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Cpu;
    use crate::runtime::{Device, Topology};

    /// Builds the shared `dma_gather_unscaled` fixture for backend `B`: an HBM table `[W=8, V=2]`
    /// (row `r` = `[10r, 10r + 1]`) and an SPM-resident (`DmTensor`) block-table index of `K=64`
    /// raw row positions. The index is a fixed non-monotonic permutation of `0..W` tiled across
    /// the `K` rows, so the gathered value cannot be reproduced from the destination position
    /// alone (this pins that the index is actually read) nor by assuming ascending indices.
    /// Returns the gathered output and the hand-derived oracle. `K=64` is the smallest legal
    /// `Slice` (see `SLICE_SIZES`).
    fn run_dma_gather_unscaled<B: Backend>() -> (Vec<i32>, Vec<i32>) {
        axes![W = 8, V = 2, K = 64];
        // Non-monotonic, hits row 0 and the max row W-1, and is decoupled from the position `k`.
        const PERM: [i32; 8] = [3, 7, 1, 5, 0, 6, 2, 4];
        let row = |k: usize| PERM[k % W::SIZE];

        let table_buf: Vec<i32> = (0..W::SIZE as i32).flat_map(|r| [10 * r, 10 * r + 1]).collect();
        let idx_buf: Vec<i32> = (0..K::SIZE).map(row).collect();
        let expected: Vec<i32> = (0..K::SIZE).flat_map(|k| [10 * row(k), 10 * row(k) + 1]).collect();

        let table = HbmTensor::<i32, m![1], m![W, V], B>::from_parts(Tensor::from_vec(table_buf));
        // The index lives in DM (SPM): `Slice = K`, the residue axis the gather iterates.
        let index = DmTensor::<i32, m![1], m![1], m![K], m![1], B>::from_parts(Tensor::from_vec(idx_buf), None);

        let output: DmTensor<i32, m![1], m![1], m![K], m![V], B> = table.dma_gather_unscaled(&index);
        (output.inner.into_vec(), expected)
    }

    /// `dma_gather_unscaled` on `Cpu`: the physical `BufStorage` gather (driven by the
    /// sequencer) matches the hand oracle. Peer of the byte-offset `dma_gather_scaled` and of the
    /// `Tensor`-level `cpu_write_gather_roundtrip_unscaled`.
    #[test]
    fn cpu_dma_gather_unscaled_roundtrip() {
        let (got, expected) = run_dma_gather_unscaled::<Cpu>();
        assert_eq!(got, expected);
    }

    /// A mutable HBM view borrows the owner's storage rather than a detached buffer. Once the view
    /// is released, reading the whole owning tensor must therefore observe the view's writes.
    #[test]
    fn cpu_hbm_view_mut_updates_owning_tensor() {
        axes![A = 8];

        let expected: Vec<i32> = (0..A::SIZE as i32).map(|x| x * 3 + 1).collect();
        let source = HbmTensor::<i32, m![1], m![A], Cpu>::from_parts(Tensor::from_vec(expected.clone()));
        let mut destination =
            HbmTensor::<i32, m![1], m![A], Cpu>::from_parts(Tensor::from_vec(std::iter::repeat_n(-1, A::SIZE)));

        {
            let source = source.view();
            let mut destination_view = destination.view_mut();
            destination_view.inner.transpose(source.inner, false);
        }

        assert_eq!(destination.into_vec(), expected);
    }

    #[test]
    fn tile_windows_end_at_allocation() {
        axes![A = 8, H = 4];

        let row_bytes = H::SIZE * std::mem::size_of::<i32>();
        let allocation_bytes = A::SIZE * row_bytes;

        // The window a `launch` argument hands the device: one row in, out to the allocation's end.
        for start in [0, A::SIZE / 2, A::SIZE - 1] {
            assert_eq!(
                hbm_window::<i32, m![1]>(A::SIZE * H::SIZE, start * H::SIZE),
                start * row_bytes..allocation_bytes
            );
        }
    }

    #[test]
    fn uses_chip_tail() {
        axes![A = 16, C2 = 2, C4 = 4, P = 32];
        type Tail = m![1 # 16];
        type TailMut = m![1 #{!} 16];
        type PackedTail = m![P = 2 # 32];

        let one_chip = HbmTensor::<i32, m![1], m![A], Cpu>::from_parts(Tensor::zeroed());
        assert_eq!(one_chip.view().tile::<m![A], 1, Tail>(1).window().len(), 60);

        let two_chips = HbmTensor::<i32, m![C2], m![A], Cpu>::from_parts(Tensor::zeroed());
        assert_eq!(two_chips.view().tile::<m![A], 1, Tail>(1).window().len(), 60);

        let mut two_chips = HbmTensor::<i32, m![C2], m![A], Cpu>::from_parts(Tensor::zeroed());
        assert_eq!(two_chips.view_mut().tile::<m![A], 1, TailMut>(1).window().len(), 60);

        let four_chips = HbmTensor::<i32, m![C4], m![A], Cpu>::from_parts(Tensor::zeroed());
        assert_eq!(four_chips.view().tile::<m![A], 1, Tail>(1).window().len(), 60);

        let packed = HbmTensor::<i4, m![C2], m![P], Cpu>::from_parts(Tensor::zeroed());
        assert_eq!(packed.view().tile::<m![P], 2, PackedTail>(2).window().len(), 15);
    }

    #[test]
    fn rejects_unaligned_view() {
        axes![C2 = 2, P = 32];
        type PackedTail = m![1 # 32];

        let packed = HbmTensor::<i4, m![C2], m![P], Cpu>::from_parts(Tensor::zeroed());
        assert!(std::panic::catch_unwind(|| packed.view().tile::<m![P], 1, PackedTail>(1).window()).is_err());
    }

    #[test]
    fn rejects_invalid_view_partitions() {
        assert!(std::panic::catch_unwind(|| hbm_window::<i32, m![0]>(64, 0)).is_err());
        assert!(std::panic::catch_unwind(|| hbm_window::<i32, m![2]>(63, 0)).is_err());
        assert!(std::panic::catch_unwind(|| hbm_window::<i32, m![2]>(0, 0)).is_err());
    }

    #[test]
    fn rejects_host_chip_tiles() {
        axes![A = 16, C4 = 4];

        let tensor = HbmTensor::<i32, m![C4], m![A], Cpu>::from_parts(Tensor::zeroed());
        let view = tensor.view().chip_tile::<m![C4], 1, m![1 # 4]>(1);
        assert!(std::panic::catch_unwind(|| view.window()).is_err());

        let mut tensor = HbmTensor::<i32, m![C4], m![A], Cpu>::from_parts(Tensor::zeroed());
        let view = tensor.view_mut().chip_tile::<m![C4], 1, m![1 #{!} 4]>(1);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| view.window())).is_err());
    }

    /// A view of a view accumulates one offset per tile: each `tile` adds its own start, so the second
    /// start is relative to the first. Checked by reading the elements the nested view selects, since
    /// which cells a view names is what the accumulation decides.
    #[test]
    fn hbm_nested_tiles_accumulate_one_offset_each() {
        axes![A = 8, H = 4];

        // Element `a * H + h` holds that flat position, so one value names the cell a view reached.
        let values: Vec<i32> = (0..(A::SIZE * H::SIZE) as i32).collect();
        let table = HbmTensor::<i32, m![1], m![A, H], Cpu>::from_parts(Tensor::from_vec(values));

        let row = table.view().tile::<m![A], 1, m![1 # 8, H]>(3);
        let half = row.tile::<m![H], 2, m![1 # 8, H = 2 # 4]>(2);

        // The tile states its out-of-window cells as padding, so the read spans the declared mapping
        // and the two live cells come first.
        let read = half.inner.read().into_vec();
        assert_eq!(
            read[..2],
            [3 * H::SIZE as i32 + 2, 3 * H::SIZE as i32 + 3],
            "row 3, then 2 columns in"
        );
    }

    #[test]
    fn cpu_redistribution_accepts_an_equivalent_output_factorization() {
        axes![X = 4, Tail = 2];

        let input = DmTensor::<i32, m![1], m![1 # 2], m![1 # 64], m![X / 2 % 2, X % 2, Tail], Cpu>::from_parts(
            Tensor::from_vec((0..2 * 64 * X::SIZE * Tail::SIZE).map(|value| value as i32)),
            None,
        );
        let expected = input.view().inner.read().into_vec();
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let output = input
            .view()
            .chip_shuffle([0])
            .to_dm::<m![X % 4, Tail]>(&mut device.tdma);

        assert_eq!(output.view().inner.read().into_vec(), expected);
    }

    #[test]
    fn cpu_to_dm_moves_a_chip_axis_into_the_element() {
        axes![A = 2, C = 2];

        let input = DmTensor::<i32, m![A], m![1], m![1 # 64], m![C], Cpu>::from_parts(
            Tensor::from_vec((0..A::SIZE * 64 * C::SIZE).map(|value| value as i32)),
            None,
        );
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let output = input.to_dm::<m![2], m![1], m![1 # 64], m![A, C]>(&mut device.tdma);

        let output_chip_size = 64 * A::SIZE * C::SIZE;
        let mut expected = vec![0; 2 * output_chip_size];
        expected[..4].copy_from_slice(&[0, 1, 128, 129]);
        expected[output_chip_size..output_chip_size + 4].copy_from_slice(&[0, 1, 128, 129]);
        assert_eq!(output.view().inner.read().into_vec(), expected);
    }
}

/// A prepared host-to-HBM transfer; see [`HostTensor::to_hbm`].
#[must_use = "transfers do nothing unless awaited"]
#[derive(Debug)]
pub struct ToHbm<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend> {
    host: &'t HostTensor<D, Element, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    _marker: PhantomData<(Chip, Element2)>,
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend>
    ToHbm<'t, D, Element, Chip, Element2, B>
{
    /// Writes into `hbm`'s existing device allocation instead of a fresh one.
    pub fn output(self, hbm: &'t mut HbmTensor<D, Chip, Element2, B>) -> impl Future<Output = Result<(), Error>> {
        B::to_hbm_into(self.host, self.dma, hbm)
    }
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend> IntoFuture
    for ToHbm<'t, D, Element, Chip, Element2, B>
{
    type Output = Result<HbmTensor<D, Chip, Element2, B>, Error>;
    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        B::to_hbm(self.host, self.dma)
    }
}

/// A prepared HBM-to-host transfer; see [`HbmTensor::to_host`].
#[must_use = "transfers do nothing unless awaited"]
#[derive(Debug)]
pub struct ToHost<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend> {
    hbm: &'t HbmTensor<D, Chip, Element, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    _marker: PhantomData<Element2>,
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend>
    ToHost<'t, D, Chip, Element, Element2, B>
{
    /// Reads into `host`'s own memory instead of fresh memory.
    pub fn output(self, host: &'t mut HostTensor<D, Element2, B>) -> impl Future<Output = Result<(), Error>> {
        B::from_hbm_into(self.hbm, self.dma, host)
    }
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend> IntoFuture
    for ToHost<'t, D, Chip, Element, Element2, B>
{
    type Output = Result<HostTensor<D, Element2, B>, Error>;
    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        B::from_hbm(self.hbm, self.dma)
    }
}
