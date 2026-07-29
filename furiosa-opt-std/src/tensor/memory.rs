//! Tensors placed on memory.

use rand::Rng;
use rand::distr::StandardUniform;
use std::any::Any;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::*;

/// Address.
///
/// TODO: check that every address is 64-bit.
pub type Address = u64;

const DMA_SRAM_WRITE_WIDTH: usize = 8;

/// Asserts that a DMA transfer from an `Src`-mapped tensor to a `Dst`-mapped
/// tensor satisfies the hardware DMA layout constraints. Two checks run:
///
/// 1. **Tail alignment** -- the reachable destination tail end (the burst packet
///    the source can feed into the destination element, from [`Mapping::dma_tails`]
///    over the two `Element` payloads) must be a multiple of `min_align` bytes.
/// 2. **Address stride alignment** -- sequencing the full `Dst` access against the
///    full `Src` buffer ([`sequence`] under [`SequencerMode::Carve`]), every stream
///    stride at or past that tail must be `min_align`-aligned (it jumps across packets).
///
/// `min_align` is the hardware DMA access width in bytes:
/// [`DMA_SRAM_WRITE_WIDTH`] for writes into SRAM (HBM→DM, DM→DM), 1 for
/// writes into DRAM (DM→HBM, HBM→HBM).
pub(crate) fn assert_dma_layout<D: Scalar, Src: M, SrcElement: M, Dst: M, DstElement: M>(min_align: usize) {
    assert!(min_align > 0, "min_align must be positive");

    // Tail: `dma_tails` over the two `Element` payloads -- the burst packet is the contiguous run
    // shared WITHIN the elements. It sees just the payload (not the full mappings), so the asymmetric
    // outer classes (only one side carries Cluster/Slice) do not mis-read the packet.
    let packet_end = check_dma_tail::<D>(&SrcElement::to_value(), &DstElement::to_value(), min_align);

    // Address stride: over the FULL Src/Dst layouts (the outer Cluster/Slice strides are what jump
    // across packets and must stay aligned).
    check_dma_address_stride::<D>(&Src::to_value(), &Dst::to_value(), min_align, packet_end);
}

fn check_dma_tail<D: Scalar>(src_element: &Mapping, dst_element: &Mapping, min_align: usize) -> usize {
    let reachable_end = reachable_end(src_element, dst_element);
    let reachable_end_bytes = D::size_in_bytes_from_length(reachable_end);

    assert!(
        reachable_end_bytes.is_multiple_of(min_align),
        "DMA tail alignment violation: reachable destination tail \
         end is not aligned to {min_align} bytes.\n  \
         reachable destination tail end (elements) = {reachable_end}\n  \
         reachable destination tail end (bytes) = {reachable_end_bytes}\n  \
         src element mapping = {src_element:?}\n  \
         dst element mapping = {dst_element:?}",
    );

    reachable_end
}

/// The reachable destination tail: the DMA burst packet the source can feed into the destination
/// element (`dma_tails`'s dst packet over the two element payloads) -- the same in-slice tail the
/// lowering pins the alignment to (see `DmaCommandArgs::dma_shapes`). Shared by `check_dma_tail` and
/// its unit tests, which exercise it without the alignment assertion.
fn reachable_end(src_element: &Mapping, dst_element: &Mapping) -> usize {
    let (_src_packet, dst_packet, _valid) = src_element.dma_tails(dst_element);
    dst_packet
}

fn check_dma_address_stride<D: Scalar>(src: &Mapping, dst: &Mapping, min_align: usize, packet_end: usize) {
    // Carve the destination access pattern (stream) against the source buffer (memory). Each config is
    // keyed by its stream-side (destination) buffer stride; a stride at or past the burst packet jumps
    // across packets, so its byte stride must be `min_align`-aligned. Sequencing (not the factor-algebra
    // division) covers a decomposed/padded destination axis such as `A # 4 / 2, A # 4 % 2`.
    let configs = sequence(&[src], &[dst], SequencerMode::Carve)
        .expect("dma layout: destination stream must be covered by the source");
    for config in &configs {
        for (&stream_stride, _entry) in config.0.iter() {
            if stream_stride < packet_end {
                continue;
            }
            let stride_bytes = D::size_in_bytes_from_length(stream_stride);
            assert!(
                stride_bytes.is_multiple_of(min_align),
                "DMA address stride alignment violation: destination stream stride {stream_stride} \
                 (at or past the burst packet) is {stride_bytes} bytes, not aligned to {min_align}-byte \
                 granularity.\n  \
                 reachable packet end (elements) = {packet_end}\n  \
                 src mapping = {src:?}\n  \
                 dst mapping = {dst:?}",
            );
        }
    }
}

/// Address in the tensor register file.
#[primitive(TrfAddress)]
#[derive(Copy, Clone, Debug)]
pub enum TrfAddress {
    /// Address in the first half of TRF.
    FirstHalf,
    /// Address in the second half of TRF.
    SecondHalf,
    /// Address in the full TRF.
    Full,
}

impl TrfAddress {
    /// Total TRF capacity in bytes for this address mode.
    /// - `Full`: 65,536 bytes (8 lanes × 2 banks × 128 rows × 32 bytes)
    /// - `FirstHalf` / `SecondHalf`: 32,768 bytes (half of Full)
    pub fn capacity(&self) -> usize {
        match self {
            Self::Full => 65_536,
            Self::FirstHalf | Self::SecondHalf => 32_768,
        }
    }
}

impl Display for TrfAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstHalf => write!(f, "TrfAddress::FirstHalf"),
            Self::SecondHalf => write!(f, "TrfAddress::SecondHalf"),
            Self::Full => write!(f, "TrfAddress::Full"),
        }
    }
}

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

    /// Creates a tensor from an initialized buffer. Panics if the buffer length does not match the
    /// mapping size.
    pub fn from_vec(data: impl IntoIterator<Item = D>) -> Self {
        Tensor::from_vec(data).into()
    }

    /// Creates a tensor from a pre-packed device byte image ([`Self::to_buf`]'s inverse), stored as-is.
    /// Contrast [`Self::from_vec`], which packs logical values; pre-packed fp4 weights come through here
    /// to avoid a decode + re-pack round-trip. Panics on a byte-length mismatch.
    pub fn from_buf(buf: Vec<u8>) -> Self {
        Tensor::from_buf(buf).into()
    }

    /// Stages this host tensor into a fresh HBM region assigned by the runtime allocator.
    pub async fn to_hbm<Chip: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Pcie }>,
    ) -> HbmTensor<D, Chip, Element2, B> {
        B::to_hbm(self).await
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
/// `BufStorage` implements as a real `Vec<D>` fill, so those work on Npu / Emulation host-side
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
    // An HBM tensor always carries a concrete address (every producer supplies one and
    // `to_host` reads through it), so this is `Address`, not `Option<Address>`, unlike the
    // on-chip DM/VRF/TRF tensors.
    address: Address,
    // Owns a backend resource (e.g. the Npu device allocation) so it is freed when this tensor
    // drops, not before `launch` reads it. `None` for compiler-managed / `from_addr` tensors.
    owner: Option<Box<dyn Any + Send + Sync>>,
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> std::fmt::Debug for HbmTensor<D, Chip, Element, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HbmTensor")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

// Manual impl: inner `Tensor` is not DeviceSend
impl<D: Scalar, Chip: M, Element: M, B: Backend> crate::runtime::DeviceSend for HbmTensor<D, Chip, Element, B> {}
impl<D: Scalar, Chip: M, Element: M, B: Backend> crate::runtime::DeviceSend for &HbmTensor<D, Chip, Element, B> {}
impl<D: Scalar, Chip: M, Element: M, B: Backend> crate::runtime::DeviceSend for &mut HbmTensor<D, Chip, Element, B> {}
impl<D: Scalar, Chip: M, Element: M, B: Backend> crate::runtime::DeviceSend for HbmTensorView<'_, D, Chip, Element, B> {}
impl<D: Scalar, Chip: M, Element: M, B: Backend> crate::runtime::DeviceSend
    for HbmTensorViewMut<'_, D, Chip, Element, B>
{
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: Address) -> Self {
        Self {
            inner,
            address,
            owner: None,
        }
    }

    pub(crate) fn owns(mut self, owner: impl Any + Send + Sync) -> Self {
        self.owner = Some(Box::new(owner));
        self
    }

    pub(crate) fn inner(&self) -> &Tensor<D, Self::Mapping, B> {
        &self.inner
    }

    /// Returns the HBM address of this tensor.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Size in bytes.
    pub fn size() -> usize {
        Pair::<Chip, Element>::SIZE * std::mem::size_of::<D>()
    }

    /// Converts to host tensor.
    ///
    /// TODO: we should optionally receive the intermediate stream's mapping expression.
    pub async fn to_host<Element2: M>(&self, _dma: &mut DmaContext<{ Dma::Pcie }>) -> HostTensor<D, Element2, B>
    where
        D: MaterializableScalar,
    {
        B::from_hbm(self).await
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
        // Emulation / Npu move the already-packed bytes out; Simulation re-packs. Both agree because
        // they share the `Scalar::store` packing primitive.
        let packed = self.inner.clone().into_buf();
        debug_assert_eq!(packed.len(), D::size_in_bytes_from_length(Pair::<Chip, Element>::SIZE));
        packed
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Creates an HBM tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    #[primitive(HbmTensor::from_addr)]
    pub unsafe fn from_addr(address: Address) -> Self {
        Self::new(Tensor::zeroed(), address)
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Creates an immutable view of the tensor.
    #[primitive(HbmTensor::view)]
    pub fn view<'l>(&'l self) -> HbmTensorView<'l, D, Chip, Element, B> {
        HbmTensorView {
            inner: self.inner.view(),
            address: self.address,
        }
    }

    /// Creates a mutable view of the tensor.
    #[primitive(HbmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> HbmTensorViewMut<'l, D, Chip, Element, B> {
        HbmTensorViewMut {
            inner: self.inner.view_mut(),
            address: self.address,
        }
    }

    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(HbmTensor::to_hbm)]
    pub fn to_hbm<const DMA: Dma, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ DMA }>,
    ) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::new(self.inner.transpose(true), 0)
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

    /// Gather DRAM rows into SRAM at `address`. See [`Self::dma_gather_scaled`].
    #[primitive(HbmTensor::dma_gather_scaled_at)]
    pub fn dma_gather_scaled_at<Cluster2: M, Slice2: M, Element2: M, Element3: M>(
        &self,
        index: &HbmTensor<i32, Chip, Element3, B>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster2, Slice2, Element2, B> {
        let mut output: DmTensor<D, Chip, Cluster2, Slice2, Element2, B> =
            DmTensor::from_parts(Tensor::zeroed(), Some(address));
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
        _dma: &mut DmaContext<{ Dma::Tensor }>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::from_parts(self.inner.transpose(true), None)
    }

    /// Converts to data memory tensor at `address`.
    #[primitive(HbmTensor::to_dm_at)]
    pub fn to_dm_at<Cluster: M, Slice: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::from_parts(self.inner.transpose(true), Some(address))
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
            address: self.address,
            owner: self.owner,
        }
    }
}
// ANCHOR_END: dma_impl

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Shuffles data across clusters on HBM (HBM ↔ HBM DMA). Not yet implemented: an `HbmTensor`
    /// has no cluster dimension (only Chip and Element); clusters are assigned later, at `to_dm`.
    pub fn hbm_cluster_shuffle<const DMA: Dma>(
        &self,
        _dma: &mut DmaContext<{ DMA }>,
        _shuffle_pattern: &[usize],
    ) -> Self {
        todo!(
            "hbm_cluster_shuffle is Under Construction. HbmTensor has no Cluster axis \
             (only Chip + Element); Cluster distribution is decided at .to_dm() time. \
             No current callers. Either the Element axis is meant to encode a Cluster \
             sub-axis (API needs to take that axis explicitly) or the operation belongs \
             on DmTensorView::dm_cluster_shuffle. Pending design review; see the doc \
             comment on hbm_cluster_shuffle."
        )
    }
}

/// View of an HBM tensor.
#[primitive(HbmTensorView)]
#[derive(Debug, Clone)]
pub struct HbmTensorView<'l, D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    inner: TensorView<'l, D, Pair<Chip, Element>, B>,
    // Inherits the HBM tensor's concrete address (always present), so `Address`, not `Option`.
    address: Address,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Element }];

    /// Returns the base HBM address of this view.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Writes to HBM tensor view. The destination's `Chip2` is free of the
    /// source's `Chip`: a read source (Top padding) may target a `view_mut`
    /// destination (Bottom padding). `transpose` validates the live layout.
    #[primitive(HbmTensorView::to_hbm_view)]
    pub fn to_hbm_view<const DMA: Dma, Chip2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ DMA }>,
        mut dst: HbmTensorViewMut<'l, D, Chip2, Element2, B>,
    ) {
        dst.inner.transpose(self.inner, true);
    }

    /// Writes to data memory tensor view.
    #[primitive(HbmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element2, B>,
    ) {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip2 }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        dst.inner.transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip2, Element, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        HbmTensorView {
            inner,
            address: self.address,
        }
    }

    /// Creates immutable views by splitting along a tile expression.
    #[primitive(HbmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip, Element2, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        HbmTensorView {
            inner,
            address: self.address,
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
            address: self.address,
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
        // The `read` tensor's buffer already IS the packed device image on Emulation / Npu, so
        // `into_buf` moves it out directly; Simulation re-packs. Both share `Scalar::store`.
        self.inner.clone().read().into_buf()
    }
}

impl<'l, D: MaterializableScalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensorView::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::from_parts(self.inner.read().transpose(true), None)
    }

    /// Converts to data memory tensor at `address`.
    #[primitive(HbmTensorView::to_dm_at)]
    pub fn to_dm_at<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::from_parts(self.inner.read().transpose(true), Some(address))
    }

    /// Perform chip shuffle using DMA commands (HBM <-> HBM transfer across chips).
    /// This operation redistributes data across chips according to the shuffle pattern.
    ///
    /// Mirrors [`DmTensorView::dm_chip_shuffle`] on the HBM side. Each entry
    /// `shuffle_pattern[target] = source` copies the source chip slot to the target chip slot of
    /// a fresh output HBM tensor — e.g. `[1, 2, 3, 0]` moves chip 1→0, 2→1, 3→2, 0→3.
    #[primitive(HbmTensorView::hbm_chip_shuffle)]
    pub fn hbm_chip_shuffle<const CHIP_DIM: usize, const DMA: Dma>(
        self,
        dma: &mut DmaContext<{ DMA }>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> HbmTensor<D, Chip, Element, B> {
        let mut shuffled: HbmTensor<D, Chip, Element, B> = unsafe { HbmTensor::from_addr(0) };

        for (target_chip_idx, source_chip_idx) in shuffle_pattern.iter().enumerate() {
            self.chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(*source_chip_idx)
                .to_hbm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM, { PaddingKind::Bottom }>>(target_chip_idx),
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
    // Inherits the HBM tensor's concrete address (always present), so `Address`, not `Option`.
    address: Address,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorViewMut<'l, D, Chip, Element, B> {
    /// Returns the base HBM address of this view.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Creates mutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorViewMut::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip2, Element, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        HbmTensorViewMut {
            inner,
            address: self.address,
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip, Element2, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        HbmTensorViewMut {
            inner,
            address: self.address,
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
            address: self.address,
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

    /// Returns the SRAM (DM) address of this tensor, if one was assigned.
    pub fn address(&self) -> Option<Address> {
        self.address
    }
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

    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(DmTensor::to_hbm)]
    pub fn to_hbm<Element2: M>(&self, _dma: &mut DmaContext<{ Dma::Tensor }>) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::new(self.inner.transpose(true), 0)
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

    /// Converts to data memory tensor. A DM → DM transfer only relayouts the `Element` payload;
    /// the `Slice` partition size is preserved (`Slice::SIZE == Slice2::SIZE`).
    #[primitive(DmTensor::to_dm)]
    pub fn to_dm<Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Slice }, { Element }],
            Element,
            m![{ Cluster }, { Slice2 }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::from_parts(self.inner.transpose(true), None)
    }

    /// Converts to data memory tensor at `address`. See [`Self::to_dm`].
    #[primitive(DmTensor::to_dm_at)]
    pub fn to_dm_at<Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Slice }, { Element }],
            Element,
            m![{ Cluster }, { Slice2 }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        DmTensor::from_parts(self.inner.transpose(true), Some(address))
    }

    /// Copies into a fresh DM tensor via parallel copy. Like [`Self::to_dm`], the `Slice` size is
    /// preserved (`Slice::SIZE == Slice2::SIZE`).
    #[primitive(DmTensor::to_dm_pcopy)]
    pub fn to_dm_pcopy<Slice2: M, Element2: M>(
        &self,
        _sub: &mut TuContext<{ Tu::Sub }>,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        DmTensor::from_parts(self.inner.transpose(true), None)
    }

    /// Copies into a fresh DM tensor at `address` via parallel copy. See [`Self::to_dm_pcopy`].
    #[primitive(DmTensor::to_dm_pcopy_at)]
    pub fn to_dm_pcopy_at<Slice2: M, Element2: M>(
        &self,
        _sub: &mut TuContext<{ Tu::Sub }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        DmTensor::from_parts(self.inner.transpose(true), Some(address))
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
        constraints::assert_reshape_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
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
        _dma: &mut DmaContext<{ Dma::Tensor }>,
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
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B>,
    ) {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Slice }, { Element }],
            Element,
            m![{ Cluster2 }, { Slice2 }, { Element2 }],
            Element2,
        >(DMA_SRAM_WRITE_WIDTH);
        dst.inner.transpose(self.inner, true);
    }

    /// Writes data to a mutable tensor view for data memory.
    #[primitive(DmTensorView::to_dm_view_pcopy)]
    pub fn to_dm_view_pcopy<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
        _sub: &mut TuContext<{ Tu::Sub }>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B>,
    ) {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
        dst.inner.transpose(self.inner, false);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Cluster.
    #[primitive(DmTensorView::cluster_tile)]
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster2, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster2, Slice, Slice>();
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorView::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice2, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
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
        constraints::assert_reshape_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
        DmTensorView {
            inner: unsafe {
                self.inner
                    .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
            },
        }
    }

    /// Redistributes data across clusters by DMA (DM ↔ DM): `shuffle_pattern[target] = source`
    /// copies the source cluster to the target cluster — e.g. `[1, 0]` swaps clusters 0 and 1.
    #[primitive(DmTensorView::dm_cluster_shuffle)]
    pub fn dm_cluster_shuffle<const CLUSTER_DIM: usize>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }>,
        shuffle_pattern: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        let mut shuffled: DmTensor<D, Chip, Cluster, Slice, Element, B> = DmTensor::new();

        for (target_cluster_idx, source_cluster_idx) in shuffle_pattern.iter().enumerate() {
            self.cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(*source_cluster_idx)
                .to_dm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM, { PaddingKind::Bottom }>>(
                            target_cluster_idx,
                        ),
                );
        }

        shuffled
    }

    /// Redistributes data across chips by Tensor DMA (DM ↔ DM): `shuffle_pattern[target] = source`
    /// copies the source chip to the target chip — e.g. `[1, 2, 3, 0]` moves chip 1→0, 2→1, 3→2, 0→3.
    #[primitive(DmTensorView::dm_chip_shuffle)]
    pub fn dm_chip_shuffle<const CHIP_DIM: usize>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        let mut shuffled: DmTensor<D, Chip, Cluster, Slice, Element, B> = DmTensor::new();

        for (target_chip_idx, source_chip_idx) in shuffle_pattern.iter().enumerate() {
            self.chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(*source_chip_idx)
                .to_dm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM, { PaddingKind::Bottom }>>(target_chip_idx),
                );
        }

        shuffled
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>
{
    /// Creates mutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorViewMut::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Cluster.
    #[primitive(DmTensorViewMut::cluster_tile)]
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster2, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster2, Slice, Slice>();
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
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
        constraints::assert_reshape_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
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
    #[expect(dead_code)]
    address: Option<TrfAddress>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
// ANCHOR_END: trf_tensor_def

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: Option<TrfAddress>) -> Self {
        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Creates a TRF tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    pub unsafe fn from_addr(address: TrfAddress) -> Self {
        Self::new(Tensor::zeroed(), Some(address))
    }
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
    #[expect(dead_code)]
    address: Option<Address>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
// ANCHOR_END: vrf_tensor_def

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: Option<Address>) -> Self {
        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Creates a VRF tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    pub unsafe fn from_addr(address: Address) -> Self {
        Self::new(Tensor::zeroed(), Some(address))
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
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

    use crate::backend::{Emulation, Typecheck};
    use crate::scalar::Scalar;

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

        let table = HbmTensor::<i32, m![1], m![W, V], B>::new(Tensor::from_vec(table_buf), 0);
        // The index lives in DM (SPM): `Slice = K`, the residue axis the gather iterates.
        let index = DmTensor::<i32, m![1], m![1], m![K], m![1], B>::from_parts(Tensor::from_vec(idx_buf), None);

        let output: DmTensor<i32, m![1], m![1], m![K], m![V], B> = table.dma_gather_unscaled(&index);
        (output.inner.into_vec(), expected)
    }

    /// `dma_gather_unscaled` on `Emulation`: the physical `BufStorage` gather (driven by the
    /// sequencer) matches the hand oracle. Peer of the byte-offset `dma_gather_scaled` and of the
    /// `Tensor`-level `emulation_write_gather_roundtrip_unscaled`.
    #[test]
    fn emulation_dma_gather_unscaled_roundtrip() {
        let (got, expected) = run_dma_gather_unscaled::<Emulation>();
        assert_eq!(got, expected);
    }

    /// Typecheck backend: `dma_gather_unscaled` propagates the same shape assertions
    /// (`gather_params` mapping algebra) as the scaled gather without iterating any buffer. The
    /// output tensor under Typecheck has no values; this only pins that the call does not panic for
    /// a well-formed block-table shape (visa->LIR lowering is pinned by the `compare_edf!` test).
    #[test]
    fn typecheck_dma_gather_unscaled_runs_assertion_only() {
        axes![W = 8, V = 2, K = 64];

        let table = HbmTensor::<i32, m![1], m![W, V], Typecheck>::new(Tensor::zeroed(), 0);
        let index = DmTensor::<i32, m![1], m![1], m![K], m![1], Typecheck>::from_parts(Tensor::zeroed(), None);
        let _output: DmTensor<i32, m![1], m![1], m![K], m![V], Typecheck> = table.dma_gather_unscaled(&index);
    }

    /// `reshape` consumes `self`, so it must hand the backend resource (`owner`) to the reshaped
    /// handle. Releasing it here would free the device allocation while the returned handle still
    /// names its address, so the next `launch` would drive a kernel over freed HBM. Only a handle
    /// that owns its allocation (`Kernel::write`, `From<Buffer>`) can show this, which is why the
    /// owner is observed through its `Drop`.
    #[test]
    fn reshape_hands_owner_to_reshaped_handle() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        axes![A = 4, B = 2, AB = 8];

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let freed = Arc::new(AtomicBool::new(false));
        let tensor = HbmTensor::<i32, m![1], m![A, B], Emulation>::new(Tensor::zeroed(), 0x1000)
            .owns(DropFlag(Arc::clone(&freed)));

        // Merging `[A, B]` into `[AB]` keeps the wire order, so the relabel moves no data.
        let reshaped = unsafe { tensor.reshape::<m![1], m![AB]>() };

        assert!(
            !freed.load(Ordering::SeqCst),
            "reshape released the backend resource; the reshaped handle's address now dangles"
        );
        assert_eq!(reshaped.address(), 0x1000);
    }

    #[test]
    fn unittest_extents_reachable_end_with_dst_padding_absorb() {
        axes![A = 8, B = 3];
        // matched B (directcast, [1,3)) + divisor padding [3,8) extend the tail.
        // matched A is non-directcast (divisor_stride=8 ≠ dividend_stride=3),
        // so the walk stops at 8 — A's iteration breaks src-side contiguity.
        assert_eq!(reachable_end(&<m![A, B]>::to_value(), &<m![A, B # 8]>::to_value()), 8);
    }

    #[test]
    fn unittest_extents_reachable_end_invariant_under_outer_cluster_slice() {
        axes![Cl = 2, Sl = 4, A = 3];
        // The tail check looks only at divisor-side spans, so adding outer
        // cluster/slice axes to the source must produce the same answer.
        assert_eq!(
            reachable_end(&<m![A]>::to_value(), &<m![A # 16]>::to_value()),
            reachable_end(&<m![Cl, Sl, A]>::to_value(), &<m![A # 16]>::to_value()),
        );
    }

    #[test]
    fn unittest_extents_reachable_end_single_element_underflows_alignment() {
        axes![A = 1];
        // Single i32 tail = 4 bytes; not aligned to DMA_SRAM_WRITE_WIDTH (= 8).
        let end = reachable_end(&<m![A]>::to_value(), &<m![A]>::to_value());
        assert_eq!(end, 1);
        assert_eq!(<i32 as Scalar>::size_in_bytes_from_length(end), 4);
        assert!(!<i32 as Scalar>::size_in_bytes_from_length(end).is_multiple_of(DMA_SRAM_WRITE_WIDTH));
    }

    #[test]
    fn unittest_assert_dma_layout_canonical_cluster_slice_passes() {
        // End-to-end wrapper test on a realistic DM-tier shape:
        // outer Cluster/Slice partitioning, inner element data.
        axes![Cl = 2, Sl = 4, A = 8, B = 4];
        assert_dma_layout::<i32, m![Cl, Sl, A, B], m![A, B], m![Cl, Sl, A, B], m![A, B]>(DMA_SRAM_WRITE_WIDTH);
    }

    #[test]
    fn unittest_assert_dma_layout_dst_padding_absorbed() {
        axes![A = 8, B = 3];
        assert_dma_layout::<i32, m![A, B], m![A, B], m![A, B # 8], m![A, B # 8]>(DMA_SRAM_WRITE_WIDTH);
    }

    #[test]
    fn unittest_assert_dma_layout_min_align_one_is_noop() {
        // DM→HBM / HBM→HBM use min_align = 1, where both the tail-end check
        // and the stride-alignment check trivially pass. This pins that
        // contract so future refactors of either check cannot regress the
        // DRAM-write path.
        axes![A = 1];
        assert_dma_layout::<i32, m![A], m![A], m![A], m![A]>(1);

        axes![Cl = 2, Sl = 4, B = 3];
        assert_dma_layout::<i32, m![Cl, Sl, B], m![B], m![Cl, Sl, B # 7], m![B # 7]>(1);
    }

    #[test]
    fn unittest_assert_dma_layout_decomposed_padded_axis() {
        // Destination splits a padded axis: `A` (live 3) padded to 4, then `(A # 4) / 2, (A # 4) % 2`.
        // The factor-algebra division does not surface the `/ 2` outer stride, so it never checked it;
        // sequencing enumerates every stream stride. For i32 (4 B) the `% 2` packet is 8 B and the
        // `/ 2` stride is 8 B, both aligned, so the layout passes.
        axes![Cl = 2, Sl = 4, A = 3];
        assert_dma_layout::<i32, m![Cl, Sl, A], m![A], m![Cl, Sl, A # 4 / 2, A # 4 % 2], m![A # 4 / 2, A # 4 % 2]>(
            DMA_SRAM_WRITE_WIDTH,
        );
    }

    /// A packed sub-byte load whose flat source element (`m![A, B]`, 32768 elements) feeds one
    /// 128-element period of a modulo-decomposed DM tile. `dma_tails` compares their semantic prefix,
    /// skipping the sixteen affine B rows despite the different factorization, so `reachable_end` is
    /// the full 128-element period (64 bytes), which is `min_align(8)`-aligned.
    #[test]
    fn unittest_assert_dma_layout_packed_subbyte_sliced_load() {
        use crate::scalar::f4e2m1;
        // A packed sub-byte load whose innermost axis is a fraction of `min_align` bytes (`B = 8`
        // `f4e2m1` = 4 bytes), feeding a sliced, modulo-decomposed DM tile.
        axes![A = 4096, B = 8];
        assert_dma_layout::<
            f4e2m1,
            m![1, A, B],
            m![A, B],
            m![1, 1 # 2, A / 16, A / 8 % 2, A % 8, B],
            m![A / 8 % 2, A % 8, B],
        >(DMA_SRAM_WRITE_WIDTH);
    }
}
