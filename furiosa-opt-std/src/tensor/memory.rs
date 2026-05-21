//! Tensors placed on memory.

use rand::Rng;
use rand::distr::StandardUniform;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::raw::gen_axes;
use crate::tensor::{BufferConvertError, *};

/// Address.
///
/// TODO: check that every address is 64-bit.
pub type Address = u64;

const DMA_SRAM_WRITE_WIDTH: usize = 8;

/// Asserts that a DMA transfer from an `Src`-mapped tensor to a `Dst`-mapped
/// tensor satisfies the hardware DMA layout constraints. Two checks share a
/// single `divide(src, dst)` and a single [`Extents`] computation:
///
/// 1. **Tail alignment** — the reachable destination tail end (the largest
///    contiguous packet the destination can expose, walked from stride 1 via
///    [`Extents::contiguous_tail`]) must be a multiple of `min_align` bytes.
///    This part looks only at divisor-side spans, so it is invariant under
///    any outer src structure (cluster/slice partitioning).
/// 2. **Address stride alignment** — matched terms whose `divisor_stride`
///    reaches at or past the contiguous tail must have that stride aligned
///    to `min_align` bytes (their iteration jumps across packets).
///
/// `min_align` is the hardware DMA access width in bytes:
/// [`DMA_SRAM_WRITE_WIDTH`] for writes into SRAM (HBM→DM, DM→DM), 1 for
/// writes into DRAM (DM→HBM, HBM→HBM).
pub(crate) fn assert_dma_layout<D: Scalar, Src: M, Dst: M>(min_align: usize) {
    assert!(min_align > 0, "min_align must be positive");
    let src = Src::to_value().factorize();
    let dst = Dst::to_value().factorize();
    let extents = src
        .clone()
        .divide(dst.clone())
        .extents()
        .unwrap_or_else(|_| panic!("DMA layout check could not build extents for src={src:?} dst={dst:?}"));

    // `check_dma_tail` returned, so `D::size_in_bytes_from_length(packet_end)`
    // is a multiple of `min_align`. `check_dma_address_stride` consumes that
    // invariant below.
    let packet_end = check_dma_tail::<D>(&extents, &src, &dst, min_align);
    check_dma_address_stride::<D>(&extents, &src, &dst, min_align, packet_end);
}

fn check_dma_tail<D: Scalar>(extents: &Extents, src: &FMapping, dst: &FMapping, min_align: usize) -> usize {
    let reachable_end = extents.contiguous_tail();
    let reachable_end_bytes = D::size_in_bytes_from_length(reachable_end);

    assert!(
        reachable_end_bytes.is_multiple_of(min_align),
        "DMA tail alignment violation: reachable destination tail \
         end is not aligned to {min_align} bytes.\n  \
         reachable destination tail end (elements) = {reachable_end}\n  \
         reachable destination tail end (bytes) = {reachable_end_bytes}\n  \
         src mapping = {src:?}\n  \
         dst mapping = {dst:?}",
    );

    reachable_end
}

fn check_dma_address_stride<D: Scalar>(
    extents: &Extents,
    src: &FMapping,
    dst: &FMapping,
    min_align: usize,
    packet_end: usize,
) {
    for bound in extents.matched.iter().filter(|b| b.term.divisor_stride >= packet_end) {
        let dst_stride = bound.term.divisor_stride;
        let dst_bytes = D::size_in_bytes_from_length(dst_stride);
        assert!(
            dst_bytes.is_multiple_of(min_align),
            "DMA address stride alignment violation: matched term {term:?} beginning at \
             or past the contiguous tail has dst stride {dst_bytes} bytes, not aligned to \
             {min_align}-byte granularity.\n  \
             reachable packet end (elements) = {packet_end}\n  \
             src mapping = {src:?}\n  \
             dst mapping = {dst:?}",
            term = bound.term.term,
        );
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
#[primitive(HostTensor)]
#[derive(Debug, Clone)]
pub struct HostTensor<D: Scalar, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Element, B>,
}

impl<D: Scalar, Element: M, B: Backend> From<Tensor<D, Element, B>> for HostTensor<D, Element, B> {
    fn from(inner: Tensor<D, Element, B>) -> Self {
        Self { inner }
    }
}

impl<D: Scalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Mapping type alias.
    pub type Mapping = Element;

    pub(crate) fn inner_tensor(&self) -> &Tensor<D, Element, B> {
        &self.inner
    }

    /// Creates a tensor from an initialized buffer.
    pub fn from_buf(data: impl IntoIterator<Item = D>) -> Self {
        Tensor::from_buf(data).into()
    }

    /// Creates a tensor from an initialized buffer, returning an error if the input cannot be
    /// lowered.
    pub fn try_from_buf(data: impl IntoIterator<Item = D>) -> Result<Self, BufferConvertError> {
        Tensor::try_from_buf(data).map(Into::into)
    }

    /// Converts to HBM tensor.
    ///
    /// TODO: `address` should be optional.
    pub async fn to_hbm<Chip: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Pcie }>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, B> {
        B::to_hbm(self, address).await
    }

    /// Consumes self and returns the inner tensor.
    pub fn into_inner(self) -> Tensor<D, Self::Mapping, B> {
        self.inner
    }

    /// Consumes self and returns the underlying backend-native raw tensor.
    pub fn into_raw(self) -> B::RawTensor<D> {
        self.inner.into_raw()
    }

    /// Returns the tensor data as a flat `Vec<D>`. On Simulation, panics if any slot is
    /// `Opt::Uninit`.
    pub fn to_buf(&self) -> Vec<D> {
        self.inner.to_buf()
    }
}

/// Host-side `HostTensor` constructors. Bound to `Backend`; under `B = Npu` / `Emulation` the
/// value-iterating methods (`zero`, `rand`) bottom out in `RawTensor::from_fn`, which on
/// `BufRawTensor` panics through its `write_index` `todo!()`. `from_buf` / `from_safetensors`
/// route through `RawTensor::try_from_buf` / `RawTensor::from_buf`, which `BufRawTensor`
/// implements as a real `Vec<D>` clone — so those work on Npu host-side staging too.
impl<D: Scalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Creates a tensor filled with zeros.
    pub fn zero() -> Self
    where
        D: num_traits::Zero,
    {
        Tensor::zero().into()
    }

    /// Creates a tensor filled with random values.
    #[primitive(HostTensor::rand)]
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Tensor::rand(rng).into()
    }

    /// Creates an uninitialized tensor.
    pub fn uninit() -> Self {
        Tensor::uninit().into()
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
        Ok(Tensor::from_buf(view.data().chunks_exact(stride).map(D::from_le_bytes)).into())
    }
}

/// `Opt`-buffer constructors and serializers — gated on `B::RawTensor<D>: RawTensorOpt<D>`
/// (Simulation / Typecheck only). Mirrors the equivalent `Tensor` surface.
impl<D: Scalar, Element: M, B: Backend> HostTensor<D, Element, B>
where
    B::RawTensor<D>: RawTensorOpt<D>,
{
    /// Creates a tensor from a logical `Opt<D>` buffer.
    pub fn from_opt_buf(data: impl IntoIterator<Item = Opt<D>>) -> Self {
        Tensor::from_opt_buf(data).into()
    }

    /// Creates a tensor from a logical `Opt<D>` buffer, returning an error if the input cannot
    /// be lowered.
    pub fn try_from_opt_buf(data: impl IntoIterator<Item = Opt<D>>) -> Result<Self, BufferConvertError> {
        Tensor::try_from_opt_buf(data).map(Into::into)
    }

    /// Returns the tensor data as a flat logical `Opt<D>` buffer in physical layout order.
    pub fn to_buf_opt(&self) -> Vec<Opt<D>> {
        self.inner.to_buf_opt()
    }
}

/// Tensor stored in HBM memory.
#[primitive(HbmTensor)]
#[derive(Debug)]
pub struct HbmTensor<D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Pair<Chip, Element>, B>,
    address: Address,
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
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: Address) -> Self {
        Self { inner, address }
    }

    pub(crate) fn inner_tensor(&self) -> &Tensor<D, Self::Mapping, B> {
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
    pub async fn to_host<Element2: M>(&self, _dma: &mut DmaContext<{ Dma::Pcie }>) -> HostTensor<D, Element2, B> {
        B::from_hbm(self).await
    }

    /// Returns the tensor data as a flat `Vec<D>` in HBM byte-layout order
    /// (`m![Chip, Element]` C-major). This is the raw byte view as it would
    /// sit in HBM memory, suitable for feeding the LIR executor.
    pub fn to_buf(&self) -> Vec<D> {
        self.inner.to_buf()
    }
}

/// `Opt`-aware extra accessors. Gated on `B::RawTensor<D>: RawTensorOpt<D>` so only backends
/// whose storage represents `Opt<D>` (Simulation, Typecheck) surface them.
impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B>
where
    B::RawTensor<D>: crate::tensor::raw::RawTensorOpt<D>,
{
    /// Like [`HbmTensor::to_buf`], but replaces `Opt::Uninit` slots with `D::zero()`.
    /// Use this for output tensors (allocated via `from_addr`) where the
    /// contents are uninitialized but a zero-filled buffer of the right size
    /// is needed.
    pub fn to_buf_or_default(&self) -> Vec<D> {
        self.inner
            .to_buf_opt()
            .into_iter()
            .map(|x| match x {
                Opt::Init(v) => v,
                Opt::Uninit => D::zero(),
            })
            .collect()
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
        let axes = gen_axes::<Pair<Chip, Element>>();
        Self::new(Tensor::from_inner(B::RawTensor::uninit_from_axes(axes)), address)
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensor::view)]
    pub fn view<'l>(&'l self) -> HbmTensorView<'l, D, Chip, Element, B> {
        HbmTensorView {
            inner: self.inner.view(),
            address: self.address,
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> HbmTensorViewMut<'l, D, Chip, Element, B> {
        HbmTensorViewMut {
            inner: self.inner.view_mut(),
            address: self.address,
        }
    }

    /// Converts to HBM tensor.
    #[primitive(HbmTensor::to_hbm)]
    pub fn to_hbm<const DMA: Dma, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ DMA }>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::new(self.inner.transpose(true), address)
    }

    /// Gather DRAM rows into SRAM at positions given by index tensor.
    ///
    /// Implements `index_select` along the table's gather-key axis (the axis present in
    /// `Element` but not in the output's `Element2`). The output's indices axes (in
    /// `Element2`, mirroring `Element3` from the index tensor) replace that gather-key axis:
    /// `output[..pre, k, ..post] = self[..pre, index[k], ..post]`.
    ///
    /// Inverse of [`DmTensor::dma_scatter`]. `scaled` matches `dma_scatter`'s semantics:
    /// `true` interprets index values as byte-offsets along the gather axis (divided
    /// internally by `payload_size * sizeof(D)` to recover the row position); `false`
    /// treats index values as raw row-position indices.
    #[primitive(HbmTensor::dma_gather)]
    pub fn dma_gather<Cluster2: M, Slice2: M, Element2: M, Element3: M>(
        &self,
        index: &HbmTensor<i32, Chip, Element3, B>,
        address: Address,
        scaled: bool,
    ) -> DmTensor<D, Chip, Cluster2, Slice2, Element2, B> {
        let mut output: DmTensor<D, Chip, Cluster2, Slice2, Element2, B> = unsafe { DmTensor::from_addr(address) };
        self.inner.write_gather::<_, _>(&mut output.inner, &index.inner, scaled);
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
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<D, m![{ Chip }, { Element }], Element2>(DMA_SRAM_WRITE_WIDTH);
        DmTensor::new(self.inner.transpose(true), address)
    }
}
// ANCHOR_END: dma_impl

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Perform cluster shuffle operation using DMA commands (HBM <-> HBM transfer).
    /// This operation redistributes data across clusters according to the shuffle pattern.
    ///
    /// # Arguments
    /// * `dma` - DMA context (Tensor DMA or PCIe DMA)
    /// * `shuffle_pattern` - Array mapping source cluster to destination cluster
    ///
    /// # Under Construction
    ///
    /// This API is semantically suspect and has no current callers.
    ///
    /// `HbmTensor` exposes only `Chip + Element` axes. There is no Cluster axis
    /// on HBM. Cluster distribution is decided at `to_dm` time, when an HBM
    /// payload is sharded across cluster SRAMs. So a "cluster shuffle on HBM"
    /// is not a well-defined operation in the current mapping algebra. It
    /// could mean any of:
    ///
    /// 1. The Element axis is expected to encode a Cluster sub-axis the caller
    ///    wants to permute. If so, the API should require that and accept the
    ///    axis explicitly, mirroring the shape of `tile` / `chip_tile`.
    /// 2. The intent is to shuffle while data is in DM, in which case the
    ///    operation belongs on `DmTensorView::dm_cluster_shuffle` (which is
    ///    implemented) and the round-trip through HBM is the caller's job.
    ///
    /// Until that design call is made, this function panics. Do not depend on
    /// the signature being stable.
    pub fn hbm_cluster_shuffle<const DMA: Dma>(
        &self,
        _dma: &mut DmaContext<{ DMA }>,
        _shuffle_pattern: &[usize],
    ) -> HbmTensor<D, Chip, Element, B> {
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
    address: Address,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Element }];

    /// Returns the base HBM address of this view.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Writes to HBM tensor view.
    #[primitive(HbmTensorView::to_hbm_view)]
    pub fn to_hbm_view<const DMA: Dma, Element2: M>(
        self,
        _dma: &mut DmaContext<{ DMA }>,
        mut dst: HbmTensorViewMut<'l, D, Chip, Element2, B>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Writes to data memory tensor view.
    pub fn to_dm_view<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2, B>,
    ) {
        assert_dma_layout::<D, m![{ Chip }, { Element }], Element2>(DMA_SRAM_WRITE_WIDTH);
        dst.inner.write_transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
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

    /// Returns the view data as a flat `Vec<D>` in HBM byte-layout order
    /// (`m![Chip, Element]` C-major). Reads the view into a temporary tensor
    /// and serializes that.
    pub fn to_buf(&self) -> Vec<D> {
        self.inner.clone().read().to_buf()
    }
}

/// `Opt`-aware accessor for views, see [`HbmTensor::to_buf_or_default`].
impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B>
where
    B::RawTensor<D>: crate::tensor::raw::RawTensorOpt<D>,
{
    /// Like [`HbmTensorView::to_buf`], but replaces `Opt::Uninit` slots with `D::zero()`.
    pub fn to_buf_or_default(&self) -> Vec<D> {
        self.inner
            .clone()
            .read()
            .to_buf_opt()
            .into_iter()
            .map(|x| match x {
                Opt::Init(v) => v,
                Opt::Uninit => D::zero(),
            })
            .collect()
    }
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensorView::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<D, m![{ Chip }, { Element }], Element2>(DMA_SRAM_WRITE_WIDTH);
        DmTensor::new(self.inner.read().transpose(true), address)
    }

    /// Perform chip shuffle using DMA commands (HBM <-> HBM transfer across chips).
    /// This operation redistributes data across chips according to the shuffle pattern.
    ///
    /// Mirrors [`DmTensorView::dm_chip_shuffle`] on the HBM side. Each entry
    /// `shuffle_pattern[target] = source` means the source chip slot is copied
    /// to the target chip slot of a fresh output HBM tensor.
    ///
    /// # Arguments
    /// * `dma` - DMA context (Tensor DMA or PCIe DMA)
    /// * `shuffle_pattern` - Array mapping target chip index to source chip index
    ///
    /// # Example
    /// For `Chip=4` with `shuffle_pattern=[1, 2, 3, 0]`:
    /// - Data from Chip 1 goes to Chip 0
    /// - Data from Chip 2 goes to Chip 1
    /// - Data from Chip 3 goes to Chip 2
    /// - Data from Chip 0 goes to Chip 3
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
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(target_chip_idx),
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
    address: Address,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorViewMut<'l, D, Chip, Element, B> {
    /// Returns the base HBM address of this view.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Creates mutable views by splitting along a tile expression over Chip.
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &mut self,
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
        &mut self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip, Element2, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        HbmTensorViewMut {
            inner,
            address: self.address,
        }
    }
}

/// Tensor stored in data memory.
#[primitive(DmTensor)]
#[derive(Debug)]
pub struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
    address: Address,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: Address) -> Self {
        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Creates a DM tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    #[primitive(DmTensor::from_addr)]
    pub unsafe fn from_addr(address: Address) -> Self {
        let axes = gen_axes::<Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>>();
        Self::new(Tensor::from_inner(B::RawTensor::uninit_from_axes(axes)), address)
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

    /// Converts to HBM tensor.
    #[primitive(DmTensor::to_hbm)]
    pub fn to_hbm<Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::new(self.inner.transpose(true), address)
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
    #[primitive(DmTensor::dma_scatter)]
    pub fn dma_scatter<Key: M, Element2: M, Element3: M>(
        &self,
        index: &HbmTensor<i32, Chip, Element3, B>,
        output: &mut HbmTensor<D, Chip, Element2, B>,
        scaled: bool,
    ) {
        let src = Pair::<Slice, Element>::to_value().factorize();
        let key = Key::to_value().factorize();
        assert!(
            src.clone().divide(key).exact_checked().is_ok(),
            "scatter key `{:?}` must be fully contained in source `{src:?}`. \
             If the key axis is split across Chip and Element, indirect DMA cannot address it.",
            Key::to_value().factorize()
        );

        self.inner
            .write_scatter::<Key, _, _>(&mut output.inner, &index.inner, scaled);
    }

    /// Converts to data memory tensor.
    pub fn to_dm<Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        assert_dma_layout::<D, m![{ Cluster }, { Slice }, { Element }], Element2>(DMA_SRAM_WRITE_WIDTH);
        DmTensor::new(self.inner.transpose(true), address)
    }

    /// Copies data to another DM tensor via parallel copy.
    ///
    /// Convenience wrapper: `self.view().to_dm_view_pcopy(sub, dst.view_mut())`.
    pub fn to_dm_pcopy<Slice2: M, Element2: M>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
        dst: &mut DmTensor<D, Chip, Cluster, Slice2, Element2, B>,
    ) {
        self.view().to_dm_view_pcopy(sub, dst.view_mut());
    }

    /// Reshapes the tensor to a different mapping at the same address.
    ///
    /// Delegates to [`Tensor::reshape`].
    ///
    /// # Safety
    ///
    /// The caller must ensure that the src/dst sizes match at every level of
    /// the DM mapping, i.e. `Chip::SIZE == Chip2::SIZE`,
    /// `Cluster::SIZE == Cluster2::SIZE`, `Slice::SIZE == Slice2::SIZE`, and
    /// `Element::SIZE == Element2::SIZE`.
    #[primitive(DmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        assert_eq!(Chip::SIZE, Chip2::SIZE);
        assert_eq!(Cluster::SIZE, Cluster2::SIZE);
        assert_eq!(Slice::SIZE, Slice2::SIZE);
        assert_eq!(Element::SIZE, Element2::SIZE);
        let reshaped = unsafe {
            self.inner
                .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
        };
        DmTensor::new(reshaped, self.address)
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
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// Writes data to a mutable tensor view for HBM.
    #[primitive(DmTensorView::to_hbm_view)]
    pub fn to_hbm_view<Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: HbmTensorViewMut<'l, D, Chip, Element2, B>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Writes data to a mutable tensor view for data memory.
    #[primitive(DmTensorView::to_dm_view)]
    pub fn to_dm_view<Slice2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice2, Element2, B>,
    ) {
        assert_dma_layout::<D, m![{ Cluster }, { Slice }, { Element }], Element2>(DMA_SRAM_WRITE_WIDTH);
        dst.inner.write_transpose(self.inner, true);
    }

    /// Writes data to a mutable tensor view for data memory.
    pub fn to_dm_view_pcopy<Slice2: M, Element2: M>(
        self,
        _sub: &mut TuContext<{ Tu::Sub }>,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice2, Element2, B>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip2, Cluster, Slice, Element, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Cluster.
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster2, Slice, Element, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorView::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice2, Element, B> {
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

    /// Perform cluster shuffle operation using DMA commands (DM <-> DM transfer).
    /// This operation redistributes data across clusters according to the shuffle pattern.
    ///
    /// # Arguments
    /// * `dma` - Tensor DMA context
    /// * `shuffle_pattern` - Array mapping source cluster to destination cluster
    ///
    /// # Example
    /// For Cluster=2 with shuffle_pattern=\[1,0\]:
    /// - Data from Cluster 1 goes to Cluster 0
    /// - Data from Cluster 0 goes to Cluster 1
    #[primitive(DmTensorView::dm_cluster_shuffle)]
    pub fn dm_cluster_shuffle<const CLUSTER_DIM: usize>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }>,
        shuffle_pattern: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        let mut shuffled: DmTensor<D, Chip, Cluster, Slice, Element, B> = unsafe { DmTensor::from_addr(0) };

        for (target_cluster_idx, source_cluster_idx) in shuffle_pattern.iter().enumerate() {
            self.cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(*source_cluster_idx)
                .to_dm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(target_cluster_idx),
                );
        }

        shuffled
    }

    /// Perform chip shuffle using Tensor DMA commands (DM <-> DM transfer across chips).
    /// This operation redistributes data across chips according to the shuffle pattern.
    ///
    /// # Arguments
    /// * `dma` - Tensor DMA context
    /// * `shuffle_pattern` - Array mapping source chip to destination chip
    ///
    /// # Example
    /// For Chip=4 with shuffle_pattern=\[1,2,3,0\]:
    /// - Data from Chip 1 goes to Chip 0
    /// - Data from Chip 2 goes to Chip 1
    /// - Data from Chip 3 goes to Chip 2
    /// - Data from Chip 0 goes to Chip 3
    #[primitive(DmTensorView::dm_chip_shuffle)]
    pub fn dm_chip_shuffle<const CHIP_DIM: usize>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        let mut shuffled: DmTensor<D, Chip, Cluster, Slice, Element, B> = unsafe { DmTensor::from_addr(0) };

        for (target_chip_idx, source_chip_idx) in shuffle_pattern.iter().enumerate() {
            self.chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(*source_chip_idx)
                .to_dm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(target_chip_idx),
                );
        }

        shuffled
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>
{
    /// Creates mutable views by splitting along a tile expression over Chip.
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &mut self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Cluster.
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &mut self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster2, Slice, Element, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &mut self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2, B> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }
}

// ANCHOR: trf_tensor_def
/// Tensor stored in the tensor register file.
#[primitive(TrfTensor)]
#[derive(Debug)]
pub struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Element>>>>, B>,
    #[expect(dead_code)]
    address: TrfAddress,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
// ANCHOR_END: trf_tensor_def

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: TrfAddress) -> Self {
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
        let axes = gen_axes::<Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Element>>>>>();
        Self::new(Tensor::from_inner(B::RawTensor::uninit_from_axes(axes)), address)
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
    address: Address,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
// ANCHOR_END: vrf_tensor_def

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>, address: Address) -> Self {
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
        let axes = gen_axes::<Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>>();
        Self::new(Tensor::from_inner(B::RawTensor::uninit_from_axes(axes)), address)
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
    /// Mapping type alias.
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

    use crate::scalar::Scalar;

    fn reachable_end<Src: M, Dst: M>() -> usize {
        Src::to_value()
            .factorize()
            .divide(Dst::to_value().factorize())
            .contiguous_tail()
            .unwrap()
    }

    #[test]
    fn unittest_extents_reachable_end_with_dst_padding_absorb() {
        axes![A = 8, B = 3];
        // matched B (directcast, [1,3)) + divisor padding [3,8) extend the tail.
        // matched A is non-directcast (divisor_stride=8 ≠ dividend_stride=3),
        // so the walk stops at 8 — A's iteration breaks src-side contiguity.
        assert_eq!(reachable_end::<m![A, B], m![A, B # 8]>(), 8);
    }

    #[test]
    fn unittest_extents_reachable_end_invariant_under_outer_cluster_slice() {
        axes![Cl = 2, Sl = 4, A = 3];
        // The tail check looks only at divisor-side spans, so adding outer
        // cluster/slice axes to the source must produce the same answer.
        assert_eq!(
            reachable_end::<m![A], m![A # 16]>(),
            reachable_end::<m![Cl, Sl, A], m![A # 16]>(),
        );
    }

    #[test]
    fn unittest_extents_reachable_end_single_element_underflows_alignment() {
        axes![A = 1];
        // Single i32 tail = 4 bytes; not aligned to DMA_SRAM_WRITE_WIDTH (= 8).
        let end = reachable_end::<m![A], m![A]>();
        assert_eq!(end, 1);
        assert_eq!(<i32 as Scalar>::size_in_bytes_from_length(end), 4);
        assert!(!<i32 as Scalar>::size_in_bytes_from_length(end).is_multiple_of(DMA_SRAM_WRITE_WIDTH));
    }

    #[test]
    fn unittest_assert_dma_layout_canonical_cluster_slice_passes() {
        // End-to-end wrapper test on a realistic DM-tier shape:
        // outer Cluster/Slice partitioning, inner element data.
        axes![Cl = 2, Sl = 4, A = 8, B = 4];
        assert_dma_layout::<i32, m![Cl, Sl, A, B], m![A, B]>(DMA_SRAM_WRITE_WIDTH);
    }

    #[test]
    fn unittest_assert_dma_layout_dst_padding_absorbed() {
        axes![A = 8, B = 3];
        assert_dma_layout::<i32, m![A, B], m![A, B # 8]>(DMA_SRAM_WRITE_WIDTH);
    }

    #[test]
    fn unittest_assert_dma_layout_min_align_one_is_noop() {
        // DM→HBM / HBM→HBM use min_align = 1, where both the tail-end check
        // and the stride-alignment check trivially pass. This pins that
        // contract so future refactors of either check cannot regress the
        // DRAM-write path.
        axes![A = 1];
        assert_dma_layout::<i32, m![A], m![A]>(1);

        axes![Cl = 2, Sl = 4, B = 3];
        assert_dma_layout::<i32, m![Cl, Sl, B], m![B # 7]>(1);
    }
}
