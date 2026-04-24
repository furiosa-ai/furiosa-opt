//! Tensors placed on memory.

use rand::Rng;
use rand::distr::StandardUniform;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_mapping_macro::primitive;
use furiosa_opt_macro::m;

use crate::context::*;
use crate::scalar::*;
use crate::tensor::*;
use crate::tensor_view::*;
use crate::vector_engine::scalar::VeScalar;

/// Address.
///
/// TODO: check that every address is 64-bit.
pub type Address = u64;

/// Address in the tensor register file.
#[primitive(TrfAddress)]
#[derive(Clone, Debug)]
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
    /// - `Full`: 65,536 bytes (8 bank rows × 128 rows × 2 columns × 32 bytes)
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
pub struct HostTensor<D: Scalar, Element: M> {
    inner: Tensor<D, Element>,
}

impl<D: Scalar, Element: M> From<Tensor<D, Element>> for HostTensor<D, Element> {
    fn from(inner: Tensor<D, Element>) -> Self {
        Self { inner }
    }
}

impl<D: Scalar, Element: M> HostTensor<D, Element> {
    /// Mapping type alias.
    pub type Mapping = Element;

    pub(crate) fn inner_tensor(&self) -> &Tensor<D, Element> {
        &self.inner
    }

    pub(crate) fn data(&self) -> &ndarray::ArrayD<Opt<D>> {
        self.inner.data()
    }

    /// Creates a tensor from a vector.
    pub fn from_buf(data: Vec<Opt<D>>) -> Self {
        Tensor::from_buf(data).into()
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
        let data: Vec<Opt<D>> = view
            .data()
            .chunks_exact(stride)
            .map(|b| Opt::Init(D::from_le_bytes(b)))
            .collect();
        Ok(Tensor::from_buf(data).into())
    }

    /// Creates a tensor filled with zeros.
    pub fn zero() -> Self {
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

    /// Converts to HBM tensor.
    ///
    /// TODO: `address` should be optional.
    #[primitive(HostTensor::to_hbm)]
    pub async fn to_hbm<Chip: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Pcie }>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2> {
        <crate::runtime::CurrentBackend as crate::runtime::Backend>::to_hbm(self, address).await
    }

    /// Consumes self and returns the inner tensor value.
    pub fn into_inner(self) -> Tensor<D, Self::Mapping> {
        self.inner
    }

    /// Returns the tensor data as a flat vector in physical layout order.
    pub fn to_buf(&self) -> Vec<Opt<D>> {
        self.inner.to_buf()
    }
}

/// Tensor stored in HBM memory.
#[primitive(HbmTensor)]
#[derive(Debug)]
pub struct HbmTensor<D: Scalar, Chip: M, Element: M> {
    inner: Tensor<D, Pair<Chip, Element>>,
    address: Address,
}

// Manual impl: inner `Tensor` is not DeviceSend
impl<D: Scalar, Chip: M, Element: M> crate::runtime::DeviceSend for HbmTensor<D, Chip, Element> {}
impl<D: Scalar, Chip: M, Element: M> crate::runtime::DeviceSend for &HbmTensor<D, Chip, Element> {}
impl<D: Scalar, Chip: M, Element: M> crate::runtime::DeviceSend for &mut HbmTensor<D, Chip, Element> {}
impl<D: Scalar, Chip: M, Element: M> crate::runtime::DeviceSend for HbmTensorView<'_, D, Chip, Element> {}
impl<D: Scalar, Chip: M, Element: M> crate::runtime::DeviceSend for HbmTensorViewMut<'_, D, Chip, Element> {}

impl<D: Scalar, Chip: M, Element: M> HbmTensor<D, Chip, Element> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping>, address: Address) -> Self {
        Self { inner, address }
    }

    pub(crate) fn inner_tensor(&self) -> &Tensor<D, Self::Mapping> {
        &self.inner
    }

    /// Returns the HBM address of this tensor.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Size in bytes.
    pub fn size() -> usize {
        <Pair<Chip, Element> as M>::SIZE * std::mem::size_of::<D>()
    }

    pub(crate) fn data(&self) -> &ndarray::ArrayD<Opt<D>> {
        self.inner.data()
    }

    /// Creates an HBM tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    #[primitive(HbmTensor::from_addr)]
    pub unsafe fn from_addr(address: Address) -> Self {
        Self::new(Tensor::uninit(), address)
    }

    /// Converts to host tensor.
    ///
    /// TODO: we should optionally receive the intermediate stream's mapping expression.
    #[primitive(HbmTensor::to_host)]
    pub async fn to_host<Element2: M>(&self, _dma: &mut DmaContext<{ Dma::Pcie }>) -> HostTensor<D, Element2> {
        <crate::runtime::CurrentBackend as crate::runtime::Backend>::to_host(self).await
    }

    /// Converts to HBM tensor.
    #[primitive(HbmTensor::to_hbm)]
    pub fn to_hbm<const DMA: Dma, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ DMA }>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2> {
        HbmTensor::new(self.inner.transpose(true), address)
    }

    /// Gather DRAM rows into SRAM at positions given by index tensor.
    ///
    /// Implements `index_select` along dim 0: `output[i] = self[index[i]]`.
    /// Inverse of [`DmTensor::dma_scatter`].
    ///
    /// TODO: implement CPU reference and LIR translation (same pattern as dma_scatter).
    #[primitive(HbmTensor::dma_gather)]
    pub fn dma_gather<Cluster2: M, Slice2: M, Element2: M, Element3: M>(
        &self,
        _index_tensor: &HbmTensor<i32, Chip, Element3>,
        _address: Address,
    ) -> DmTensor<D, Chip, Cluster2, Slice2, Element2> {
        todo!()
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensor::view)]
    pub fn view<'l>(&'l self) -> HbmTensorView<'l, D, Chip, Element> {
        HbmTensorView {
            inner: self.inner.view(),
            address: self.address,
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> HbmTensorViewMut<'l, D, Chip, Element> {
        HbmTensorViewMut {
            inner: self.inner.view_mut(),
            address: self.address,
        }
    }
}

// ANCHOR: dma_impl
impl<D: Scalar, Chip: M, Element: M> HbmTensor<D, Chip, Element> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensor::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2> {
        DmTensor::new(self.inner.transpose(true), address)
    }
}
// ANCHOR_END: dma_impl

/// View of an HBM tensor.
#[primitive(HbmTensorView)]
#[derive(Debug, Clone)]
pub struct HbmTensorView<'l, D: Scalar, Chip: M, Element: M> {
    inner: TensorView<'l, D, Pair<Chip, Element>>,
    address: Address,
}

impl<'l, D: Scalar, Chip: M, Element: M> HbmTensorView<'l, D, Chip, Element> {
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
        mut dst: HbmTensorViewMut<'l, D, Chip, Element2>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Converts to data memory tensor.
    #[primitive(HbmTensorView::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2> {
        DmTensor::new(self.inner.read().transpose(true), address)
    }

    /// Writes to data memory tensor view.
    pub fn to_dm_view<Cluster: M, Slice: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression.
    #[primitive(HbmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(&self, start: usize) -> HbmTensorView<'l, D, Chip, Element2> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        HbmTensorView {
            inner,
            address: self.address,
        }
    }
}

/// Mutable view of an HBM tensor.
#[primitive(HbmTensorViewMut)]
#[derive(Debug)]
pub struct HbmTensorViewMut<'l, D: Scalar, Chip: M, Element: M> {
    inner: TensorViewMut<'l, D, Pair<Chip, Element>>,
    address: Address,
}

impl<'l, D: Scalar, Chip: M, Element: M> HbmTensorViewMut<'l, D, Chip, Element> {
    /// Returns the base HBM address of this view.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &mut self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip, Element2> {
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
pub struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>>,
    address: Address,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> DmTensor<D, Chip, Cluster, Slice, Element> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping>, address: Address) -> Self {
        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }

    /// Creates a DM tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    #[primitive(DmTensor::from_addr)]
    pub unsafe fn from_addr(address: Address) -> Self {
        Self::new(Tensor::uninit(), address)
    }

    /// Converts to HBM tensor.
    #[primitive(DmTensor::to_hbm)]
    pub fn to_hbm<Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2> {
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
        index: &HbmTensor<i32, Chip, Element3>,
        output: &mut HbmTensor<D, Chip, Element2>,
        scaled: bool,
    ) {
        let src = Pair::<Slice, Element>::to_value().factorize();
        let key = Key::to_value().factorize();
        assert!(
            src.clone().divide_relaxed(key).exact().is_ok(),
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
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2> {
        DmTensor::new(self.inner.transpose(true), address)
    }

    /// Copies data to another DM tensor via parallel copy.
    ///
    /// Convenience wrapper: `self.view().to_dm_view_pcopy(sub, dst.view_mut())`.
    pub fn to_dm_pcopy<Slice2: M, Element2: M>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
        dst: &mut DmTensor<D, Chip, Cluster, Slice2, Element2>,
    ) {
        self.view().to_dm_view_pcopy(sub, dst.view_mut());
    }

    /// Creates immutable views by splitting along a tile expression.
    #[primitive(DmTensor::view)]
    pub fn view<'l>(&'l self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element> {
        DmTensorView {
            inner: self.inner.view(),
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(DmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element> {
        DmTensorViewMut {
            inner: self.inner.view_mut(),
        }
    }

    /// Reshapes the tensor to a different mapping at the same address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new mapping accurately describes
    /// the data currently at this address.
    #[primitive(DmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2> {
        DmTensor::new(
            unsafe {
                self.inner
                    .clone()
                    .reshape::<Chip, Chip2, m![{ Chip2 }, { Cluster }, { Slice }, { Element }]>()
                    .reshape::<Cluster, Cluster2, m![{ Chip2 }, { Cluster2 }, { Slice }, { Element }]>()
                    .reshape::<Slice, Slice2, m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element }]>()
                    .reshape::<Element, Element2, m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
            },
            self.address,
        )
    }
}

/// Mutable view of a data memory tensor.
#[primitive(DmTensorViewMut)]
#[derive(Debug)]
pub struct DmTensorViewMut<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    pub(crate) inner: TensorViewMut<'l, D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>>,
}

/// View of a data memory tensor.
#[primitive(DmTensorView)]
#[derive(Debug, Clone)]
pub struct DmTensorView<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    pub(crate) inner: TensorView<'l, D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>>,
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>
    From<DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element>> for DmTensorView<'l, D, Chip, Cluster, Slice, Element>
{
    fn from(view: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element>) -> Self {
        Self {
            inner: view.inner.into(),
        }
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> DmTensorView<'l, D, Chip, Cluster, Slice, Element> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// Writes data to a mutable tensor view for HBM.
    #[primitive(DmTensorView::to_hbm_view)]
    pub fn to_hbm_view<Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: HbmTensorViewMut<'l, D, Chip, Element2>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Writes data to a mutable tensor view for data memory.
    #[primitive(DmTensorView::to_dm_view)]
    pub fn to_dm_view<Slice2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }>,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice2, Element2>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Writes data to a mutable tensor view for data memory.
    pub fn to_dm_view_pcopy<Slice2: M, Element2: M>(
        self,
        _sub: &mut TuContext<{ Tu::Sub }>,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice2, Element2>,
    ) {
        dst.inner.write_transpose(self.inner, true);
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip2, Cluster, Slice, Element> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Cluster.
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster2, Slice, Element> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorView::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice2, Element> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorView { inner }
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element> {
    /// Creates mutable views by splitting along a tile expression over Chip.
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &mut self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Cluster.
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &mut self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster2, Slice, Element> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &mut self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2> {
        let inner = self.inner.tile::<Index, _, LEN>(start);
        DmTensorViewMut { inner }
    }
}

/// Tensor stored in the tensor register file.
#[primitive(TrfTensor)]
#[derive(Debug)]
pub struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Element: M> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Row, Element>>>>>,
    #[expect(dead_code)]
    address: TrfAddress,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Row, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Element: M> TrfTensor<D, Chip, Cluster, Slice, Row, Element> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Row }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping>, address: TrfAddress) -> Self {
        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }

    /// Creates a TRF tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    pub unsafe fn from_addr(address: TrfAddress) -> Self {
        Self::new(Tensor::uninit(), address)
    }

    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping> {
        self.inner.view()
    }
}

/// Tensor stored in the vector register file (VRF).
#[primitive(VrfTensor)]
#[derive(Debug, Clone)]
pub struct VrfTensor<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>>,
    #[expect(dead_code)]
    address: Address,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> VrfTensor<D, Chip, Cluster, Slice, Element> {
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping>, address: Address) -> Self {
        Self {
            inner,
            address,
            _marker: PhantomData,
        }
    }

    /// Creates a VRF tensor handle at the given raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible
    /// with the tensor mapping.
    pub unsafe fn from_addr(address: Address) -> Self {
        Self::new(Tensor::uninit(), address)
    }

    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping> {
        self.inner.view()
    }
}

/// Tensor stored in dot product engine
#[derive(Debug)]
pub struct DpeTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Row: M, Packet: M> {
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Pair<Row, Packet>>>>>>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Row: M, Packet: M>
    DpeTensor<D, Chip, Cluster, Slice, Time, Row, Packet>
{
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Row }, { Packet }];

    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping> {
        self.inner.view()
    }
}
