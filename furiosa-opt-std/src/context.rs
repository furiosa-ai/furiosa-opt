use std::fmt::Debug;
use std::marker::ConstParamTy;
use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use super::scalar::{MaterializableScalar, Scalar};
use super::tensor::Tensor;
use super::tensor::memory::DmTensorView;
use super::tensor::tu::BeginTensor;
use crate::Error;
use crate::backend::Backend;
use crate::runtime::{Buffers, DeviceSend};

/// Tensor units.
#[derive(Debug, PartialEq, Eq, ConstParamTy)]
pub enum Tu {
    /// Main context.
    Main,
    /// Sub context.
    Sub,
}

/// DMA units.
#[derive(Debug, PartialEq, Eq, ConstParamTy)]
pub enum Dma {
    /// Tensor DMA.
    Tensor,
    /// PCIe DMA.
    Pcie,
}

/// Context for a specific tensor unit.
#[primitive(TuContext)]
#[derive(Debug)]
pub struct TuContext<const T: Tu> {
    _marker: PhantomData<()>,
}

impl<const T: Tu> DeviceSend for TuContext<T> {
    fn bind(&self, _: &mut Buffers) -> Result<(), Error> {
        Ok(())
    }
}

/// Context for a DMA engine.
#[primitive(DmaContext)]
pub struct DmaContext<const DMA: Dma, B: Backend = crate::runtime::CurrentBackend> {
    device: B::Device,
    _marker: PhantomData<()>,
}

impl<const DMA: Dma, B: Backend> Debug for DmaContext<DMA, B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DmaContext").finish_non_exhaustive()
    }
}

impl<const DMA: Dma, B: Backend> DmaContext<DMA, B> {
    pub(crate) fn on(device: B::Device) -> Self {
        Self {
            device,
            _marker: PhantomData,
        }
    }

    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<const DMA: Dma, B: Backend> DeviceSend for DmaContext<DMA, B> {
    fn bind(&self, _: &mut Buffers) -> Result<(), Error> {
        Ok(())
    }
}

impl<const T: Tu> TuContext<{ T }> {
    pub(crate) const fn on() -> Self {
        Self { _marker: PhantomData }
    }

    /// Begin a tensor unit operation in this context.
    #[primitive(TuContext::begin)]
    pub fn begin<'l, D: MaterializableScalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        &'l mut self,
        tensor: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
    ) -> BeginTensor<'l, { T }, D, Chip, Cluster, Slice, Identity, Element> {
        // The mappings differ only by `Identity`; `transmute` is a safe rewrap after the storage
        // mapping erasure.
        BeginTensor::new(self, tensor.inner.read().transmute())
    }

    /// Begin a tensor unit operation in this context with interleaved tensors.
    #[primitive(TuContext::begin_interleaved)]
    pub fn begin_interleaved<'l, I: AxisName, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        &'l mut self,
        lhs: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
        rhs: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
    ) -> BeginTensor<'l, { T }, D, Chip, Cluster, Slice, Symbol<I>, Element> {
        let mut output = Tensor::<D, m![{ Chip }, { Cluster }, { Slice }, { Symbol<I> }, { Element }]>::zeroed();

        for (i, input) in [lhs, rhs].into_iter().enumerate() {
            output
                .view_mut()
                .tile_derived::<Symbol<I>, 1>(i)
                .transpose(input.inner, false);
        }

        BeginTensor::new(self, output)
    }
}
