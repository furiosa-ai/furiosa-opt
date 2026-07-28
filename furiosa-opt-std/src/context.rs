use std::fmt::Debug;
use std::marker::ConstParamTy;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::MutexGuard;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::prelude::DmTensor;

use super::scalar::{MaterializableScalar, Scalar};
use super::tensor::Tensor;
use super::tensor::memory::DmTensorView;
use super::tensor::tu::BeginTensor;

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

impl<const T: Tu> crate::runtime::DeviceSend for &mut TuContext<T> {}

/// Context for a specific DMA engine.
#[primitive(DmaContext)]
#[derive(Debug)]
pub struct DmaContext<const DMA: Dma> {
    _marker: PhantomData<()>,
}

impl<const DMA: Dma> crate::runtime::DeviceSend for &mut DmaContext<DMA> {}

/// Logical device a kernel runs on, declared by `#[device(chip, pe)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    /// Number of chips.
    pub chip: u8,
    /// Number of PEs per chip.
    pub pe: u8,
}

/// Device context.
#[primitive(Context)]
#[derive(Debug)]
pub struct Context {
    /// Tensor unit for the main context.
    pub main: TuContext<{ Tu::Main }>,
    /// Tensor unit for the sub context.
    pub sub: TuContext<{ Tu::Sub }>,
    /// Tensor DMA context.
    pub tdma: DmaContext<{ Dma::Tensor }>,
    /// PCIe DMA context.
    pub pdma: DmaContext<{ Dma::Pcie }>,
}

impl crate::runtime::DeviceSend for &mut Context {}

impl Context {
    /// Acquire the tensor units. Chain [`Acquired::bind`] to fix the NPU device before any host
    /// I/O (`to_hbm`/`from_hbm`) initializes the runtime.
    pub fn acquire() -> Acquired {
        static SINGLETON: LazyLock<Mutex<Context>> = LazyLock::new(|| {
            Mutex::new(Context {
                main: TuContext::<{ Tu::Main }> { _marker: PhantomData },
                sub: TuContext::<{ Tu::Sub }> { _marker: PhantomData },
                tdma: DmaContext::<{ Dma::Tensor }> { _marker: PhantomData },
                pdma: DmaContext::<{ Dma::Pcie }> { _marker: PhantomData },
            })
        });
        Acquired(SINGLETON.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
    }
}

/// Process-wide [`Context`] handle returned by [`Context::acquire`].
#[derive(Debug)]
pub struct Acquired(MutexGuard<'static, Context>);

impl Deref for Acquired {
    type Target = Context;
    fn deref(&self) -> &Context {
        &self.0
    }
}

impl DerefMut for Acquired {
    fn deref_mut(&mut self) -> &mut Context {
        &mut self.0
    }
}

impl Acquired {
    /// Binds the process to a kernel's logical [`Device`] (`kernel.device()`) before any host I/O
    /// initializes the runtime. First bind wins; a conflicting one panics. Non-NPU backends ignore
    /// the device.
    pub fn bind(self, device: Device) -> Self {
        crate::runtime::CurrentBackend::bind_device(device);
        self
    }
}

impl<const T: Tu> TuContext<{ T }> {
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
                .tile::<Symbol<I>, m![{ Chip }, { Cluster }, { Slice }, 1 #{!} 2, { Element }], 1>(i)
                .transpose(input.inner, false);
        }

        BeginTensor::new(self, output)
    }
}

impl TuContext<{ Tu::Sub }> {
    /// Asymmetric cluster slice via ParallelCopy (stos): each cluster selects its own slice
    /// position from `slice_indices` (one per cluster) — e.g. `[1, 0]` gives cluster 0 slice 1,
    /// cluster 1 slice 0.
    ///
    /// `AxisToSlice` must be the outermost axis in `Element`.
    #[primitive(TuContext::parallel_copy_cluster_slice)]
    pub fn parallel_copy_cluster_slice<
        'l,
        const CLUSTER_DIM: usize,
        AxisToSlice: M,
        AxisSlicedElement: M,
        Element2: M,
        D: Scalar,
        Chip: M,
        Cluster: M,
        Slice: M,
        Element: M,
    >(
        &mut self,
        tensor: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
        slice_indices: &[usize; CLUSTER_DIM],
    ) -> super::tensor::memory::DmTensor<D, Chip, Cluster, Slice, Element2> {
        let mut sliced = DmTensor::new();

        for (cluster_idx, slice_idx) in slice_indices.iter().enumerate() {
            let cluster_slice = tensor.cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(cluster_idx);
            cluster_slice
                .tile::<AxisToSlice, 1, AxisSlicedElement>(*slice_idx)
                .to_dm_view_pcopy(
                    self,
                    sliced
                        .view_mut()
                        .cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM, { PaddingKind::Bottom }>>(
                            cluster_idx,
                        ),
                );
        }

        sliced
    }

    /// Asymmetric chip slice via ParallelCopy (stos): each chip selects its own slice position
    /// from `slice_indices` (one per chip) — e.g. `[3, 0, 1, 2]` gives chip 0 slice 3, chip 1
    /// slice 0, chip 2 slice 1, chip 3 slice 2.
    #[primitive(TuContext::parallel_copy_chip_slice)]
    pub fn parallel_copy_chip_slice<
        'l,
        const CHIP_DIM: usize,
        AxisToSlice: M,
        AxisSlicedElement: M,
        Element2: M,
        D: Scalar,
        Chip: M,
        Cluster: M,
        Slice: M,
        Element: M,
    >(
        &mut self,
        tensor: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
        slice_indices: &[usize; CHIP_DIM],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2> {
        let mut sliced = DmTensor::new();

        for (chip_idx, slice_idx) in slice_indices.iter().enumerate() {
            let chip_slice = tensor.chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(chip_idx);
            chip_slice
                .tile::<AxisToSlice, 1, AxisSlicedElement>(*slice_idx)
                .to_dm_view_pcopy(
                    self,
                    sliced
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM, { PaddingKind::Bottom }>>(chip_idx),
                );
        }

        sliced
    }
}
