use std::fmt::Debug;
use std::marker::ConstParamTy;
use std::marker::PhantomData;
use std::ops::DerefMut;
use std::sync::LazyLock;
use std::sync::Mutex;

use furiosa_mapping::*;
use furiosa_mapping_macro::primitive;
use furiosa_opt_macro::m;

use crate::prelude::DmTensor;
use crate::prelude::HbmTensor;

use super::memory_tensor::DmTensorView;
use super::scalar::Scalar;
use super::stream_tensor::BeginTensor;
use super::tensor::Tensor;

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
    /// Acquire the tensor units.
    pub fn acquire() -> impl DerefMut<Target = Context> {
        static SINGLETON: LazyLock<Mutex<Context>> = LazyLock::new(|| {
            Mutex::new(Context {
                main: TuContext::<{ Tu::Main }> { _marker: PhantomData },
                sub: TuContext::<{ Tu::Sub }> { _marker: PhantomData },
                tdma: DmaContext::<{ Dma::Tensor }> { _marker: PhantomData },
                pdma: DmaContext::<{ Dma::Pcie }> { _marker: PhantomData },
            })
        });
        SINGLETON.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl<const DMA: Dma> DmaContext<DMA> {
    /// Perform cluster shuffle operation using DMA commands (HBM <-> HBM transfer).
    /// This operation redistributes data across clusters according to the shuffle pattern.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor with cluster dimension
    /// * `shuffle_pattern` - Array mapping source cluster to destination cluster
    ///
    /// # Example
    /// For Cluster=2 with shuffle_pattern=\[1,0\]:
    /// - Data from Cluster 1 goes to Cluster 0
    /// - Data from Cluster 0 goes to Cluster 1
    pub fn hbm_cluster_shuffle<D: Scalar, Chip: M, Element: M>(
        &mut self,
        _tensor: &HbmTensor<D, Chip, Element>,
        _shuffle_pattern: &[usize],
    ) -> HbmTensor<D, Chip, Element> {
        // Low-level implementation using DMA commands for inter-cluster data transfer
        // This will be lowered to actual hardware DMA commands by the compiler
        todo!("dm_cluster_shuffle: Low-level DMA command implementation for cluster <-> cluster HBM transfer")
    }

    /// Perform chip shuffle using PCIe/Tensor DMA commands (HBM <-> HBM transfer across chips).
    /// This operation redistributes data across chips according to the shuffle pattern.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor with chip dimension
    /// * `shuffle_pattern` - Array mapping source chip to destination chip
    ///
    /// # Example
    /// For Chip=4 with shuffle_pattern=\[1,2,3,0\]:
    /// - Data from Chip 1 goes to Chip 0
    /// - Data from Chip 2 goes to Chip 1
    /// - Data from Chip 3 goes to Chip 2
    /// - Data from Chip 0 goes to Chip 3
    pub fn hbm_chip_shuffle<D: Scalar, Chip: M, Element: M>(
        &mut self,
        _tensor: &HbmTensor<D, Chip, Element>,
        _shuffle_pattern: &[usize],
    ) -> HbmTensor<D, Chip, Element> {
        todo!("dm_chip_shuffle: Low-level DMA command implementation for chip<->chip HBM transfer")
    }
}

impl DmaContext<{ Dma::Tensor }> {
    /// Perform cluster shuffle operation using DMA commands (DM <-> DM transfer).
    /// This operation redistributes data across clusters according to the shuffle pattern.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor with cluster dimension
    /// * `shuffle_pattern` - Array mapping source cluster to destination cluster
    ///
    /// # Example
    /// For Cluster=2 with shuffle_pattern=\[1,0\]:
    /// - Data from Cluster 1 goes to Cluster 0
    /// - Data from Cluster 0 goes to Cluster 1
    #[primitive(DmaContext::dm_cluster_shuffle)]
    pub fn dm_cluster_shuffle<D: Scalar, const CLUSTER_DIM: usize, Chip: M, Cluster: M, Slice: M, Element: M>(
        &mut self,
        tensor: DmTensorView<D, Chip, Cluster, Slice, Element>,
        shuffle_pattern: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element> {
        let mut shuffled: DmTensor<D, Chip, Cluster, Slice, Element> = unsafe { DmTensor::from_addr(0) };

        for (target_cluster_idx, source_cluster_idx) in shuffle_pattern.iter().enumerate() {
            tensor
                .cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(*source_cluster_idx)
                .to_dm_view(
                    self,
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
    /// * `tensor` - Input tensor with chip dimension
    /// * `shuffle_pattern` - Array mapping source chip to destination chip
    ///
    /// # Example
    /// For Chip=4 with shuffle_pattern=\[1,2,3,0\]:
    /// - Data from Chip 1 goes to Chip 0
    /// - Data from Chip 2 goes to Chip 1
    /// - Data from Chip 3 goes to Chip 2
    /// - Data from Chip 0 goes to Chip 3
    #[primitive(DmaContext::dm_chip_shuffle)]
    pub fn dm_chip_shuffle<D: Scalar, const CHIP_DIM: usize, Chip: M, Cluster: M, Slice: M, Element: M>(
        &mut self,
        tensor: DmTensorView<D, Chip, Cluster, Slice, Element>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element> {
        let mut shuffled: DmTensor<D, Chip, Cluster, Slice, Element> = unsafe { DmTensor::from_addr(0) };

        for (target_chip_idx, source_chip_idx) in shuffle_pattern.iter().enumerate() {
            tensor
                .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(*source_chip_idx)
                .to_dm_view(
                    self,
                    shuffled
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(target_chip_idx),
                );
        }

        shuffled
    }
}

impl<const T: Tu> TuContext<{ T }> {
    /// Begin a tensor unit operation in this context.
    #[primitive(TuContext::begin)]
    pub fn begin<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        &'l mut self,
        tensor: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
    ) -> BeginTensor<'l, { T }, D, Chip, Cluster, Slice, Identity, Element> {
        // SAFETY: the mappings differ only by `Identity`.
        BeginTensor::new(self, unsafe { tensor.inner.read().transmute() })
    }

    /// Begin a tensor unit operation in this context with interleaved tensors.
    #[primitive(TuContext::begin_interleaved)]
    pub fn begin_interleaved<'l, I: AxisName, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        &'l mut self,
        lhs: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
        rhs: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
    ) -> BeginTensor<'l, { T }, D, Chip, Cluster, Slice, Symbol<I>, Element> {
        let mut output = Tensor::<D, m![{ Chip }, { Cluster }, { Slice }, { Symbol<I> }, { Element }]>::uninit();

        for (i, input) in [lhs, rhs].into_iter().enumerate() {
            output
                .view_mut()
                .tile::<Symbol<I>, m![{ Chip }, { Cluster }, { Slice }, 1 # 2, { Element }], 1>(i)
                .write_transpose(input.inner, false);
        }

        BeginTensor::new(self, output)
    }
}

impl TuContext<{ Tu::Sub }> {
    /// Perform asymmetric cluster slice operation using ParallelCopy (stos command).
    /// This operation allows different clusters to select different slice positions.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor with cluster dimension
    /// * `slice_indices` - Array of slice indices, one per cluster
    ///
    /// # Example
    /// For Cluster=2 with slice_indices=\[1,0\]:
    /// - Cluster 0 selects slice position 1
    /// - Cluster 1 selects slice position 0
    ///
    /// # Restrictions
    /// The `AxisToSlice` should be the outermost axis in `Element`.
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
    ) -> super::memory_tensor::DmTensor<D, Chip, Cluster, Slice, Element2> {
        let mut sliced = unsafe { DmTensor::from_addr(0) };

        for (cluster_idx, slice_idx) in slice_indices.iter().enumerate() {
            let cluster_slice = tensor.cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(cluster_idx);
            cluster_slice
                .tile::<AxisToSlice, 1, AxisSlicedElement>(*slice_idx)
                .to_dm_view_pcopy(
                    self,
                    sliced
                        .view_mut()
                        .cluster_tile::<Cluster, 1, Padding<Identity, CLUSTER_DIM>>(cluster_idx),
                );
        }

        sliced
    }

    /// Perform asymmetric chip slice operation using ParallelCopy (stos command).
    /// This operation allows different chips to select different slice positions.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor with chip dimension
    /// * `slice_indices` - Array of slice indices, one per chip
    ///
    /// # Example
    /// For Chip=4 with slice_indices=\[3,0,1,2\]:
    /// - Chip 0 selects slice position 3
    /// - Chip 1 selects slice position 0
    /// - Chip 2 selects slice position 1
    /// - Chip 3 selects slice position 2
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
        let mut sliced = unsafe { DmTensor::from_addr(0) };

        for (chip_idx, slice_idx) in slice_indices.iter().enumerate() {
            let chip_slice = tensor.chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(chip_idx);
            chip_slice
                .tile::<AxisToSlice, 1, AxisSlicedElement>(*slice_idx)
                .to_dm_view_pcopy(
                    self,
                    sliced
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, CHIP_DIM>>(chip_idx),
                );
        }

        sliced
    }
}
