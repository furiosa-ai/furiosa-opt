//! Deferred SRAM redistribution.

use super::*;
use furiosa_opt_lower::{
    ClusterPlacement, SliceRequest, SramRedistributeEntry, config_outermost_dm_slice, config_slice,
    validate_chip_shuffle, validate_slice_indices, validate_sram_redistribution,
};

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Swaps the two clusters; the output mapping does not record the selected source clusters.
    #[primitive(DmTensor::cluster_swap)]
    pub fn cluster_swap(
        &self,
    ) -> DmRedistributeView<'_, D, Chip, Cluster, Slice, Element, Element, NoAxesToSlice, ClusterSwap, B> {
        self.view().cluster_swap()
    }

    /// Selects source chips by `shuffle_pattern[target] = source`.
    /// The output mapping does not record the selected source chips.
    #[primitive(DmTensor::chip_shuffle)]
    pub fn chip_shuffle(
        &self,
        shuffle_pattern: impl AsRef<[usize]>,
    ) -> DmRedistributeView<'_, D, Chip, Cluster, Slice, Element, Element, NoAxesToSlice, ChipShuffle, B> {
        self.view().chip_shuffle(shuffle_pattern)
    }

    /// Selects an axis position per chip for a later DMA transfer.
    /// The output mapping removes the axis but does not record the selected positions.
    #[primitive(DmTensor::chip_slice)]
    pub fn chip_slice<AxisToSlice: M, Element2: M>(
        &self,
        indices: impl AsRef<[usize]>,
    ) -> DmRedistributeView<
        '_,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        Element,
        AxesToSliceList<NoAxesToSlice, AxisToSlice>,
        ChipSlice,
        B,
    > {
        self.view().chip_slice::<AxisToSlice, Element2>(indices)
    }

    /// Selects an axis position per cluster for a later DMA transfer.
    /// The output mapping removes the axis but does not record the selected positions.
    #[primitive(DmTensor::cluster_slice)]
    pub fn cluster_slice<AxisToSlice: M, Element2: M>(
        &self,
        indices: impl AsRef<[usize]>,
    ) -> DmRedistributeView<
        '_,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        Element,
        AxesToSliceList<NoAxesToSlice, AxisToSlice>,
        ClusterSlice,
        B,
    > {
        self.view().cluster_slice::<AxisToSlice, Element2>(indices)
    }
}

#[doc(hidden)]
pub trait AxesToSlice: std::fmt::Debug + Clone {
    type Indices: Copy + std::fmt::Debug;

    fn axes_to_slice() -> Vec<Mapping>;

    fn slice_indices(indices: Self::Indices) -> Vec<usize>;
}

/// A deferred SRAM redistribution whose source choices are not encoded by its output mapping.
/// Materialize it with `to_dm` or `to_dm_view`.
#[primitive(DmRedistributeView)]
#[derive(Debug)]
pub struct DmRedistributeView<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage,
    B: Backend = CurrentBackend,
> {
    input: DmTensorView<'l, D, Chip, Cluster, Slice, SourceElement, B>,
    sources: Vec<(ClusterPlacement, Axes::Indices)>,
    _marker: PhantomData<(Element, State)>,
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct NoAxesToSlice;

impl AxesToSlice for NoAxesToSlice {
    type Indices = ();

    fn axes_to_slice() -> Vec<Mapping> {
        Vec::new()
    }

    fn slice_indices((): Self::Indices) -> Vec<usize> {
        Vec::new()
    }
}

#[doc(hidden)]
pub trait RedistributionStage: std::fmt::Debug {}

#[doc(hidden)]
pub trait CanRedistributeTo<Next: RedistributionStage>: RedistributionStage {}

#[doc(hidden)]
#[derive(Debug)]
pub struct Init;
impl RedistributionStage for Init {}

#[doc(hidden)]
#[derive(Debug)]
pub struct ChipShuffle;
impl RedistributionStage for ChipShuffle {}

#[doc(hidden)]
#[derive(Debug)]
pub struct ClusterSwap;
impl RedistributionStage for ClusterSwap {}

#[doc(hidden)]
#[derive(Debug)]
pub struct ChipSlice;
impl RedistributionStage for ChipSlice {}

#[doc(hidden)]
#[derive(Debug)]
pub struct ClusterSlice;
impl RedistributionStage for ClusterSlice {}

impl CanRedistributeTo<ChipShuffle> for Init {}

impl CanRedistributeTo<ClusterSwap> for Init {}
impl CanRedistributeTo<ClusterSwap> for ChipShuffle {}

impl CanRedistributeTo<ChipSlice> for Init {}
impl CanRedistributeTo<ChipSlice> for ChipShuffle {}
impl CanRedistributeTo<ChipSlice> for ClusterSwap {}
impl CanRedistributeTo<ChipSlice> for ChipSlice {}

impl CanRedistributeTo<ClusterSlice> for Init {}
impl CanRedistributeTo<ClusterSlice> for ChipShuffle {}
impl CanRedistributeTo<ClusterSlice> for ClusterSwap {}
impl CanRedistributeTo<ClusterSlice> for ChipSlice {}
impl CanRedistributeTo<ClusterSlice> for ClusterSlice {}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct AxesToSliceList<Axes, AxisToSlice>(PhantomData<(Axes, AxisToSlice)>);

impl<Axes: AxesToSlice, AxisToSlice: M> AxesToSlice for AxesToSliceList<Axes, AxisToSlice> {
    type Indices = (Axes::Indices, usize);

    fn axes_to_slice() -> Vec<Mapping> {
        let mut axes = Axes::axes_to_slice();
        axes.push(AxisToSlice::to_value());
        axes
    }

    fn slice_indices((indices, index): Self::Indices) -> Vec<usize> {
        let mut indices = Axes::slice_indices(indices);
        indices.push(index);
        indices
    }
}

#[derive(Debug, Clone)]
struct AxisSlicedMapping<Axes, Element>(PhantomData<(Axes, Element)>);

impl<Axes: AxesToSlice, Element: M> M for AxisSlicedMapping<Axes, Element> {
    const SIZE: usize = Element::SIZE;

    fn to_value() -> Mapping {
        Axes::axes_to_slice()
            .into_iter()
            .fold(Element::to_value(), |element, axis| {
                furiosa_opt_lower::tile_mapping(furiosa_opt_lower::TileMappingInput {
                    index: axis,
                    element,
                    len: 1,
                    hole_fill: PaddingKind::Top,
                })
                .unwrap_or_else(|error| panic!("{error}"))
            })
    }

    fn map(index: usize) -> Cell {
        Self::to_value().index(index)
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorView<'l, D, Chip, Cluster, Slice, Element, B>
{
    /// Swaps the two clusters; the output mapping does not record the selected source clusters.
    #[primitive(DmTensorView::cluster_swap)]
    pub fn cluster_swap(
        self,
    ) -> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, Element, NoAxesToSlice, ClusterSwap, B> {
        assert_eq!(Cluster::SIZE, 2, "cluster_swap requires exactly two clusters");
        let sources = (0..Chip::SIZE * Cluster::SIZE)
            .map(|target| {
                let placement = ClusterPlacement::from_target_index(target, Cluster::SIZE);
                (
                    ClusterPlacement {
                        cluster: placement.cluster ^ 1,
                        ..placement
                    },
                    (),
                )
            })
            .collect();
        DmRedistributeView {
            input: self,
            sources,
            _marker: PhantomData,
        }
    }

    /// Selects source chips by `shuffle_pattern[target] = source`.
    /// The output mapping does not record the selected source chips.
    #[primitive(DmTensorView::chip_shuffle)]
    pub fn chip_shuffle(
        self,
        shuffle_pattern: impl AsRef<[usize]>,
    ) -> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, Element, NoAxesToSlice, ChipShuffle, B> {
        let shuffle_pattern = shuffle_pattern.as_ref();
        assert_chip_shuffle_pattern(shuffle_pattern, Chip::SIZE);
        let sources = shuffle_pattern
            .iter()
            .copied()
            .flat_map(|chip| (0..Cluster::SIZE).map(move |cluster| (ClusterPlacement { chip, cluster }, ())))
            .collect();
        DmRedistributeView {
            input: self,
            sources,
            _marker: PhantomData,
        }
    }

    /// Selects an axis position per chip for a later DMA transfer.
    /// The output mapping removes the axis but does not record the selected positions.
    #[primitive(DmTensorView::chip_slice)]
    pub fn chip_slice<AxisToSlice: M, Element2: M>(
        self,
        indices: impl AsRef<[usize]>,
    ) -> DmRedistributeView<
        'l,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        Element,
        AxesToSliceList<NoAxesToSlice, AxisToSlice>,
        ChipSlice,
        B,
    > {
        self.initial_slice::<AxisToSlice, Element2, ChipSlice>(indices, SliceIndexing::PerChip)
    }

    /// Selects an axis position per cluster for a later DMA transfer.
    /// The output mapping removes the axis but does not record the selected positions.
    #[primitive(DmTensorView::cluster_slice)]
    pub fn cluster_slice<AxisToSlice: M, Element2: M>(
        self,
        indices: impl AsRef<[usize]>,
    ) -> DmRedistributeView<
        'l,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        Element,
        AxesToSliceList<NoAxesToSlice, AxisToSlice>,
        ClusterSlice,
        B,
    > {
        self.initial_slice::<AxisToSlice, Element2, ClusterSlice>(indices, SliceIndexing::PerCluster)
    }

    fn initial_slice<AxisToSlice: M, Element2: M, State: RedistributionStage>(
        self,
        indices: impl AsRef<[usize]>,
        indexing: SliceIndexing,
    ) -> DmRedistributeView<
        'l,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        Element,
        AxesToSliceList<NoAxesToSlice, AxisToSlice>,
        State,
        B,
    > {
        let indices = indices.as_ref();
        let targets = match indexing {
            SliceIndexing::PerChip => Chip::SIZE,
            SliceIndexing::PerCluster => Cluster::SIZE,
        };
        assert_slice_indices::<AxisToSlice>(indices, targets);
        assert_slice_stage::<D, Element, Element, AxisToSlice, Element2, AxesToSliceList<NoAxesToSlice, AxisToSlice>>();
        let sources = (0..Chip::SIZE * Cluster::SIZE)
            .map(|target| {
                let source = ClusterPlacement::from_target_index(target, Cluster::SIZE);
                let index = match indexing {
                    SliceIndexing::PerChip => indices[source.chip],
                    SliceIndexing::PerCluster => indices[source.cluster],
                };
                (source, ((), index))
            })
            .collect();
        DmRedistributeView {
            input: self,
            sources,
            _marker: PhantomData,
        }
    }

    fn apply_axis_slices<Axes: AxesToSlice>(
        &self,
        indices: Axes::Indices,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, AxisSlicedMapping<Axes, Element>, B> {
        let axes = Axes::axes_to_slice();
        let starts = Axes::slice_indices(indices);
        DmTensorView {
            inner: self
                .inner
                .retile_axes::<m![{ Chip }, { Cluster }, { Slice }, { AxisSlicedMapping<Axes, Element> }]>(
                    &axes, &starts,
                ),
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage,
    B: Backend,
> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, State, B>
{
    /// Reassigns whole source entries according to each target placement.
    fn recompose(
        &self,
        remap: impl Fn(ClusterPlacement) -> ClusterPlacement,
    ) -> Vec<(ClusterPlacement, Axes::Indices)> {
        (0..Chip::SIZE * Cluster::SIZE)
            .map(|target| {
                let target_placement = ClusterPlacement::from_target_index(target, Cluster::SIZE);
                let source_target = remap(target_placement).to_target_index(Cluster::SIZE);
                self.sources[source_target]
            })
            .collect()
    }

    /// Appends a slice index derived from each target's chip or cluster.
    fn append_slice_index(
        &self,
        index_for: impl Fn(ClusterPlacement) -> usize,
    ) -> Vec<(ClusterPlacement, (Axes::Indices, usize))> {
        self.sources
            .iter()
            .copied()
            .enumerate()
            .map(|(target, (source, indices))| {
                let target_placement = ClusterPlacement::from_target_index(target, Cluster::SIZE);
                (source, (indices, index_for(target_placement)))
            })
            .collect()
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage + CanRedistributeTo<ChipShuffle>,
    B: Backend,
> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, State, B>
{
    /// Composes one chip permutation into this deferred redistribution.
    #[primitive(DmRedistributeView::chip_shuffle)]
    pub fn chip_shuffle(
        self,
        shuffle_pattern: impl AsRef<[usize]>,
    ) -> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, ChipShuffle, B> {
        let shuffle_pattern = shuffle_pattern.as_ref();
        assert_chip_shuffle_pattern(shuffle_pattern, Chip::SIZE);
        // The composed target reads the source assigned to this slot by the preceding plan.
        let sources = self.recompose(|target| ClusterPlacement {
            chip: shuffle_pattern[target.chip],
            ..target
        });
        DmRedistributeView {
            input: self.input,
            sources,
            _marker: PhantomData,
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage + CanRedistributeTo<ClusterSwap>,
    B: Backend,
> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, State, B>
{
    /// Composes one two-cluster swap into this deferred redistribution.
    #[primitive(DmRedistributeView::cluster_swap)]
    pub fn cluster_swap(
        self,
    ) -> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, ClusterSwap, B> {
        assert_eq!(Cluster::SIZE, 2, "cluster_swap requires exactly two clusters");
        let sources = self.recompose(|target| ClusterPlacement {
            cluster: target.cluster ^ 1,
            ..target
        });
        DmRedistributeView {
            input: self.input,
            sources,
            _marker: PhantomData,
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage + CanRedistributeTo<ChipSlice>,
    B: Backend,
> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, State, B>
{
    /// Selects one additional source axis position for every target chip.
    /// The output mapping removes the axis but does not record the selected positions.
    #[primitive(DmRedistributeView::chip_slice)]
    pub fn chip_slice<AxisToSlice: M, Element2: M>(
        self,
        indices: impl AsRef<[usize]>,
    ) -> DmRedistributeView<
        'l,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        SourceElement,
        AxesToSliceList<Axes, AxisToSlice>,
        ChipSlice,
        B,
    > {
        let indices = indices.as_ref();
        assert_slice_indices::<AxisToSlice>(indices, Chip::SIZE);
        assert_slice_stage::<D, SourceElement, Element, AxisToSlice, Element2, AxesToSliceList<Axes, AxisToSlice>>();
        let sources = self.append_slice_index(|target| indices[target.chip]);
        DmRedistributeView {
            input: self.input,
            sources,
            _marker: PhantomData,
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage + CanRedistributeTo<ClusterSlice>,
    B: Backend,
> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, State, B>
{
    /// Selects one additional source axis position for every target cluster.
    /// The output mapping removes the axis but does not record the selected positions.
    #[primitive(DmRedistributeView::cluster_slice)]
    pub fn cluster_slice<AxisToSlice: M, Element2: M>(
        self,
        indices: impl AsRef<[usize]>,
    ) -> DmRedistributeView<
        'l,
        D,
        Chip,
        Cluster,
        Slice,
        Element2,
        SourceElement,
        AxesToSliceList<Axes, AxisToSlice>,
        ClusterSlice,
        B,
    > {
        let indices = indices.as_ref();
        assert_slice_indices::<AxisToSlice>(indices, Cluster::SIZE);
        assert_slice_stage::<D, SourceElement, Element, AxisToSlice, Element2, AxesToSliceList<Axes, AxisToSlice>>();
        let sources = self.append_slice_index(|target| indices[target.cluster]);
        DmRedistributeView {
            input: self.input,
            sources,
            _marker: PhantomData,
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    SourceElement: M,
    Axes: AxesToSlice,
    State: RedistributionStage,
    B: Backend,
> DmRedistributeView<'l, D, Chip, Cluster, Slice, Element, SourceElement, Axes, State, B>
{
    /// Materializes this redistribution in data memory.
    #[primitive(DmRedistributeView::to_dm)]
    pub fn to_dm<Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        let mut output = DmTensor::new();
        config_slice(SliceRequest::Layout {
            axes: Vec::new(),
            element: Element::to_value(),
            output: Element2::to_value(),
        })
        .unwrap_or_else(|error| panic!("{error}"));
        assert_sram_redistribute::<Chip, Cluster, Axes>(&self.sources);
        self.materialize_to(dma, output.view_mut());
        output
    }

    /// Materializes this redistribution into an existing data-memory view.
    /// CPU evaluation uses a canonical intermediate; NPU lowering emits one fused DMA.
    #[primitive(DmRedistributeView::to_dm_view)]
    pub fn to_dm_view<'o, Slice2: M, Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        output: DmTensorViewMut<'o, D, Chip, Cluster, Slice2, Element2, B>,
    ) {
        assert_sram_redistribute::<Chip, Cluster, Axes>(&self.sources);
        let mut selected: DmTensor<D, Chip, Cluster, Slice, AxisSlicedMapping<Axes, SourceElement>, B> =
            DmTensor::new();
        self.materialize_to(dma, selected.view_mut());
        selected.view().to_dm_view(dma, output);
    }

    fn materialize_to<'o, Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut output: DmTensorViewMut<'o, D, Chip, Cluster, Slice, Element2, B>,
    ) {
        for (target, (source, slice_indices)) in self.sources.into_iter().enumerate() {
            let target_placement = ClusterPlacement::from_target_index(target, Cluster::SIZE);
            if !matches!(Chip::map(target_placement.chip), Cell::Index(_))
                || !matches!(Cluster::map(target_placement.cluster), Cell::Index(_))
            {
                continue;
            }
            self.input
                .chip_tile_derived::<Chip, 1>(source.chip)
                .cluster_tile_derived::<Cluster, 1>(source.cluster)
                .apply_axis_slices::<Axes>(slice_indices)
                .to_dm_view(
                    dma,
                    output
                        .reborrow()
                        .chip_tile_derived::<Chip, 1>(target_placement.chip)
                        .cluster_tile_derived::<Cluster, 1>(target_placement.cluster),
                );
        }
    }
}

fn assert_slice_stage<D: Scalar, SourceElement: M, Element: M, AxisToSlice: M, Element2: M, Axes: AxesToSlice>() {
    config_slice(SliceRequest::Layout {
        axes: vec![AxisToSlice::to_value()],
        element: Element::to_value(),
        output: Element2::to_value(),
    })
    .unwrap_or_else(|error| panic!("{error}"));
    config_slice(SliceRequest::AlignedStrides {
        axes: Axes::axes_to_slice(),
        element: SourceElement::to_value(),
        output: Element2::to_value(),
        element_bits: D::BITS,
    })
    .unwrap_or_else(|error| panic!("{error}"));
}

fn assert_sram_redistribute<Chip: M, Cluster: M, Axes: AxesToSlice>(sources: &[(ClusterPlacement, Axes::Indices)]) {
    let sources = sources
        .iter()
        .map(|&(placement, index)| SramRedistributeEntry {
            source_chip: placement.chip,
            source_cluster: placement.cluster,
            slice_indices: Axes::slice_indices(index),
        })
        .collect::<Vec<_>>();
    validate_sram_redistribution(
        &sources,
        &Chip::to_value(),
        &Cluster::to_value(),
        &Axes::axes_to_slice(),
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_asymmetric_slice<D: Scalar, AxisToSlice: M, Element: M, Element2: M>() {
    config_outermost_dm_slice(
        AxisToSlice::to_value(),
        Element::to_value(),
        Element2::to_value(),
        D::BITS,
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_slice_indices<Axis: M>(indices: &[usize], num_targets: usize) {
    validate_slice_indices(indices, num_targets, &Axis::to_value()).unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_chip_shuffle_pattern(pattern: &[usize], num_chips: usize) {
    validate_chip_shuffle(pattern, num_chips).unwrap_or_else(|error| panic!("{error}"));
}
