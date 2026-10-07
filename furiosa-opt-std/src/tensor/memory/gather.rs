//! Builds deferred HBM gathers by declaring the payload, index domain, and output placement.

use super::*;
use crate::backend::indirect::IndexUnit;
use furiosa_opt_lower::{
    DmPlacement, GatherPayload, IndexMapping, SpmPlacement, validate_gather_output, validate_gather_payload,
    validate_index_chip, validate_spm_indirect_index,
};
use std::num::NonZeroUsize;

/// A deferred gather whose `Stage` determines the available index and prefix.
#[primitive(GatherPlan)]
#[derive(Debug)]
pub struct GatherPlan<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, Stage, B: Backend = CurrentBackend>
where
    Stage: GatherStage<B>,
{
    table: &'l HbmTensor<D, Chip, TableAxes, B>,
    /// The checked payload and indexed-axis stride.
    payload: GatherPayload,
    index: Stage::IndexTensor,
    prefix: Stage::Prefix,
    _marker: PhantomData<IndexedAxis>,
}

/// Values available at each gather stage.
#[doc(hidden)]
pub trait GatherStage<B: Backend>: sealed::Sealed {
    /// The index tensor this stage reads, or `()` before one is supplied.
    type IndexTensor;
    /// The runtime prefix that shortens it, or `()` when every position runs.
    type Prefix;
}

/// A stage with an index, ready for an output placement.
#[doc(hidden)]
pub trait IndexedStage<B: Backend>: GatherStage<B> {}

mod sealed {
    /// Seals the gather stages to this crate.
    pub trait Sealed {}
}

/// The payload is settled and no index has been supplied.
#[primitive(Begin)]
#[doc(hidden)]
#[derive(Debug)]
pub struct Begin;

/// Byte offsets held in HBM have been supplied; the index's own mapping is the domain.
#[primitive(IndexedByteOffsets)]
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexedByteOffsets<Chip: M, Domain: M>(PhantomData<(Chip, Domain)>);

/// SPM positions over `Domain`, the named cluster axes followed by the index list.
#[primitive(IndexedPositions)]
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexedPositions<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M>(
    PhantomData<(Chip, IdxCluster, IdxPe, IdxElement, Domain)>,
);

/// Runs a runtime-valid prefix of the index positions.
#[primitive(Prefixed)]
#[doc(hidden)]
#[derive(Debug)]
pub struct Prefixed<Stage>(PhantomData<Stage>);

impl sealed::Sealed for Begin {}
impl<Chip: M, Domain: M> sealed::Sealed for IndexedByteOffsets<Chip, Domain> {}
impl<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M> sealed::Sealed
    for IndexedPositions<Chip, IdxCluster, IdxPe, IdxElement, Domain>
{
}
impl<Stage> sealed::Sealed for Prefixed<Stage> {}

impl<B: Backend> GatherStage<B> for Begin {
    type IndexTensor = ();
    type Prefix = ();
}

impl<Chip: M, Domain: M, B: Backend> GatherStage<B> for IndexedByteOffsets<Chip, Domain> {
    type IndexTensor = GatherIndexTensor<Pair<Chip, Domain>, B>;
    type Prefix = ();
}

impl<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M, B: Backend> GatherStage<B>
    for IndexedPositions<Chip, IdxCluster, IdxPe, IdxElement, Domain>
{
    type IndexTensor = GatherIndexTensor<Pair<Chip, Pair<IdxCluster, Pair<IdxPe, IdxElement>>>, B>;
    type Prefix = ();
}

impl<Stage: GatherStage<B>, B: Backend> GatherStage<B> for Prefixed<Stage> {
    type IndexTensor = Stage::IndexTensor;
    type Prefix = RuntimePrefix;
}

impl<Chip: M, Domain: M, B: Backend> IndexedStage<B> for IndexedByteOffsets<Chip, Domain> {}
impl<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M, B: Backend> IndexedStage<B>
    for IndexedPositions<Chip, IdxCluster, IdxPe, IdxElement, Domain>
{
}

/// A gather index with its entry domain and value unit.
#[derive(Debug)]
pub struct GatherIndexTensor<IdxAxes: M, B: Backend> {
    tensor: Tensor<i32, IdxAxes, B>,
    domain: Mapping,
    unit: IndexUnit,
}

/// The runtime length requested for a sparse gather.
#[derive(Debug)]
pub struct RuntimePrefix {
    valid_length: i32,
}

impl<D: Scalar, Chip: M, TableAxes: M, B: Backend> HbmTensor<D, Chip, TableAxes, B> {
    /// Starts a gather that replaces `IndexedAxis`, reading `Payload` per index entry.
    /// `Payload` includes any padding left after removing the indexed axis.
    #[primitive(HbmTensor::gather)]
    pub fn gather<IndexedAxis: M, Payload: M>(&self) -> GatherPlan<'_, D, Chip, TableAxes, IndexedAxis, Begin, B> {
        let payload = validate_gather_payload(&TableAxes::to_value(), &IndexedAxis::to_value(), &Payload::to_value())
            .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: self,
            payload,
            index: (),
            prefix: (),
            _marker: PhantomData,
        }
    }
}

/// Adds a byte-offset index to a gather plan.
#[doc(hidden)]
pub trait GatherByteOffsets<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, B: Backend> {
    /// Supplies byte offsets into the table, with `Domain` inferred from the index mapping.
    ///
    /// The index chip axis must match the table's chip axis or broadcast across it.
    #[furiosa_opt::primitive = "GatherPlan::by_byte_offsets"]
    fn by_byte_offsets<Domain: M>(
        self,
        index: &'l HbmTensor<i32, IdxChip, Domain, B>,
    ) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B>;
}

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, B: Backend>
    GatherByteOffsets<'l, D, Chip, TableAxes, IndexedAxis, IdxChip, B>
    for GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Begin, B>
{
    #[furiosa_opt::primitive = "GatherPlan::by_byte_offsets"]
    fn by_byte_offsets<Domain: M>(
        self,
        index: &'l HbmTensor<i32, IdxChip, Domain, B>,
    ) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B> {
        validate_index_chip(&IdxChip::to_value(), &Chip::to_value()).unwrap_or_else(|error| panic!("{error}"));
        let unit = IndexUnit::Bytes(indexed_axis_stride::<D>(&self.payload));
        let domain = IndexMapping::ByteOffsets(&Domain::to_value())
            .domain()
            .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: self.table,
            payload: self.payload,
            index: GatherIndexTensor {
                tensor: index.inner.clone(),
                domain,
                unit,
            },
            prefix: (),
            _marker: PhantomData,
        }
    }
}

/// Adds SPM positions to a gather, inferring placement from the index tensor.
#[doc(hidden)]
pub trait GatherPositions<
    'l,
    D: Scalar,
    Chip: M,
    TableAxes: M,
    IndexedAxis: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    B: Backend,
>
{
    /// Supplies unscaled positions over `Domain`: the named cluster axes followed by the list.
    /// Broadcast or padded single-copy clusters share one list and do not join `Domain`.
    #[furiosa_opt::primitive = "GatherPlan::by_positions"]
    fn by_positions<Domain: M>(
        self,
        index: &'l SpmTensor<i32, IdxChip, IdxCluster, IdxPe, IdxElement, B>,
    ) -> GatherPlan<
        'l,
        D,
        Chip,
        TableAxes,
        IndexedAxis,
        IndexedPositions<IdxChip, IdxCluster, IdxPe, IdxElement, Domain>,
        B,
    >;
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    TableAxes: M,
    IndexedAxis: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    B: Backend,
> GatherPositions<'l, D, Chip, TableAxes, IndexedAxis, IdxChip, IdxCluster, IdxPe, IdxElement, B>
    for GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Begin, B>
{
    #[furiosa_opt::primitive = "GatherPlan::by_positions"]
    fn by_positions<Domain: M>(
        self,
        index: &'l SpmTensor<i32, IdxChip, IdxCluster, IdxPe, IdxElement, B>,
    ) -> GatherPlan<
        'l,
        D,
        Chip,
        TableAxes,
        IndexedAxis,
        IndexedPositions<IdxChip, IdxCluster, IdxPe, IdxElement, Domain>,
        B,
    > {
        validate_index_chip(&IdxChip::to_value(), &Chip::to_value()).unwrap_or_else(|error| panic!("{error}"));
        validate_spm_indirect_index(&IdxPe::to_value()).unwrap_or_else(|error| panic!("{error}"));
        let (cluster, pe, element) = (IdxCluster::to_value(), IdxPe::to_value(), IdxElement::to_value());
        let index_mapping = IndexMapping::SpmPositions(SpmPlacement {
            cluster: &cluster,
            pe: &pe,
            element: &element,
        });
        let domain = index_mapping.domain().unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            domain,
            Domain::to_value().normalize(),
            "gather domain must be the one the index carries"
        );
        GatherPlan {
            table: self.table,
            payload: self.payload,
            index: GatherIndexTensor {
                tensor: index.inner.clone(),
                domain,
                unit: IndexUnit::Positions,
            },
            prefix: (),
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxAxes: M, Stage, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Stage, B>
where
    Stage: IndexedStage<B, Prefix = (), IndexTensor = GatherIndexTensor<IdxAxes, B>>,
{
    fn materialize<Cluster: M, Slice: M, Element: M>(
        self,
        prefix: Option<usize>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        validate_placement::<Cluster, Slice, Element>(&self.payload, &self.index.domain);
        let mut output = DmTensor::from_parts(Tensor::zeroed());
        self.table.inner.gather::<IndexedAxis, _, _>(
            &mut output.inner,
            &self.index.tensor,
            &self.index.domain,
            &Chip::to_value(),
            prefix,
            self.index.unit,
        );
        output
    }
}

// Separate inherent impls avoid overlap with prefixed plans; primitive attributes cannot be shared
// through `macro_rules!`.
impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, Domain: M, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B>
{
    /// Materializes the gather in the requested data-memory shape.
    #[primitive(GatherPlan::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        self.materialize(None)
    }

    /// Limits each chip's gathered entries to `valid_length`, in `Domain` order; zero or less writes none.
    /// All indices must remain valid: DMA may execute past the prefix, and the CPU validates the whole list.
    #[primitive(GatherPlan::sparse_prefix)]
    pub fn sparse_prefix(
        self,
        valid_length: i32,
    ) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Prefixed<IndexedByteOffsets<IdxChip, Domain>>, B> {
        GatherPlan {
            table: self.table,
            payload: self.payload,
            index: self.index,
            prefix: RuntimePrefix { valid_length },
            _marker: PhantomData,
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    TableAxes: M,
    IndexedAxis: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    Domain: M,
    B: Backend,
> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedPositions<IdxChip, IdxCluster, IdxPe, IdxElement, Domain>, B>
{
    /// Materializes the gather in the requested data-memory shape.
    #[primitive(GatherPlan::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        self.materialize(None)
    }
}

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxAxes: M, Stage, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Prefixed<Stage>, B>
where
    Stage: IndexedStage<B, Prefix = (), IndexTensor = GatherIndexTensor<IdxAxes, B>>,
{
    /// Materializes the gather in the requested data-memory shape.
    #[primitive(GatherPlan::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        let RuntimePrefix { valid_length } = self.prefix;
        let requested = usize::try_from(valid_length).unwrap_or(0);
        let plan = GatherPlan::<'l, D, Chip, TableAxes, IndexedAxis, Stage, B> {
            table: self.table,
            payload: self.payload,
            index: self.index,
            prefix: (),
            _marker: PhantomData,
        };
        plan.materialize(Some(requested))
    }
}

fn indexed_axis_stride<D: Scalar>(payload: &GatherPayload) -> NonZeroUsize {
    let stride = payload
        .indexed_axis_stride_bytes(D::BITS)
        .unwrap_or_else(|error| panic!("{error}"));
    NonZeroUsize::new(stride).expect("an indexed-axis stride spans at least one byte")
}

fn validate_placement<Cluster: M, Slice: M, Element: M>(payload: &GatherPayload, domain: &Mapping) {
    let (cluster, slice, element) = (Cluster::to_value(), Slice::to_value(), Element::to_value());
    validate_gather_output(
        &payload.payload,
        domain,
        DmPlacement {
            cluster: &cluster,
            slice: &slice,
            element: &element,
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
}
