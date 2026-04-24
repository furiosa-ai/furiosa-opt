//! Pipeline stage definitions and transition traits for VE typestate pattern.
//!
//! # Typestate goals
//! - **Context markers** (`Standalone`, `Group`, `Zipped`) track which VE operations
//!   are valid for a tensor view. For example, `Zipped` tensors already represent the
//!   zipped result of a binary op, so they cannot `stash()` or `filter()` again.
//! - **Stage markers** (Branch, Logic, Fxp, …) mirror the hardware pipeline. Transition traits
//!   (`CanTransitionTo`) ensure the Rust type system disallows illegal stage ordering.
//! - **Commit gating**: only contexts that represent a full tensor (`Standalone`, `Zipped`)
//!   implement `Commitable`, preventing accidental commits of intermediate group states.
//!
//! # Pipeline order
//! Branch → Logic → Fxp → FxpToFp → Narrow → Fp → IntraSliceReduce → FpDiv → Widen → FpToFxp → Clip → Output

use std::marker::ConstParamTy;

// ============================================================================
// Tensor context markers
// ============================================================================

/// Marker trait for VE tensor context state.
/// Used to distinguish tensor contexts and restrict available operations.
/// there are three Contexts: Standalone/Group/Zipped.
/// depending on the TensorContext, commit/stash/filter operations may be allowed or disallowed.
pub trait VeTensorContext {}

/// Standalone tensor - represents a single tensor view that can use the full VE API
/// (stash/filter, stage transitions, commit).
#[derive(Debug, Clone, Copy)]
pub struct Standalone;
impl VeTensorContext for Standalone {}

/// In group context - used for individual groups managed by VectorTensorPair.
/// Only per-group operations (fxp, logic, clip, etc.) are allowed. Stash/common
/// operations are disabled so both groups stay in lock-step via the pair API.
#[derive(Debug, Clone, Copy)]
pub struct Group;
impl VeTensorContext for Group {}

/// After group zip - result of merging two groups. Filter/stash are NOT available
/// because VE already reduced to a single group.
#[derive(Debug, Clone, Copy)]
pub struct Zipped;
impl VeTensorContext for Zipped {}

/// Marker trait for contexts that can commit to `DmTensor`.
/// `Group` is intentionally excluded so users must zip the groups first.
pub trait Commitable: VeTensorContext {}

impl Commitable for Standalone {}

impl Commitable for Zipped {}

// ============================================================================
// ============================================================================
// VE entry order (IntraFirst)
// ============================================================================

/// Tracks which VE block was entered first.
///
/// - `IntraFirst`: VAU (intra-slice) was entered first via `vector_intra_slice_branch` or `vector_intra_slice_unzip`.
/// - `InterFirst`: VRU (inter-slice) was entered first via `vector_inter_slice_reduce` on `VectorInitTensor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ConstParamTy)]
pub enum VeOrder {
    /// VAU was entered first.
    IntraFirst,
    /// VRU was entered first.
    InterFirst,
}

// ============================================================================
// Packet mode markers (4-way / 8-way)
// ============================================================================

/// Packet mode — tracks whether the tensor is in 8-way or 4-way mode.
///
/// The hardware VE flit is 8 elements wide, but float ALU only processes the front 4.
/// `Way8` is the default mode (full 8-element flit).
/// `Way4` indicates the packet has been narrowed to 4 elements via `vector_split` or `vector_trim_way4`.
/// Float operations are only available in `Way4` mode, enforced at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ConstParamTy)]
pub enum PacketMode {
    /// 8-way packet mode (default). Full 8-element flit.
    /// Float operations are NOT available in this mode.
    /// Use `vector_split` (>4 real elements) or `vector_trim_way4` (≤4 real elements) to transition to `Way4`.
    Way8,
    /// 4-way packet mode. Front-4-only flit after `vector_split` or `vector_trim_way4`.
    /// Float operations are available in this mode.
    /// Use `vector_concat` or `vector_pad_way8` to transition back to `Way8`.
    Way4,
}

// ============================================================================
// Stage marker types
// ============================================================================

/// Trait for pipeline stage markers.
pub trait Stage {}

/// Trait for VAU (intra-slice) stages: Branch through Clip.
pub trait IntraSliceStage: Stage {}

/// Trait for VRU (inter-slice) stages.
pub trait InterSliceStage: Stage {}

/// Trait for stages that support stash operation.
/// Only specific stages can stash current data (based on hardware ALU availability).
/// Supported: Branch, Logic, Fxp, Narrow, Fp, FpDiv, Clip
/// NOT supported: FxpToFp, IntraSliceReduce, Widen, FpToFxp, Output
pub trait Stashable: Stage {}

/// Branch unit configuration stage.
#[derive(Debug, Clone, Copy)]
pub struct Branch;
impl Stage for Branch {}
impl IntraSliceStage for Branch {}
impl Stashable for Branch {}

/// Logic cluster stage (max 5 ops).
#[derive(Debug, Clone, Copy)]
pub struct Logic;
impl Logic {
    /// Maximum number of operations in Logic cluster.
    pub const MAX_OPS: usize = 5;
}
impl Stage for Logic {}
impl IntraSliceStage for Logic {}
impl Stashable for Logic {}

/// Fixed-point cluster stage (max 4 ops).
#[derive(Debug, Clone, Copy)]
pub struct Fxp;
impl Fxp {
    /// Maximum number of operations in Fxp cluster.
    pub const MAX_OPS: usize = 4;
}
impl Stage for Fxp {}
impl IntraSliceStage for Fxp {}
impl Stashable for Fxp {}

/// FxpToFp conversion stage.
#[derive(Debug, Clone, Copy)]
pub struct FxpToFp;
impl Stage for FxpToFp {}
impl IntraSliceStage for FxpToFp {}

/// Narrow layer stage.
#[derive(Debug, Clone, Copy)]
pub struct Narrow;
impl Stage for Narrow {}
impl IntraSliceStage for Narrow {}
impl Stashable for Narrow {}

/// Fp cluster stage (max 5 ops).
#[derive(Debug, Clone, Copy)]
pub struct Fp;
impl Fp {
    /// Maximum number of operations in Fp cluster.
    pub const MAX_OPS: usize = 5;
}
impl Stage for Fp {}
impl IntraSliceStage for Fp {}
impl Stashable for Fp {}

/// Intra-slice reduce stage.
#[derive(Debug, Clone, Copy)]
pub struct IntraSliceReduce;
impl Stage for IntraSliceReduce {}
impl IntraSliceStage for IntraSliceReduce {}

/// FpDiv stage.
#[derive(Debug, Clone, Copy)]
pub struct FpDiv;
impl Stage for FpDiv {}
impl IntraSliceStage for FpDiv {}
impl Stashable for FpDiv {}

/// Widen layer stage.
#[derive(Debug, Clone, Copy)]
pub struct Widen;
impl Stage for Widen {}
impl IntraSliceStage for Widen {}

/// FpToFxp conversion stage.
#[derive(Debug, Clone, Copy)]
pub struct FpToFxp;
impl Stage for FpToFxp {}
impl IntraSliceStage for FpToFxp {}

/// Clip cluster stage (max 3 ops).
#[derive(Debug, Clone, Copy)]
pub struct Clip;
impl Clip {
    /// Maximum number of operations in Clip cluster.
    pub const MAX_OPS: usize = 3;
}
impl Stage for Clip {}
impl IntraSliceStage for Clip {}
impl Stashable for Clip {}

/// Inter-slice reduce stage — after inter-slice reduce operation.
#[derive(Debug, Clone, Copy)]
pub struct InterSliceReduce;
impl Stage for InterSliceReduce {}
impl InterSliceStage for InterSliceReduce {}

/// Filter stage (applies branch filtering before output).
#[derive(Debug, Clone, Copy)]
pub struct Filter;
impl Stage for Filter {}

/// Output configuration stage (filter, write_branch).
#[derive(Debug, Clone, Copy)]
pub struct Output;
impl Stage for Output {}

// ============================================================================
// Stage transition traits
// ============================================================================

/// Generic marker trait for stage transitions.
/// `CanTransitionTo<Target>` indicates that a stage can transition to `Target`.
/// Target must implement `Stage` trait.
pub trait CanTransitionTo<Target: Stage> {}

// Logic transitions
impl CanTransitionTo<Logic> for Branch {}
impl CanTransitionTo<Logic> for Logic {} // stay in Logic

// Fxp transitions
impl CanTransitionTo<Fxp> for Branch {}
impl CanTransitionTo<Fxp> for Logic {}
impl CanTransitionTo<Fxp> for Fxp {} // stay in Fxp

// FxpToFp transitions
impl CanTransitionTo<FxpToFp> for Branch {}
impl CanTransitionTo<FxpToFp> for Logic {}
impl CanTransitionTo<FxpToFp> for Fxp {}

// Narrow transitions
impl CanTransitionTo<Narrow> for Branch {}
impl CanTransitionTo<Narrow> for Logic {}
impl CanTransitionTo<Narrow> for Fxp {}
impl CanTransitionTo<Narrow> for FxpToFp {}

// Fp transitions
impl CanTransitionTo<Fp> for Branch {}
impl CanTransitionTo<Fp> for Logic {}
impl CanTransitionTo<Fp> for Fxp {}
impl CanTransitionTo<Fp> for FxpToFp {}
impl CanTransitionTo<Fp> for Narrow {}
impl CanTransitionTo<Fp> for Fp {} // stay in Fp

// IntraSliceReduce transitions
impl CanTransitionTo<IntraSliceReduce> for Branch {}
impl CanTransitionTo<IntraSliceReduce> for Logic {}
impl CanTransitionTo<IntraSliceReduce> for Fxp {}
impl CanTransitionTo<IntraSliceReduce> for FxpToFp {}
impl CanTransitionTo<IntraSliceReduce> for Narrow {}
impl CanTransitionTo<IntraSliceReduce> for Fp {}

// FpDiv transitions
impl CanTransitionTo<FpDiv> for Branch {}
impl CanTransitionTo<FpDiv> for Logic {}
impl CanTransitionTo<FpDiv> for Fxp {}
impl CanTransitionTo<FpDiv> for FxpToFp {}
impl CanTransitionTo<FpDiv> for Narrow {}
impl CanTransitionTo<FpDiv> for Fp {}
impl CanTransitionTo<FpDiv> for IntraSliceReduce {}

// Widen transitions
impl CanTransitionTo<Widen> for Branch {}
impl CanTransitionTo<Widen> for Logic {}
impl CanTransitionTo<Widen> for Fxp {}
impl CanTransitionTo<Widen> for FxpToFp {}
impl CanTransitionTo<Widen> for Narrow {}
impl CanTransitionTo<Widen> for Fp {}
impl CanTransitionTo<Widen> for IntraSliceReduce {}
impl CanTransitionTo<Widen> for FpDiv {}

// FpToFxp transitions
impl CanTransitionTo<FpToFxp> for Branch {}
impl CanTransitionTo<FpToFxp> for Logic {}
impl CanTransitionTo<FpToFxp> for Fxp {}
impl CanTransitionTo<FpToFxp> for FxpToFp {}
impl CanTransitionTo<FpToFxp> for Narrow {}
impl CanTransitionTo<FpToFxp> for Fp {}
impl CanTransitionTo<FpToFxp> for IntraSliceReduce {}
impl CanTransitionTo<FpToFxp> for FpDiv {}
impl CanTransitionTo<FpToFxp> for Widen {}

// Clip transitions
impl CanTransitionTo<Clip> for Branch {}
impl CanTransitionTo<Clip> for Logic {}
impl CanTransitionTo<Clip> for Fxp {}
impl CanTransitionTo<Clip> for FxpToFp {}
impl CanTransitionTo<Clip> for Narrow {}
impl CanTransitionTo<Clip> for Fp {}
impl CanTransitionTo<Clip> for IntraSliceReduce {}
impl CanTransitionTo<Clip> for FpDiv {}
impl CanTransitionTo<Clip> for Widen {}
impl CanTransitionTo<Clip> for FpToFxp {}
impl CanTransitionTo<Clip> for Clip {} // stay in Clip

// InterSliceReduce transitions (after reduce: can enter VAU, or proceed to Filter/Output)
impl CanTransitionTo<Branch> for InterSliceReduce {}
impl CanTransitionTo<Filter> for InterSliceReduce {}
impl CanTransitionTo<Output> for InterSliceReduce {}

// Filter transitions (all stages can transition to Filter)
impl CanTransitionTo<Filter> for Branch {}
impl CanTransitionTo<Filter> for Logic {}
impl CanTransitionTo<Filter> for Fxp {}
impl CanTransitionTo<Filter> for FxpToFp {}
impl CanTransitionTo<Filter> for Narrow {}
impl CanTransitionTo<Filter> for Fp {}
impl CanTransitionTo<Filter> for IntraSliceReduce {}
impl CanTransitionTo<Filter> for FpDiv {}
impl CanTransitionTo<Filter> for Widen {}
impl CanTransitionTo<Filter> for FpToFxp {}
impl CanTransitionTo<Filter> for Clip {}

// Output transitions (all stages can transition to Output for commit)
impl CanTransitionTo<Output> for Branch {}
impl CanTransitionTo<Output> for Logic {}
impl CanTransitionTo<Output> for Fxp {}
impl CanTransitionTo<Output> for Filter {}
impl CanTransitionTo<Output> for FxpToFp {}
impl CanTransitionTo<Output> for Narrow {}
impl CanTransitionTo<Output> for Fp {}
impl CanTransitionTo<Output> for IntraSliceReduce {}
impl CanTransitionTo<Output> for FpDiv {}
impl CanTransitionTo<Output> for Widen {}
impl CanTransitionTo<Output> for FpToFxp {}
impl CanTransitionTo<Output> for Clip {}
impl CanTransitionTo<Output> for Output {} // self-transition for commit

// IntraSlice → InterSliceReduce (direct reduce from intra stages, Path 2)
impl CanTransitionTo<InterSliceReduce> for Branch {}
impl CanTransitionTo<InterSliceReduce> for Logic {}
impl CanTransitionTo<InterSliceReduce> for Fxp {}
impl CanTransitionTo<InterSliceReduce> for FxpToFp {}
impl CanTransitionTo<InterSliceReduce> for Widen {}
impl CanTransitionTo<InterSliceReduce> for FpToFxp {}
impl CanTransitionTo<InterSliceReduce> for Clip {}
