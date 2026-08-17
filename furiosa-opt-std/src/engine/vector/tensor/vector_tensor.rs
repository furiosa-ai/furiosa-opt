//! VectorTensor definition and implementations.
//!
//! This module defines `VectorTensor<S: Stage>`, a unified tensor type for the Vector Engine pipeline.
//! The stage parameter `S` tracks pipeline progression at compile time, with type-safe
//! transitions between stages enforced through `CanTransitionTo`, `IntraSliceStage`, and `InterSliceStage`.
//!
//! # VE Entry / Exit
//! - Entry (on CollectTensor / ContractTensor):
//!   - `vector_init()` → initialized VE input
//!   - `vector_init()`, then `vector_intra_slice_tag(TagMode)` → intra-slice (single stream)
//!   - `vector_init()`, then `vector_intra_slice_unzip(...)` → intra-slice (two-group)
//!   - `vector_init()`, then `vector_inter_slice_reduce(...)` → inter-slice reduce
//! - Exit: `vector_final()` → `VectorFinalTensor` → commit/cast/transpose
//!
//! # VeOrder (VE_ORDER)
//! Tracks which unit was entered first. Set once, never changes after.
//! - `IntraFirst` after `vector_intra_slice_tag` / `vector_intra_slice_unzip`
//! - `InterFirst` after `vector_inter_slice_reduce` from `VectorInitTensor`
//! - Preserved through all subsequent operations

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use super::VeTensorShape;

use super::VectorFinalTensor;
use crate::backend::Backend;
use crate::context::*;
use crate::engine::vector::alu::RngdAlu;
use crate::engine::vector::branch::{
    BinaryBranchedOperand, ExecutionId, RfPort, TagGuard, TagMode, TernaryBranchedOperand, VeOperandLayout,
    apply_branch_config,
};
use crate::engine::vector::layer::{FpToFxp, FxpToFp, Reinterpret};
use crate::engine::vector::op::semantics::HasConversionOp;
use crate::engine::vector::op::semantics::{HasBinaryOp, HasTernaryOp, HasUnaryOp};
use crate::engine::vector::op::{
    BinaryArgMode, ClipBinaryOpF32, ClipBinaryOpI32, FpBinaryOp, FpDivBinaryOp, FpTernaryOp, FxpBinaryOp, HasAlu,
    InterSliceReduceOpF32, InterSliceReduceOpI32, IntraSliceReduceOpF32, IntraSliceReduceOpI32, LogicBinaryOpF32,
    LogicBinaryOpI32, TernaryArgMode,
};
use crate::runtime::CurrentBackend;
use crate::tensor::*;

use crate::engine::vector::operand::{IntoBranchedOperand, IntoGuardedUnaryOp, IntoTernaryOperand, StashTransition};
use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::stage::markers as stage;
use crate::engine::vector::stage::markers::CanTransitionTo;
use crate::engine::vector::stage::markers::VeOrder;
use crate::engine::vector::stage::markers::Way::{self, Way4, Way8};
use crate::engine::vector::stage::state::VeState;
use crate::engine::vector::stash_slot::{Fresh, Occupied, StashState};
use crate::engine::vector::tensor::verify::{
    verify_vector_narrow_split, verify_vector_narrow_trim, verify_vector_widen_concat, verify_vector_widen_pad,
};

use super::vector_tensor_pair::VectorTensorPair;

/// VE input after `vector_init()`, before choosing the first block.
#[derive(Debug)]
pub struct VectorInitTensor<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Creates a new VectorInitTensor.
    pub fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> Self {
        Self { ctx, inner }
    }
}

// ============================================================================
// VeTensorData - Common tensor data without context (shared by VectorTensor and VectorTensorPair)
// ============================================================================

/// Common tensor data for VE pipeline stages, without a context reference; shared between
/// `VectorTensor` and `VectorTensorPair` groups.
#[derive(Debug)]
pub struct VeTensorData<
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    const VE_ORDER: VeOrder,
    FS: stage::VeTensorContext = stage::Standalone,
    const W: Way = { Way8 },
> {
    pub(crate) inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    pub(crate) tag: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    pub(crate) ve_state: VeState<Stash>,
    pub(crate) _stage: PhantomData<S>,
    pub(crate) _filter_state: PhantomData<FS>,
}

// ============================================================================
// VectorTensor - Context + VeTensorData
// ============================================================================

/// Unified tensor type for all VE pipeline stages.
///
/// The `S` type parameter represents the current pipeline stage, enabling
/// compile-time verification of stage transitions via the `CanTransitionTo` trait.
///
/// The `FS` type parameter represents the tensor context:
/// - `Standalone`: Single-stream context (default). `vector_filter` and `vector_stash` are available.
/// - `Group`: Per-group context after `vector_intra_slice_unzip`. `vector_filter` and `vector_stash` are NOT available.
/// - `Zipped`: After merging the two groups via a `_zip` method. `vector_filter` and `vector_stash` are NOT available.
///
/// The `W` type parameter represents the way:
/// - `Way8`: Default 8-element flit mode. Float operations are NOT available.
/// - `Way4`: After `vector_narrow_split` or `vector_narrow_trim`, front-4-only. Float operations are available.
#[derive(Debug)]
pub struct VectorTensor<
    'l,
    const T: Tu,
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    const VE_ORDER: VeOrder,
    FS: stage::VeTensorContext = stage::Standalone,
    const W: Way = { Way8 },
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) data: VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>,
}

// ============================================================================
// VeTensorData - Basic accessors
// ============================================================================

impl<
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const W: Way,
    const VE_ORDER: VeOrder,
> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>
{
    /// Returns a mutable reference to the VE state.
    pub fn ve_state_mut(&mut self) -> &mut VeState<Stash> {
        &mut self.ve_state
    }

    /// Returns a reference to the VE state.
    pub fn ve_state(&self) -> &VeState<Stash> {
        &self.ve_state
    }

    /// Returns a reference to the inner tensor.
    pub fn inner(&self) -> &Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        &self.inner
    }

    /// Returns a reference to the tag tensor.
    pub fn tag(&self) -> &Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        &self.tag
    }

    /// Consumes the data and returns its parts.
    pub fn into_parts(
        self,
    ) -> (
        Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VeState<Stash>,
    ) {
        (self.inner, self.tag, self.ve_state)
    }

    /// Internal helper for binary operations.
    /// Applies a binary operation with ALU tracking and stash support.
    pub(crate) fn apply_binary<NextStage: stage::Stage, NextFS: stage::VeTensorContext>(
        mut self,
        alu: RngdAlu,
        op_fn: impl Fn(D, D) -> D + Sync,
        operand: &BinaryBranchedOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        stash_data: Option<Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>>,
    ) -> VeTensorData<NextStage, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, NextFS, W> {
        self.ve_state.use_alu(alu);
        let result = apply_binary_op(&self.inner, &self.tag, op_fn, operand, stash_data.as_ref());
        VeTensorData {
            inner: result,
            tag: self.tag,
            ve_state: self.ve_state,
            _stage: PhantomData,
            _filter_state: PhantomData,
        }
    }
}

impl<
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const W: Way,
    const VE_ORDER: VeOrder,
> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>
{
    /// Re-tags the stash typestate via the operand's [`StashTransition`], after the op already
    /// ran through the ordinary binary/ternary path (so vISA lowering is unchanged).
    pub(crate) fn apply_stash_transition<Op: StashTransition<Stash, D, W>>(
        self,
    ) -> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER, FS, W> {
        VeTensorData {
            inner: self.inner,
            tag: self.tag,
            ve_state: Op::transition(self.ve_state),
            _stage: PhantomData,
            _filter_state: PhantomData,
        }
    }
}

// ============================================================================
// VectorTensor - Basic accessors (delegates to VeTensorData)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const W: Way,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>
{
    /// Consumes the tensor and returns its parts.
    pub fn into_parts(
        self,
    ) -> (
        &'l mut TuContext<{ T }>,
        Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VeState<Stash>,
    ) {
        let (inner, tag, ve_state) = self.data.into_parts();
        (self.ctx, inner, tag, ve_state)
    }

    /// Consumes the tensor and returns ctx and data separately.
    pub fn into_ctx_and_data(
        self,
    ) -> (
        &'l mut TuContext<{ T }>,
        VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>,
    ) {
        (self.ctx, self.data)
    }

    /// Returns a mutable reference to the VE state.
    pub fn ve_state_mut(&mut self) -> &mut VeState<Stash> {
        self.data.ve_state_mut()
    }

    /// Returns a reference to the VE state.
    pub fn ve_state(&self) -> &VeState<Stash> {
        self.data.ve_state()
    }

    /// Returns a reference to the inner tensor.
    pub fn inner(&self) -> &Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        self.data.inner()
    }

    /// Returns a reference to the tag tensor.
    pub fn tag(&self) -> &Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        self.data.tag()
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W> {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W> {
        &mut self.data
    }

    /// Creates a new VectorTensor from parts.
    pub fn from_parts(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        tag: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        ve_state: VeState<Stash>,
    ) -> Self {
        Self {
            ctx,
            data: VeTensorData {
                inner,
                tag,
                ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }

    /// Creates a new VectorTensor from context and data.
    pub fn from_ctx_and_data(
        ctx: &'l mut TuContext<{ T }>,
        data: VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>,
    ) -> Self {
        Self { ctx, data }
    }

    /// Internal helper for binary operations: reads the stash the operand asks for, runs the op with ALU
    /// tracking, and moves the stash typestate. All three come from the one `Op`, so a read without a
    /// transition is not expressible at the call sites. The `D` and `W` in the bound are this impl's own,
    /// so an op method naming a different scalar or way fails here rather than reading the stash at the
    /// wrong one.
    pub(crate) fn do_binary<NextStage: stage::Stage, NextFS: stage::VeTensorContext, Op>(
        self,
        op: impl HasAlu + HasBinaryOp<D>,
        mode: Option<BinaryArgMode>,
        operand: Op,
    ) -> VectorTensor<'l, T, NextStage, D, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER, NextFS, W>
    where
        Op: IntoBranchedOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<Stash, D, W>,
    {
        let stash_data = Op::stashed(self.ve_state());
        let operand = operand.into_branched_operand();
        let data = self
            .data
            .apply_binary(op.alu(), op.binary_op_fn(mode), &operand, stash_data)
            .apply_stash_transition::<Op>();
        VectorTensor { ctx: self.ctx, data }
    }
}

// Separate impl for stash on VeTensorData - only stages implementing Stashable can use this
// Separate impl for stash on VectorTensor - delegates to VeTensorData
// Also requires Filterable state (not available after binary operations)
impl<
    'l,
    const T: Tu,
    S: stage::Stashable,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    const W: Way,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Fresh, VE_ORDER, stage::Standalone, W>
{
    /// Writes the current tensor data to the operand register (the RNGD "stash").
    ///
    /// Write-once: this op is defined only on the [`Fresh`](crate::engine::vector::stash_slot::Fresh)
    /// state and flips it to [`Occupied`](crate::engine::vector::stash_slot::Occupied), so a second
    /// `vector_stash` has no impl and does not compile. Read-once is enforced by the
    /// [`Stash`](crate::engine::vector::operand::Stash) operand, which consumes `Occupied` into
    /// [`Spent`](crate::engine::vector::stash_slot::Spent) rather than back to `Fresh`, so a read never
    /// re-arms the write. See the stash-slot [module docs] for the state machine. Only on `Stashable`
    /// stages (Tag, Logic, Fxp, Narrow, Fp, FpDiv, Clip) in the `Standalone` context.
    ///
    /// Both HW-illegal double writes are compile errors:
    ///
    /// ```text
    /// t.vector_stash().vector_fp_binary(op, Stash).vector_stash();  // read then re-stash: Spent has no vector_stash
    /// t.vector_stash().vector_stash();                              // no read between: Occupied has no vector_stash
    /// ```
    ///
    /// The stash takes the scalar the stream carries at the write, so `vector_reinterpret` before a
    /// `vector_stash` decides what gets stashed, and a read at the other scalar does not compile.
    ///
    /// [module docs]: crate::engine::vector::stash_slot
    #[primitive(VectorTensor::vector_stash)]
    pub fn vector_stash(
        self,
    ) -> VectorTensor<
        'l,
        T,
        S,
        D,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        Occupied<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>, W>,
        VE_ORDER,
        stage::Standalone,
        W,
    > {
        let new_ve_state = self.data.ve_state.write_stash(&self.data.inner);
        VectorTensor {
            ctx: self.ctx,
            data: VeTensorData {
                inner: self.data.inner,
                tag: self.data.tag,
                ve_state: new_ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }
}

// ============================================================================
// Output operations for VectorTensor (all stages, all filter states)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::Commitable,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
{
    /// Exits the Vector Engine pipeline and returns a stream tensor.
    /// After this, commit/cast/transpose are available through the stream tensor API.
    #[primitive(VectorTensor::vector_final)]
    pub fn vector_final(self) -> VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet> {
        VectorFinalTensor::new(self.ctx, self.data.inner)
    }
}

// ============================================================================
// Inter-slice reduce — from IntraSlice stages
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::IntraSliceStage + stage::CanTransitionTo<stage::InterSliceReduce>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::Commitable,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, Stash, { VeOrder::IntraFirst }, FS, { Way8 }>
{
    /// Performs inter-slice reduce for i32 from intra-slice stages.
    /// Only available when VeOrder::IntraFirst (intra-slice was entered first).
    #[primitive(VectorTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpI32,
    ) -> VectorInterSliceReduceTensor<'l, T, i32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::IntraFirst }>
    {
        let reduced = self.data.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}

impl<
    'l,
    const T: Tu,
    S: stage::IntraSliceStage + stage::CanTransitionTo<stage::InterSliceReduce>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::Commitable,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, { VeOrder::IntraFirst }, FS, { Way8 }>
{
    /// Performs inter-slice reduce for f32 from intra-slice stages.
    /// Only available when VeOrder::IntraFirst (intra-slice was entered first).
    #[primitive(VectorTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpF32,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::IntraFirst }>
    {
        let reduced = self.data.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}

// ============================================================================
// Constructor helpers (pub(crate) for use from stream_tensor.rs)
// ============================================================================

/// Creates a VectorInterSliceReduceTensor from reduced tensor data.
pub(crate) fn create_inter_slice_reduce_tensor<
    'l,
    const T: Tu,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    const VE_ORDER: VeOrder,
>(
    ctx: &'l mut TuContext<{ T }>,
    inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
) -> VectorInterSliceReduceTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, VE_ORDER> {
    VectorTensor {
        ctx,
        data: VeTensorData {
            inner,
            tag: Tensor::zeroed(),
            ve_state: VeState::new(),
            _stage: PhantomData,
            _filter_state: PhantomData,
        },
    }
}

// ============================================================================
// VectorInitTensor methods
// ============================================================================

// ANCHOR: vector_init_intra_slice_methods_impl
impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Enters VE intra-slice pipeline (single stream).
    // ANCHOR: vector_intra_slice_tag
    #[primitive(VectorInitTensor::vector_intra_slice_tag)]
    pub fn vector_intra_slice_tag(
        self,
        branch: TagMode<D>,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, Fresh, { VeOrder::IntraFirst }> {
        // ANCHOR_END: vector_intra_slice_tag
        VectorBranchTensor::new(self.ctx, self.inner, branch)
    }

    /// Enters VE intra-slice pipeline (two-group / unzip).
    // ANCHOR: vector_intra_slice_unzip
    #[primitive(VectorInitTensor::vector_intra_slice_unzip)]
    pub fn vector_intra_slice_unzip<I: AxisName, TileTime: M, SplitTime: M>(
        self,
    ) -> VectorTensorPair<'l, T, D, stage::Tag, Chip, Cluster, Slice, SplitTime, Packet> {
        // ANCHOR_END: vector_intra_slice_unzip
        VectorTensorPair::new::<I, Time, TileTime>(self.ctx, self.inner)
    }
}
// ANCHOR_END: vector_init_intra_slice_methods_impl

// ANCHOR: vector_init_inter_slice_reduce_i32_impl
impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for i32 as the first VE operation.
    // ANCHOR: init_inter_slice_reduce_i32
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpI32,
    ) -> VectorInterSliceReduceTensor<'l, T, i32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::InterFirst }>
    {
        // ANCHOR_END: init_inter_slice_reduce_i32
        let reduced = self.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}
// ANCHOR_END: vector_init_inter_slice_reduce_i32_impl

// ANCHOR: vector_init_inter_slice_reduce_f32_impl
impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for f32 as the first VE operation.
    // ANCHOR: init_inter_slice_reduce_f32
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpF32,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::InterFirst }>
    {
        // ANCHOR_END: init_inter_slice_reduce_f32
        let reduced = self.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}
// ANCHOR_END: vector_init_inter_slice_reduce_f32_impl

// ============================================================================
// Type aliases for VectorTensor at each stage
// ============================================================================

/// Tensor after inter-slice reduce.
pub type VectorInterSliceReduceTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, const VE_ORDER: VeOrder> =
    VectorTensor<
        'l,
        T,
        stage::InterSliceReduce,
        D,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        Fresh,
        VE_ORDER,
        stage::Standalone,
        { Way8 },
    >;

// ============================================================================
// Inter→Intra entry — vector_intra_slice_tag on InterSliceStage (requires VeOrder::InterFirst)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::InterSliceStage + stage::CanTransitionTo<stage::Tag>,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::Commitable,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, { VeOrder::InterFirst }, FS, { Way8 }>
{
    /// Enters intra-slice pipeline from inter-slice output. Requires VeOrder::InterFirst.
    /// Preserves VeOrder::InterFirst.
    #[primitive(VectorTensor::vector_intra_slice_tag)]
    pub fn vector_intra_slice_tag(
        self,
        branch: TagMode<D>,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, Fresh, { VeOrder::InterFirst }> {
        VectorBranchTensor::new(self.ctx, self.data.inner, branch)
    }
}

/// Tensor after branch unit.
pub type VectorBranchTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Tag, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, const VE_ORDER: VeOrder>
    VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, Fresh, VE_ORDER>
{
    /// Creates a new VectorBranchTensor from inner tensor and branch configuration.
    pub fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        branch_config: TagMode<D>,
    ) -> Self {
        assert_eq!(
            Packet::SIZE,
            8,
            "VectorTensor requires Packet of 8 elements (one flit) in Way8 mode, got {}",
            Packet::SIZE,
        );
        let tag = apply_branch_config(&inner, &branch_config);
        Self::from_parts(ctx, inner, tag, VeState::new())
    }
}

/// Tensor after logic operations.
pub type VectorLogicTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Logic, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after fixed-point operations.
pub type VectorFxpTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Fxp, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after i32 to f32 conversion.
pub type VectorFxpToFpTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::FxpToFp, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after narrow layer (split or clip).
pub type VectorNarrowTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way4 },
> = VectorTensor<'l, T, stage::Narrow, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after floating-point operations.
pub type VectorFpTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way4 },
> = VectorTensor<'l, T, stage::Fp, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after intra-slice reduce operations.
pub type VectorIntraSliceReduceTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way4 },
> = VectorTensor<'l, T, stage::IntraSliceReduce, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after fp division.
pub type VectorFpDivTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way4 },
> = VectorTensor<'l, T, stage::FpDiv, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after widen layer (concat or pad).
pub type VectorWidenTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Widen, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after f32 to i32 conversion.
pub type VectorFpToFxpTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::FpToFxp, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after clip operations.
pub type VectorClipTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Clip, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

/// Tensor after filter operation.
pub type VectorFilterTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Filter, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>;

// ============================================================================
// Helper functions for applying operations with tag
// ============================================================================

/// The tag tensor's byte as an execution id.
///
/// The tag unit writes four bits, so a wider value means the tensor was built by something other than
/// `apply_branch_config` -- worth failing on rather than aliasing onto the low four bits.
fn exec_id(raw: u8) -> ExecutionId {
    ExecutionId::try_new(raw)
        .unwrap_or_else(|| panic!("an execution id is four bits, but the tag tensor holds {raw:#04x}"))
}

/// Applies one VE pass: every cell takes the first filled slot of `operand` whose [`TagGuard`] its
/// `tag` satisfies, and `combine` runs once with that slot's operands. A cell no slot claims passes
/// through unchanged, and empty (`None`) slots contribute nothing.
///
/// The early return below is not an engine shortcut: first match is the hardware's rule, so a later
/// slot never re-combines a cell an earlier one claimed. See `BranchedOperand`.
///
/// Binary and ternary passes share this body because they share the slot layout ([`VeOperandLayout`]);
/// the arity shows up only as the `operand1` handed to `combine`. Stash data arrives as a `Tensor`,
/// already transposed, and is read by the rhs port when that port carries no register.
fn apply_slots<D: VeScalar, Mapping: M, L: VeOperandLayout<D, Mapping>>(
    data: &Tensor<D, Mapping>,
    tag: &Tensor<u8, Mapping>,
    operand: &L,
    stash_data: Option<&Tensor<D, Mapping>>,
    combine: impl Fn(D, D, L::Operand1) -> D + Sync,
) -> Tensor<D, Mapping> {
    // A slot's guard, and an immediate slot's rhs, are constants for the whole pass, so the per-cell
    // closure captures them by value instead of borrowing the operand or broadcasting the rhs to a
    // full tensor.
    let regs = operand
        .regs()
        .map(|slot| slot.map(|(guard, operand0, operand1)| (*guard, operand0, operand1)));
    let port = operand.port().map(|(guard, port, operand1)| {
        let rhs = match port {
            RfPort::External(register) => register,
            RfPort::Stash => stash_data.expect(
                "a stash operand reached the engine without a stashed tensor: use the `Stash` operand, whose \
                 `StashTransition` supplies it, not a hand-built slot",
            ),
        };
        (*guard, rhs, operand1)
    });
    // The port register feeds the port slot alone, so with no port the mainstream stands in for it
    // and the closure below never reads that argument.
    let port_rhs = port.map_or(data, |(_, rhs, _)| rhs);
    let port_slot = port.map(|(guard, _, operand1)| (guard, operand1));

    let inner = CurrentBackend::zip3_with(
        &data.inner,
        &port_rhs.inner,
        &tag.inner,
        move |cell, port_cell, raw_id| {
            let id = exec_id(raw_id);
            for (guard, operand0, operand1) in regs.iter().flatten() {
                if guard.admits(id) {
                    return combine(cell, *operand0, *operand1);
                }
            }
            match port_slot {
                Some((guard, operand1)) if guard.admits(id) => combine(cell, port_cell, operand1),
                _ => cell,
            }
        },
    );
    Tensor::from_inner(inner)
}

/// Applies a binary operation with branch-conditional execution: [`apply_slots`] with no `operand1`.
pub(super) fn apply_binary_op<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    tag: &Tensor<u8, Mapping>,
    op: impl Fn(D, D) -> D + Sync,
    operand: &BinaryBranchedOperand<D, Mapping>,
    stash_data: Option<&Tensor<D, Mapping>>,
) -> Tensor<D, Mapping> {
    apply_slots(data, tag, operand, stash_data, |mainstream, rhs, ()| {
        op(mainstream, rhs)
    })
}

/// Applies a unary operation where `guard` matches, leaving every other cell alone.
///
/// A unary node has no slots, so there is no priority order to resolve: one guard either claims a cell
/// or does not. [`apply_slots`] is what the branched arities need instead.
pub(super) fn apply_unary_op_where<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    tag: &Tensor<u8, Mapping>,
    guard: TagGuard,
    op: impl Fn(D) -> D + Sync,
) -> Tensor<D, Mapping> {
    let inner = CurrentBackend::zip_with(&data.inner, &tag.inner, move |cell, raw_id| {
        if guard.admits(exec_id(raw_id)) { op(cell) } else { cell }
    });
    Tensor::from_inner(inner)
}

/// Applies a ternary operation with branch-conditional execution: [`apply_slots`] with each slot's
/// own `operand1`, which the hardware selects per branch just like `operand0`. Ternary operations are
/// only supported for f32 tensors.
pub(super) fn apply_ternary_op<Mapping: M>(
    data: &Tensor<f32, Mapping>,
    tag: &Tensor<u8, Mapping>,
    op: impl Fn(f32, f32, f32) -> f32 + Sync,
    operand: &TernaryBranchedOperand<f32, Mapping>,
    stash_data: Option<&Tensor<f32, Mapping>>,
) -> Tensor<f32, Mapping> {
    apply_slots(data, tag, operand, stash_data, op)
}

// ============================================================================
// Logic operations (i32 only)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Logic>,
{
    /// Logic binary operation (i32 only). Requires `Way8` mode. A [`Stash`](crate::prelude::Stash) operand reads
    /// the stash (read-once).
    #[primitive(VectorTensor::vector_logic)]
    pub fn vector_logic<Op>(
        self,
        op: LogicBinaryOpI32,
        operand: Op,
    ) -> VectorLogicTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, i32, { Way8 }>,
    {
        self.do_binary::<stage::Logic, stage::Standalone, Op>(op, None, operand)
    }

    /// Logic binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic_with_mode)]
    pub fn vector_logic_with_mode<Op>(
        self,
        op: LogicBinaryOpI32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorLogicTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, i32, { Way8 }>,
    {
        self.do_binary::<stage::Logic, stage::Standalone, Op>(op, Some(mode), operand)
    }
}

// ============================================================================
// Logic operations (f32 only)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Logic>,
{
    /// Logic binary operation (f32 only). Requires `Way8` mode. A [`Stash`](crate::prelude::Stash) operand reads
    /// the stash (read-once).
    #[primitive(VectorTensor::vector_logic)]
    pub fn vector_logic<Op>(
        self,
        op: LogicBinaryOpF32,
        operand: Op,
    ) -> VectorLogicTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way8 }>,
    {
        self.do_binary::<stage::Logic, stage::Standalone, Op>(op, None, operand)
    }

    /// Logic binary operation with explicit mode (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic_with_mode)]
    pub fn vector_logic_with_mode<Op>(
        self,
        op: LogicBinaryOpF32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorLogicTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way8 }>,
    {
        self.do_binary::<stage::Logic, stage::Standalone, Op>(op, Some(mode), operand)
    }
}

// ============================================================================
// Fixed-point operations (i32 only)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Fxp>,
{
    /// Fixed-point binary operation (i32 only). Requires `Way8` mode. A [`Stash`](crate::prelude::Stash) operand
    /// reads the stash (read-once).
    #[primitive(VectorTensor::vector_fxp)]
    pub fn vector_fxp<Op>(
        self,
        op: FxpBinaryOp,
        operand: Op,
    ) -> VectorFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, i32, { Way8 }>,
    {
        self.do_binary::<stage::Fxp, stage::Standalone, Op>(op, None, operand)
    }

    /// Fixed-point binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fxp_with_mode)]
    pub fn vector_fxp_with_mode<Op>(
        self,
        op: FxpBinaryOp,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, i32, { Way8 }>,
    {
        self.do_binary::<stage::Fxp, stage::Standalone, Op>(op, Some(mode), operand)
    }
}

// ============================================================================
// FxpToFp conversion (i32 -> f32)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::FxpToFp>,
{
    /// Converts i32 to f32. Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fxp_to_fp)]
    pub fn vector_fxp_to_fp(
        self,
        int_width: u32,
    ) -> VectorFxpToFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER> {
        let op = FxpToFp::new(int_width);
        let op_fn = op.op_fn();

        let result = self.inner().map(&op_fn);

        let (ctx, _inner, tag, ve_state) = self.into_parts();
        VectorFxpToFpTensor::from_parts(ctx, result, tag, ve_state)
    }
}

// ============================================================================
// Narrow operations (split / clip)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Narrow>,
{
    /// Narrow layer (split). Requires `Way8` mode.
    ///
    /// Takes an 8-element packet, splits it into front 4 + back 4.
    /// The factor of 2 goes into Time, and the output is `Way4` with 4-element packets.
    /// Output: `Time2 = Time × 2`, `Packet2` = front 4 of Packet (size 4).
    #[primitive(VectorTensor::vector_narrow_split)]
    pub fn vector_narrow_split<Time2: M, Packet2: M>(
        self,
    ) -> VectorNarrowTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, Stash, VE_ORDER, FS, { Way4 }> {
        verify_vector_narrow_split::<Time, Packet, Time2, Packet2>();

        let (ctx, inner, tag, ve_state) = self.into_parts();

        let split_inner = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);
        let split_eid = tag.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);

        VectorNarrowTensor::from_parts(ctx, split_inner, split_eid, ve_state)
    }
}

// ============================================================================
// vector_narrow_trim: strip back-4 dummy from Packet 8 → 4 (type-only, no-op at hardware level)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Narrow>,
{
    /// Strip the back-4 dummy lanes from an 8-element packet, yielding a 4-element packet.
    /// Transitions from `Way8` to `Way4` mode and enters the `Narrow` stage.
    ///
    /// This is a type-system-only operation — no hardware instruction is emitted.
    /// Use this when the back 4 lanes are already padding (≤ 4 real elements).
    /// For packets with > 4 real elements, use `vector_narrow_split()` instead.
    #[primitive(VectorTensor::vector_narrow_trim)]
    pub fn vector_narrow_trim<Packet2: M>(
        self,
    ) -> VectorNarrowTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet2, Stash, VE_ORDER, FS, { Way4 }> {
        verify_vector_narrow_trim::<Packet, Packet2>();

        let (ctx, inner, tag, ve_state) = self.into_parts();

        let stripped = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);
        let stripped_eid = tag.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);

        VectorNarrowTensor::from_parts(ctx, stripped, stripped_eid, ve_state)
    }
}

// ============================================================================
// Floating-point operations (f32 only)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Fp>,
{
    /// Fp unary operation (f32 only), applied where its guard matches.
    ///
    /// Takes the op bare to run everywhere, or paired with a [`TagGuard`] to run on the elements that
    /// guard admits:
    ///
    /// ```ignore
    /// tensor.vector_fp_unary(FpUnaryOp::Sigmoid)                   // everywhere
    /// tensor.vector_fp_unary((negative, FpUnaryOp::Sigmoid))       // where `negative` matches
    /// ```
    ///
    /// One method rather than two because that is how the binary and logic ops read: an operand is
    /// bare or it is a [`Branched`](crate::engine::vector::operand::Branched) layout, and the bare one
    /// means [`TagGuard::all`]. A unary op has no operand to carry the guard, so the pairing goes on
    /// the op.
    #[primitive(VectorTensor::vector_fp_unary)]
    pub fn vector_fp_unary<Op: IntoGuardedUnaryOp>(
        mut self,
        op: Op,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER> {
        let (guard, op) = op.into_guarded_unary_op();
        self.ve_state_mut().use_alu(op.alu());
        let result = apply_unary_op_where(self.inner(), self.tag(), guard, op.unary_op_fn());
        let (ctx, _inner, tag, ve_state) = self.into_parts();
        VectorFpTensor::from_parts(ctx, result, tag, ve_state)
    }

    /// Fp binary operation (f32 only). The operand is a const, `&VrfTensor`, or
    /// [`Stash`](crate::prelude::Stash); the return typestate follows its [`StashTransition`].
    #[primitive(VectorTensor::vector_fp_binary)]
    pub fn vector_fp_binary<Op>(
        self,
        op: FpBinaryOp,
        operand: Op,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way4 }>,
    {
        self.do_binary::<stage::Fp, stage::Standalone, Op>(op, None, operand)
    }

    /// Fp binary operation with explicit mode (f32 only). See [`vector_fp_binary`](Self::vector_fp_binary).
    #[primitive(VectorTensor::vector_fp_binary_with_mode)]
    pub fn vector_fp_binary_with_mode<Op>(
        self,
        op: FpBinaryOp,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way4 }>,
    {
        self.do_binary::<stage::Fp, stage::Standalone, Op>(op, Some(mode), operand)
    }

    /// Fp ternary operation (f32 only).
    ///
    /// # Example
    /// ```ignore
    /// // FmaF: result = data * operand0 + operand1; operand0 may be const/VRF, or Stash to read
    /// // the stash: tensor.vector_fp_ternary(FpTernaryOp::FmaF, (Stash, 3.0f32))
    /// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 3.0f32))
    /// ```
    #[primitive(VectorTensor::vector_fp_ternary)]
    pub fn vector_fp_ternary<Op>(
        self,
        op: FpTernaryOp,
        operands: Op,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoTernaryOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way4 }>,
    {
        self.vector_fp_ternary_with_mode(op, TernaryArgMode::Mode012, operands)
    }

    /// Fp ternary operation with explicit mode (f32 only). A `(Stash, c)` operand reads the
    /// stash (read-once) as operand0.
    #[primitive(VectorTensor::vector_fp_ternary_with_mode)]
    pub fn vector_fp_ternary_with_mode<Op>(
        mut self,
        op: FpTernaryOp,
        mode: TernaryArgMode,
        operands: Op,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoTernaryOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way4 }>,
    {
        let stash_data = Op::stashed(self.ve_state());
        let operands = operands.into_ternary_operand();
        self.ve_state_mut().use_alu(op.alu());
        let op_fn = op.ternary_op_fn(Some(mode));
        let result = apply_ternary_op(self.inner(), self.tag(), op_fn, &operands, stash_data.as_ref());
        let (ctx, _inner, tag, ve_state) = self.into_parts();
        let data = VeTensorData {
            inner: result,
            tag,
            ve_state,
            _stage: PhantomData,
            _filter_state: PhantomData,
        }
        .apply_stash_transition::<Op>();
        VectorFpTensor::from_ctx_and_data(ctx, data)
    }
}

// ============================================================================
// Intra-slice reduce operations
// ============================================================================

/// Verifies that all reduced axes (quotient of input / output shape) match the expected ident.
fn verify_reduce_label(time: Mapping, packet: Mapping, out_time: Mapping, out_packet: Mapping, reduce_label: &Ident) {
    furiosa_opt_lower::config_reduce_label(&time, &packet, &out_time, &out_packet, reduce_label)
        .unwrap_or_else(|message| panic!("{message}"));
}

/// Reduces tag tensor by keeping the last value (hardware semantics: all reduced
/// elements share the same tag).
fn reduce_tag<Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, OutTime: M, OutPacket: M>(
    tag: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
) -> Tensor<u8, VeTensorShape<Chip, Cluster, Slice, OutTime, OutPacket>> {
    tag.reduce::<VeTensorShape<Chip, Cluster, Slice, OutTime, OutPacket>>(|_, y| y, 0, false)
}

// ANCHOR: intra_slice_reduce_i32
impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::IntraSliceReduce>,
{
    /// Intra-slice reduce operation (i32).
    #[primitive(VectorTensor::vector_intra_slice_reduce)]
    pub fn vector_intra_slice_reduce<Reduce: AxisName, OutTime: M, OutPacket: M>(
        mut self,
        op: IntraSliceReduceOpI32,
    ) -> VectorIntraSliceReduceTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        OutTime,
        OutPacket,
        Stash,
        VE_ORDER,
        stage::Standalone,
        { Way4 },
    >
// ANCHOR_END: intra_slice_reduce_i32
    {
        self.ve_state_mut().use_alu(op.alu());
        let (ctx, inner, tag, ve_state) = self.into_parts();
        verify_reduce_label(
            Time::to_value(),
            Packet::to_value(),
            OutTime::to_value(),
            OutPacket::to_value(),
            &Reduce::NAME,
        );
        let reduced_inner = inner.reduce::<VeTensorShape<Chip, Cluster, Slice, OutTime, OutPacket>>(
            op.reduce_fn(),
            op.identity(),
            false,
        );
        let reduced_eid = reduce_tag::<Chip, Cluster, Slice, Time, Packet, OutTime, OutPacket>(tag);
        VectorIntraSliceReduceTensor::from_parts(ctx, reduced_inner, reduced_eid, ve_state)
    }
}

// ANCHOR: intra_slice_reduce_f32
impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::IntraSliceReduce>,
{
    /// Intra-slice reduce operation (f32).
    #[primitive(VectorTensor::vector_intra_slice_reduce)]
    pub fn vector_intra_slice_reduce<Reduce: AxisName, OutTime: M, OutPacket: M>(
        mut self,
        op: IntraSliceReduceOpF32,
    ) -> VectorIntraSliceReduceTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        OutTime,
        OutPacket,
        Stash,
        VE_ORDER,
        stage::Standalone,
        { Way4 },
    >
// ANCHOR_END: intra_slice_reduce_f32
    {
        self.ve_state_mut().use_alu(op.alu());
        let (ctx, inner, tag, ve_state) = self.into_parts();
        verify_reduce_label(
            Time::to_value(),
            Packet::to_value(),
            OutTime::to_value(),
            OutPacket::to_value(),
            &Reduce::NAME,
        );
        let reduced_inner = inner.reduce::<VeTensorShape<Chip, Cluster, Slice, OutTime, OutPacket>>(
            op.reduce_fn(),
            op.identity(),
            false,
        );
        let reduced_eid = reduce_tag::<Chip, Cluster, Slice, Time, Packet, OutTime, OutPacket>(tag);
        VectorIntraSliceReduceTensor::from_parts(ctx, reduced_inner, reduced_eid, ve_state)
    }
}

// ============================================================================
// FpDiv operations
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::FpDiv>,
{
    /// Floating-point division. The fp-div ALU only supports a single op (`DivF`), so the
    /// operation enum is implicit; only operand and mode are user-facing. Like `vector_fp_binary`,
    /// a [`Stash`](crate::prelude::Stash) divisor reads the stash and the return typestate follows
    /// its [`StashTransition`] (read-once).
    #[primitive(VectorTensor::vector_fp_div)]
    pub fn vector_fp_div<Op>(
        self,
        operand: Op,
    ) -> VectorFpDivTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER, FS, { Way4 }>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way4 }>,
    {
        self.do_binary::<stage::FpDiv, FS, Op>(FpDivBinaryOp::DivF, None, operand)
    }

    /// Floating-point division with explicit mode. See [`vector_fp_div`](Self::vector_fp_div).
    #[primitive(VectorTensor::vector_fp_div_with_mode)]
    pub fn vector_fp_div_with_mode<Op>(
        self,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorFpDivTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER, FS, { Way4 }>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way4 }>,
    {
        self.do_binary::<stage::FpDiv, FS, Op>(FpDivBinaryOp::DivF, Some(mode), operand)
    }
}

// ============================================================================
// Widen operations (concat / pad)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Widen>,
{
    /// Widen layer (concat). Requires `Way4` mode.
    ///
    /// Reverse of split. Takes 4-element packets from 2 consecutive time steps,
    /// merges them into one 8-element packet and transitions to `Way8`.
    /// `Time2 = Time / 2`, `Packet2` = Packet combined with factor of 2 from Time.
    #[primitive(VectorTensor::vector_widen_concat)]
    pub fn vector_widen_concat<Time2: M, Packet2: M>(
        self,
    ) -> VectorWidenTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, Stash, VE_ORDER, FS, { Way8 }> {
        verify_vector_widen_concat::<Time, Packet, Time2, Packet2>();

        let (ctx, inner, tag, ve_state) = self.into_parts();

        let concat_inner = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);
        let concat_eid = tag.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);

        VectorWidenTensor::from_parts(ctx, concat_inner, concat_eid, ve_state)
    }
}

// ============================================================================
// vector_widen_pad: pad Packet 4 → 8 with dummy (type-only, no-op at hardware level)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Widen>,
{
    /// Pad a 4-element packet back to 8 by adding dummy lanes.
    /// Transitions from `Way4` to `Way8` mode and enters the `Widen` stage.
    ///
    /// This is a type-system-only operation — no hardware instruction is emitted.
    /// Reverse of `vector_narrow_trim`. Use this when no time-dimension merging is needed.
    /// For merging split time steps back, use `vector_widen_concat()` instead.
    #[primitive(VectorTensor::vector_widen_pad)]
    pub fn vector_widen_pad<Packet2: M>(
        self,
    ) -> VectorWidenTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet2, Stash, VE_ORDER, FS, { Way8 }> {
        verify_vector_widen_pad::<Packet, Packet2>();

        let (ctx, inner, tag, ve_state) = self.into_parts();

        let padded = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);
        let padded_eid = tag.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);

        VectorWidenTensor::from_parts(ctx, padded, padded_eid, ve_state)
    }
}

// ============================================================================
// FpToFxp conversion (f32 -> i32)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::FpToFxp>,
{
    /// Converts f32 to i32. Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fp_to_fxp)]
    pub fn vector_fp_to_fxp(
        self,
        int_width: u32,
    ) -> VectorFpToFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER> {
        let op = FpToFxp::new(int_width);
        let op_fn = op.op_fn();
        let result = self.inner().map(&op_fn);

        let (ctx, _inner, tag, ve_state) = self.into_parts();
        VectorFpToFxpTensor::from_parts(ctx, result, tag, ve_state)
    }
}

// ============================================================================
// Clip operations (i32 only)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Clip>,
{
    /// Clip binary operation (i32 only). Requires `Way8` mode. A [`Stash`](crate::prelude::Stash) operand reads
    /// the stash (read-once).
    #[primitive(VectorTensor::vector_clip)]
    pub fn vector_clip<Op>(
        self,
        op: ClipBinaryOpI32,
        operand: Op,
    ) -> VectorClipTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, i32, { Way8 }>,
    {
        self.do_binary::<stage::Clip, stage::Standalone, Op>(op, None, operand)
    }

    /// Clip binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip_with_mode)]
    pub fn vector_clip_with_mode<Op>(
        self,
        op: ClipBinaryOpI32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorClipTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, i32, { Way8 }>,
    {
        self.do_binary::<stage::Clip, stage::Standalone, Op>(op, Some(mode), operand)
    }
}

// ============================================================================
// Clip operations (f32 only)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Clip>,
{
    /// Clip binary operation (f32 only). Requires `Way8` mode. A [`Stash`](crate::prelude::Stash) operand reads
    /// the stash (read-once).
    #[primitive(VectorTensor::vector_clip)]
    pub fn vector_clip<Op>(
        self,
        op: ClipBinaryOpF32,
        operand: Op,
    ) -> VectorClipTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way8 }>,
    {
        self.do_binary::<stage::Clip, stage::Standalone, Op>(op, None, operand)
    }

    /// Clip binary operation with explicit mode (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip_with_mode)]
    pub fn vector_clip_with_mode<Op>(
        self,
        op: ClipBinaryOpF32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorClipTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Op::Next, VE_ORDER>
    where
        Op: IntoBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
            + StashTransition<Stash, f32, { Way8 }>,
    {
        self.do_binary::<stage::Clip, stage::Standalone, Op>(op, Some(mode), operand)
    }
}

// ============================================================================
// Filter operations
// ============================================================================

impl<
    'l,
    const T: Tu,
    S,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, stage::Standalone, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Filter>,
{
    /// Filter by branch ID. Requires `Way8` mode and the `Standalone` context (not `Group` or `Zipped`).
    #[primitive(VectorTensor::vector_filter)]
    pub fn vector_filter<Time2: M>(
        self,
        _config: TagGuard,
    ) -> VectorFilterTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet, Stash, VE_ORDER, stage::Standalone, { Way8 }>
    {
        todo!("Implement vector_filter operation")
    }
}

// ============================================================================
// Reinterpret (no hardware, so no stage)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    Stash: StashState,
    FS: stage::VeTensorContext,
    const W: Way,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W>
{
    /// Rereads the stream's bits as `D2`, so `1.0f32` becomes `0x3f80_0000`. Emits no instruction and
    /// claims no ALU, so it holds the stage and `Way` and may sit anywhere in the chain.
    #[primitive(VectorTensor::vector_reinterpret)]
    pub fn vector_reinterpret<D2: VeScalar>(
        self,
    ) -> VectorTensor<'l, T, S, D2, Chip, Cluster, Slice, Time, Packet, Stash, VE_ORDER, FS, W> {
        let op_fn = HasConversionOp::<D, D2>::conversion_op_fn(&Reinterpret);
        let result = self.inner().map(&op_fn);
        let (ctx, _inner, tag, ve_state) = self.into_parts();
        VectorTensor::from_parts(ctx, result, tag, ve_state)
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_common_ir::{BitReq, BranchedOperand, TagGuard};

    use super::*;

    /// One branch per element, so a cell an earlier slot claimed is not combined again by a later one.
    /// `Max` makes the difference visible: applying both slots to cell 0 would give `max(0, 10, 50) = 50`
    /// where first match gives `max(0, 10) = 10`.
    ///
    /// Bit 2 set is "negative" in the tag mode the branch-unit example uses.
    #[test]
    fn an_overlapping_later_slot_does_not_recombine_a_claimed_cell() {
        axes![A = 4];
        let negative = TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::One, BitReq::Ignore]);
        let data = Tensor::<i32, m![A]>::from_vec([0, 0, 0, 0]);
        let tag = Tensor::<u8, m![A]>::from_vec([0b0100, 0b0000, 0b0100, 0b0000]);
        let register = Tensor::<i32, m![A]>::from_vec([50, 50, 50, 50]);
        let operand: BinaryBranchedOperand<i32, m![A]> = BranchedOperand::try_from_slots(
            [Some((negative, 10)), None, None],
            Some((TagGuard::all(), RfPort::External(register))),
        )
        .unwrap();

        let out = apply_binary_op(&data, &tag, |a, b| a.max(b), &operand, None);

        // Cells 0 and 2 match `negative` and stop at reg0; cells 1 and 3 fall through to the port.
        assert_eq!(out.into_vec(), vec![10, 50, 10, 50]);
    }

    /// A guarded unary touches the cells its guard claims and leaves the rest, which is the whole of
    /// what a guard means on a node with no slots to place it in.
    ///
    /// `vector_fp_unary` is this with `TagGuard::all`, so the unguarded form is covered by the same
    /// path rather than a second one.
    #[test]
    fn a_guarded_unary_leaves_unclaimed_cells_alone() {
        axes![A = 4];
        let negative = TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::One, BitReq::Ignore]);
        let data = Tensor::<i32, m![A]>::from_vec([1, 2, 3, 4]);
        let tag = Tensor::<u8, m![A]>::from_vec([0b0100, 0b0000, 0b0100, 0b0000]);

        let guarded = apply_unary_op_where(&data, &tag, negative, |v| v * 10);
        assert_eq!(guarded.into_vec(), vec![10, 2, 30, 4]);

        let everywhere = apply_unary_op_where(&data, &tag, TagGuard::all(), |v| v * 10);
        assert_eq!(everywhere.into_vec(), vec![10, 20, 30, 40]);
    }

    /// A cell no slot claims passes through untouched, which is what makes a guarded pass a partial
    /// update rather than a whole-tensor one.
    #[test]
    fn a_cell_no_slot_claims_passes_through() {
        axes![A = 2];
        let only_bit0 = TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
        let data = Tensor::<i32, m![A]>::from_vec([7, 7]);
        let tag = Tensor::<u8, m![A]>::from_vec([0b0001, 0b0000]);
        let operand: BinaryBranchedOperand<i32, m![A]> =
            BranchedOperand::try_from_slots([Some((only_bit0, 1)), None, None], None).unwrap();

        let out = apply_binary_op(&data, &tag, |a, b| a + b, &operand, None);

        assert_eq!(out.into_vec(), vec![8, 7]);
    }

    /// An empty layout leaves the stream alone entirely.
    #[test]
    fn an_empty_layout_leaves_the_stream_alone() {
        axes![A = 2];
        let data = Tensor::<i32, m![A]>::from_vec([3, 4]);
        let tag = Tensor::<u8, m![A]>::from_vec([0b1111, 0b0000]);
        let operand = BinaryBranchedOperand::<i32, m![A]>::default();

        let out = apply_binary_op(&data, &tag, |a, b| a + b, &operand, None);

        assert_eq!(out.into_vec(), vec![3, 4]);
    }

    /// Each ternary slot feeds its own `operand1`, which is why a ternary slot carries a pair at all:
    /// two groups in one pass can multiply by different constants.
    #[test]
    fn each_ternary_slot_feeds_its_own_operand1() {
        axes![A = 4];
        // Bit 3 alone is what a group id means to the hardware. Written out rather than through
        // `TagGuard::group` so the bit this rests on is visible in the test.
        let group0 = TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::Zero]);
        let group1 = TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::One]);
        let data = Tensor::<f32, m![A]>::from_vec([1.0, 1.0, 1.0, 1.0]);
        let tag = Tensor::<u8, m![A]>::from_vec([0b0000, 0b1000, 0b0000, 0b1000]);
        let operand: TernaryBranchedOperand<f32, m![A]> =
            BranchedOperand::try_from_slots([Some((group0, (2.0, 10.0))), Some((group1, (3.0, 20.0))), None], None)
                .unwrap();

        // fma: stream * operand0 + operand1, so each group shows both of its own values.
        let out = apply_ternary_op(&data, &tag, |a, b, c| a * b + c, &operand, None);

        assert_eq!(out.into_vec(), vec![12.0, 23.0, 12.0, 23.0]);
    }

    /// The port slot reads the stash tensor rather than the stream, and only where its guard matches.
    #[test]
    fn the_port_slot_reads_the_stash_tensor() {
        axes![A = 2];
        let bit0 = TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
        let data = Tensor::<i32, m![A]>::from_vec([1, 1]);
        let tag = Tensor::<u8, m![A]>::from_vec([0b0001, 0b0000]);
        let stash = Tensor::<i32, m![A]>::from_vec([40, 40]);
        let operand: BinaryBranchedOperand<i32, m![A]> =
            BranchedOperand::try_from_slots([None, None, None], Some((bit0, RfPort::Stash))).unwrap();

        let out = apply_binary_op(&data, &tag, |a, b| a + b, &operand, Some(&stash));

        assert_eq!(out.into_vec(), vec![41, 1], "the unguarded cell is untouched");
    }
}
