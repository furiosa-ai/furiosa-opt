//! VectorTensor definition and implementations.
//!
//! This module defines `VectorTensor<S: Stage>`, a unified tensor type for the Vector Engine pipeline.
//! The stage parameter `S` tracks pipeline progression at compile time, with type-safe
//! transitions between stages enforced through `CanTransitionTo`, `IntraSliceStage`, and `InterSliceStage`.
//!
//! # VE Entry / Exit
//! - Entry (on CollectTensor / AccumulationTensor):
//!   - `vector_init()` → initialized VE input
//!   - `vector_init()`, then `vector_intra_slice_branch(BranchMode)` → intra-slice (single stream)
//!   - `vector_init()`, then `vector_intra_slice_unzip(...)` → intra-slice (two-group)
//!   - `vector_init()`, then `vector_inter_slice_reduce(...)` → inter-slice reduce
//! - Exit: `vector_final()` → `VectorFinalTensor` → commit/cast/transpose
//!
//! # VeOrder (VE_ORDER)
//! Tracks which unit was entered first. Set once, never changes after.
//! - `IntraFirst` after `vector_intra_slice_branch` / `vector_intra_slice_unzip`
//! - `InterFirst` after `vector_inter_slice_reduce` from `VectorInitTensor`
//! - Preserved through all subsequent operations

use std::marker::PhantomData;

use furiosa_mapping::{Ident, *};
use furiosa_mapping_macro::primitive;
use furiosa_opt_macro::m;

use super::VeTensorShape;

use crate::array_vec::ArrayVec;
use crate::context::*;
use crate::prelude::ValidBranchIds;
use crate::scalar::Opt;
use crate::stream_tensor::VectorFinalTensor;
use crate::tensor::*;
use crate::vector_engine::MAX_BRANCHES;
use crate::vector_engine::branch::{BranchMode, apply_branch_config};
use crate::vector_engine::layer::{FpToFxp, FxpToFp};
use crate::vector_engine::op::semantics::{HasBinaryOp, HasTernaryOp, HasUnaryOp};
use crate::vector_engine::op::{
    BinaryArgMode, ClipBinaryOpF32, ClipBinaryOpI32, FpBinaryOp, FpDivOp, FpTernaryOp, FpUnaryOp, FxpBinaryOp, HasAlu,
    InterSliceReduceOpF32, InterSliceReduceOpI32, IntraSliceReduceOpF32, IntraSliceReduceOpI32, LogicBinaryOpF32,
    LogicBinaryOpI32, TernaryArgMode, UnaryArgMode,
};

use crate::tensor_state::{HasTensor, NoTensor, TensorState};
use crate::vector_engine::operand::{IntoOperands, IntoTernaryOperands, TernaryOperand, VeBranchOperand, VeRhs};
use crate::vector_engine::scalar::VeScalar;
use crate::vector_engine::stage::markers as stage;
use crate::vector_engine::stage::markers::CanTransitionTo;
use crate::vector_engine::stage::markers::PacketMode::{self, Way4, Way8};
use crate::vector_engine::stage::markers::VeOrder;
use crate::vector_engine::stage::state::VeState;
use crate::vector_engine::tensor::verify::{
    verify_vector_concat, verify_vector_pad_way8, verify_vector_split, verify_vector_trim_way4,
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

/// Common tensor data for VE pipeline stages, without context reference.
/// This expects sharing implementation between VectorTensor and VectorTensorPair groups.
///
/// The `S` type parameter represents the current pipeline stage.
/// The `FS` type parameter represents the filter state.
/// The `StashD` type parameter represents the scalar type of the stash tensor.
/// The `Stash` type parameter represents the stash type (for compile-time type checking).
/// The `W` type parameter represents the packet mode (Way8 or Way4).
#[derive(Debug)]
pub struct VeTensorData<
    S: stage::Stage,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    const VE_ORDER: VeOrder,
    FS: stage::VeTensorContext = stage::Standalone,
    const W: PacketMode = { Way8 },
> {
    pub(crate) inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    pub(crate) execution_id: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    pub(crate) ve_state: VeState<StashD, Stash>,
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
/// The `FS` type parameter represents the filter state:
/// - `Standalone`: Normal state, filter and stash operations are available
/// - `AfterBinary`: After binary operation, filter and stash are NOT available
///
/// The `W` type parameter represents the packet mode:
/// - `Way8`: Default 8-element flit mode. Float operations are NOT available.
/// - `Way4`: After `vector_split` or `vector_trim_way4`, front-4-only. Float operations are available.
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    const VE_ORDER: VeOrder,
    FS: stage::VeTensorContext = stage::Standalone,
    const W: PacketMode = { Way8 },
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) data: VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>,
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const W: PacketMode,
    const VE_ORDER: VeOrder,
> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>
{
    /// Returns a mutable reference to the VE state.
    pub fn ve_state_mut(&mut self) -> &mut VeState<StashD, Stash> {
        &mut self.ve_state
    }

    /// Returns a reference to the VE state.
    pub fn ve_state(&self) -> &VeState<StashD, Stash> {
        &self.ve_state
    }

    /// Returns a reference to the inner tensor.
    pub fn inner(&self) -> &Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        &self.inner
    }

    /// Returns a reference to the execution_id tensor.
    pub fn execution_id(&self) -> &Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        &self.execution_id
    }

    /// Consumes the data and returns its parts.
    pub fn into_parts(
        self,
    ) -> (
        Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VeState<StashD, Stash>,
    ) {
        (self.inner, self.execution_id, self.ve_state)
    }

    /// Internal helper for binary operations.
    /// Applies a binary operation with ALU tracking and stash support.
    pub(crate) fn apply_binary<NextStage: stage::Stage, NextFS: stage::VeTensorContext>(
        mut self,
        alu: crate::vector_engine::alu::RngdAlu,
        op_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
        operands: &ArrayVec<VeBranchOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>, MAX_BRANCHES>,
    ) -> VeTensorData<NextStage, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, NextFS, W> {
        // Only read stash if actually used by an operand
        let uses_stash = operands.iter().any(|op| matches!(op.operand, VeRhs::Stash));
        let stash_data: Option<Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>> = if uses_stash {
            self.ve_state.force_clone_stash_as()
        } else {
            None
        };
        self.ve_state.use_alu(alu);
        let result = apply_binary_op(
            &self.inner,
            &self.execution_id,
            op_fn,
            operands.as_slice(),
            stash_data.as_ref(),
        );
        VeTensorData {
            inner: result,
            execution_id: self.execution_id,
            ve_state: self.ve_state,
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const W: PacketMode,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>
{
    /// Consumes the tensor and returns its parts.
    pub fn into_parts(
        self,
    ) -> (
        &'l mut TuContext<{ T }>,
        Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VeState<StashD, Stash>,
    ) {
        let (inner, execution_id, ve_state) = self.data.into_parts();
        (self.ctx, inner, execution_id, ve_state)
    }

    /// Consumes the tensor and returns ctx and data separately.
    pub fn into_ctx_and_data(
        self,
    ) -> (
        &'l mut TuContext<{ T }>,
        VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>,
    ) {
        (self.ctx, self.data)
    }

    /// Returns a mutable reference to the VE state.
    pub fn ve_state_mut(&mut self) -> &mut VeState<StashD, Stash> {
        self.data.ve_state_mut()
    }

    /// Returns a reference to the VE state.
    pub fn ve_state(&self) -> &VeState<StashD, Stash> {
        self.data.ve_state()
    }

    /// Returns a reference to the inner tensor.
    pub fn inner(&self) -> &Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        self.data.inner()
    }

    /// Returns a reference to the execution_id tensor.
    pub fn execution_id(&self) -> &Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        self.data.execution_id()
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W> {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn data_mut(
        &mut self,
    ) -> &mut VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W> {
        &mut self.data
    }

    /// Creates a new VectorTensor from parts.
    pub fn from_parts(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        execution_id: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        ve_state: VeState<StashD, Stash>,
    ) -> Self {
        VectorTensor {
            ctx,
            data: VeTensorData {
                inner,
                execution_id,
                ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }

    /// Creates a new VectorTensor from context and data.
    pub fn from_ctx_and_data(
        ctx: &'l mut TuContext<{ T }>,
        data: VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>,
    ) -> Self {
        VectorTensor { ctx, data }
    }

    /// Internal helper for binary operations.
    /// Applies operation with ALU tracking and stash support, returns new VectorTensor.
    /// If mode is None, uses the default mode (Mode01).
    pub(crate) fn do_binary<NextStage: stage::Stage, NextFS: stage::VeTensorContext>(
        self,
        op: impl HasAlu + HasBinaryOp<D>,
        mode: Option<BinaryArgMode>,
        operands: ArrayVec<VeBranchOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>, MAX_BRANCHES>,
    ) -> VectorTensor<'l, T, NextStage, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, NextFS, W> {
        let data = self.data.apply_binary(op.alu(), op.binary_op_fn(mode), &operands);
        VectorTensor { ctx: self.ctx, data }
    }
}

// Separate impl for stash on VeTensorData - only stages implementing Stashable can use this
impl<
    S: stage::Stashable,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    const W: PacketMode,
    const VE_ORDER: VeOrder,
> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, D, NoTensor, VE_ORDER, stage::Standalone, W>
{
    /// Writes the current tensor data to the operand register.
    /// The data can later be read using VeRhs::Stash in binary operations.
    /// Returns a new VeTensorData with the stash's mapping set to the current tensor's mapping.
    pub fn stash(
        self,
    ) -> VeTensorData<
        S,
        D,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        D,
        HasTensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VE_ORDER,
        stage::Standalone,
        W,
    > {
        let new_ve_state = self.ve_state.stash(&self.inner);
        VeTensorData {
            inner: self.inner,
            execution_id: self.execution_id,
            ve_state: new_ve_state,
            _stage: PhantomData,
            _filter_state: PhantomData,
        }
    }
}

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
    const W: PacketMode,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, D, NoTensor, VE_ORDER, stage::Standalone, W>
{
    /// Writes the current tensor data to the operand register.
    /// The data can later be read using VeRhs::Stash in binary operations.
    ///
    /// Only available for stages that support stash operation:
    /// Branch, Logic, Fxp, Narrow, Fp, FpDiv, Clip
    ///
    /// NOT available after binary operations (AfterBinary state).
    /// Returns a new VectorTensor with the stash's mapping set to the current tensor's mapping.
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
        D,
        HasTensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VE_ORDER,
        stage::Standalone,
        W,
    > {
        let new_ve_state = self.data.ve_state.stash(&self.data.inner);
        VectorTensor {
            ctx: self.ctx,
            data: VeTensorData {
                inner: self.data.inner,
                execution_id: self.data.execution_id,
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::Commitable,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::Commitable,
>
    VectorTensor<
        'l,
        T,
        S,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        StashD,
        Stash,
        { VeOrder::IntraFirst },
        FS,
        { Way8 },
    >
{
    /// Performs inter-slice reduce for i32 from intra-slice stages.
    /// Only available when VeOrder::IntraFirst (intra-slice was entered first).
    #[primitive(VectorTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<Slice2: M, Time2: M>(
        self,
        op: InterSliceReduceOpI32,
    ) -> VectorInterSliceReduceTensor<'l, T, i32, Chip, Cluster, Slice2, Time2, Packet, { VeOrder::IntraFirst }> {
        let reduced = self
            .data
            .inner
            .reduce_then_broadcast_with(op.lifted_reduce_fn(), Opt::Uninit);
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::Commitable,
>
    VectorTensor<
        'l,
        T,
        S,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        StashD,
        Stash,
        { VeOrder::IntraFirst },
        FS,
        { Way8 },
    >
{
    /// Performs inter-slice reduce for f32 from intra-slice stages.
    /// Only available when VeOrder::IntraFirst (intra-slice was entered first).
    #[primitive(VectorTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<Slice2: M, Time2: M>(
        self,
        op: InterSliceReduceOpF32,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, Slice2, Time2, Packet, { VeOrder::IntraFirst }> {
        let reduced = self
            .data
            .inner
            .reduce_then_broadcast_with(op.lifted_reduce_fn(), Opt::Uninit);
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
            execution_id: Tensor::uninit(),
            ve_state: VeState::new(),
            _stage: PhantomData,
            _filter_state: PhantomData,
        },
    }
}

// ============================================================================
// VectorInitTensor methods
// ============================================================================

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Enters VE intra-slice pipeline (single stream).
    // ANCHOR: vector_intra_slice_branch
    #[primitive(VectorInitTensor::vector_intra_slice_branch)]
    pub fn vector_intra_slice_branch(
        self,
        branch: BranchMode,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, NoTensor, { VeOrder::IntraFirst }> {
        // ANCHOR_END: vector_intra_slice_branch
        VectorBranchTensor::new(self.ctx, self.inner, branch)
    }

    /// Enters VE intra-slice pipeline (two-group / unzip).
    // ANCHOR: vector_intra_slice_unzip
    #[primitive(VectorInitTensor::vector_intra_slice_unzip)]
    pub fn vector_intra_slice_unzip<I: AxisName, TileTime: M, SplitTime: M>(
        self,
    ) -> VectorTensorPair<'l, T, D, stage::Branch, Chip, Cluster, Slice, SplitTime, Packet> {
        // ANCHOR_END: vector_intra_slice_unzip
        VectorTensorPair::new::<I, Time, TileTime>(self.ctx, self.inner)
    }
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for i32 as the first VE operation.
    // ANCHOR: init_inter_slice_reduce_i32
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<Slice2: M, Time2: M>(
        self,
        op: InterSliceReduceOpI32,
    ) -> VectorInterSliceReduceTensor<'l, T, i32, Chip, Cluster, Slice2, Time2, Packet, { VeOrder::InterFirst }> {
        // ANCHOR_END: init_inter_slice_reduce_i32
        let reduced = self
            .inner
            .reduce_then_broadcast_with(op.lifted_reduce_fn(), Opt::Uninit);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for f32 as the first VE operation.
    // ANCHOR: init_inter_slice_reduce_f32
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<Slice2: M, Time2: M>(
        self,
        op: InterSliceReduceOpF32,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, Slice2, Time2, Packet, { VeOrder::InterFirst }> {
        // ANCHOR_END: init_inter_slice_reduce_f32
        let reduced = self
            .inner
            .reduce_then_broadcast_with(op.lifted_reduce_fn(), Opt::Uninit);
        create_inter_slice_reduce_tensor(self.ctx, reduced)
    }
}

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
        D,
        NoTensor,
        VE_ORDER,
        stage::Standalone,
        { Way8 },
    >;

// ============================================================================
// Inter→Intra entry — vector_intra_slice_branch on InterSliceStage (requires VeOrder::InterFirst)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::InterSliceStage + stage::CanTransitionTo<stage::Branch>,
    D: VeScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::Commitable,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, { VeOrder::InterFirst }, FS, { Way8 }>
{
    /// Enters intra-slice pipeline from inter-slice output. Requires VeOrder::InterFirst.
    /// Preserves VeOrder::InterFirst.
    #[primitive(VectorTensor::vector_intra_slice_branch)]
    pub fn vector_intra_slice_branch(
        self,
        branch: BranchMode,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, NoTensor, { VeOrder::InterFirst }> {
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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::Branch, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, const VE_ORDER: VeOrder>
    VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, NoTensor, VE_ORDER>
{
    /// Creates a new VectorBranchTensor from inner tensor and branch configuration.
    pub fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        branch_config: BranchMode,
    ) -> Self {
        assert_eq!(
            Packet::SIZE,
            8,
            "VectorTensor requires Packet of 8 elements (one flit) in Way8 mode, got {}",
            Packet::SIZE,
        );
        let execution_id = apply_branch_config(&inner, &branch_config);
        Self::from_parts(ctx, inner, execution_id, VeState::new())
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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::Logic, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::Fxp, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::FxpToFp, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

/// Tensor after narrow layer (split or trim).
pub type VectorNarrowTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way4 },
> = VectorTensor<'l, T, stage::Narrow, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way4 },
> = VectorTensor<'l, T, stage::Fp, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way4 },
> = VectorTensor<'l, T, stage::IntraSliceReduce, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way4 },
> = VectorTensor<'l, T, stage::FpDiv, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::Widen, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::FpToFxp, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::Clip, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: PacketMode = { Way8 },
> = VectorTensor<'l, T, stage::Filter, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

// ============================================================================
// Helper functions for applying operations with execution_id
// ============================================================================

/// Collects RHS values into a Vec for efficient iteration.
/// Returns None if the operand type doesn't have data (shouldn't happen in valid usage).
fn get_rhs_values<D: VeScalar, Mapping: M>(
    rhs: &VeRhs<D, Mapping>,
    stash_data: Option<&Tensor<D, Mapping>>,
    len: usize,
) -> Option<Vec<Opt<D>>> {
    match rhs {
        VeRhs::Const { v } => Some(vec![Opt::Init(*v); len]),
        VeRhs::Stash => stash_data.map(|t| t.data().iter().copied().collect()),
        VeRhs::Vrf { data } => {
            if data.data().is_empty() {
                None
            } else {
                Some(data.data().iter().cycle().take(len).copied().collect())
            }
        }
    }
}

/// Applies a binary operation to a tensor with branch-conditional execution.
/// Stash data is passed as a Tensor (already transposed to match current mapping).
pub(super) fn apply_binary_op<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    execution_id: &Tensor<u8, Mapping>,
    op: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
    operands: &[VeBranchOperand<D, Mapping>],
    stash_data: Option<&Tensor<D, Mapping>>,
) -> Tensor<D, Mapping> {
    let mut result = data.clone();
    let len = data.data().len();

    for operand in operands {
        if let Some(rhs_values) = get_rhs_values(&operand.operand, stash_data, len) {
            for ((data_val, eid), rhs_val) in result
                .data_mut()
                .iter_mut()
                .zip(execution_id.data().iter())
                .zip(rhs_values.iter())
            {
                if operand.valid_branch_ids.matches(*eid) {
                    *data_val = op(*data_val, *rhs_val);
                }
            }
        }
    }
    result
}

/// Applies a unary operation to a tensor with branch-conditional execution.
pub(super) fn apply_unary_op<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    execution_id: &Tensor<u8, Mapping>,
    op: impl Fn(Opt<D>) -> Opt<D>,
    operands: &[VeBranchOperand<D, Mapping>],
) -> Tensor<D, Mapping> {
    let mut result = data.clone();
    // Unary ops don't use operand values, but still respect branch conditions
    if operands.is_empty() {
        // No branch filtering, apply to all
        for data_val in result.data_mut().iter_mut() {
            *data_val = op(*data_val);
        }
    } else {
        for operand in operands {
            for (data_val, eid) in result.data_mut().iter_mut().zip(execution_id.data().iter()) {
                if operand.valid_branch_ids.matches(*eid) {
                    *data_val = op(*data_val);
                }
            }
        }
    }
    result
}

/// Applies a ternary operation to a tensor with branch-conditional execution.
/// Ternary operations are only supported for f32 tensors.
pub(super) fn apply_ternary_op<Mapping: M>(
    data: &Tensor<f32, Mapping>,
    execution_id: &Tensor<u8, Mapping>,
    op: impl Fn(Opt<f32>, Opt<f32>, Opt<f32>) -> Opt<f32>,
    operands: &[TernaryOperand<Mapping>],
    stash_data: Option<&Tensor<f32, Mapping>>,
) -> Tensor<f32, Mapping> {
    let mut result = data.clone();
    let len = data.data().len();

    for operand in operands {
        let rhs1_val = Opt::Init(operand.operand1);

        if let Some(rhs0_values) = get_rhs_values(&operand.operand0, stash_data, len) {
            for ((data_val, eid), rhs0_val) in result
                .data_mut()
                .iter_mut()
                .zip(execution_id.data().iter())
                .zip(rhs0_values.iter())
            {
                if operand.valid_branch_ids.matches(*eid) {
                    *data_val = op(*data_val, *rhs0_val, rhs1_val);
                }
            }
        }
    }
    result
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Logic>,
{
    /// Logic binary operation (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic)]
    pub fn vector_logic(
        self,
        op: LogicBinaryOpI32,
        operand: impl IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorLogicTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Logic binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic_with_mode)]
    pub fn vector_logic_with_mode(
        self,
        op: LogicBinaryOpI32,
        mode: BinaryArgMode,
        operand: impl IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorLogicTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, Some(mode), operand.into_operands())
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Logic>,
{
    /// Logic binary operation (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic)]
    pub fn vector_logic(
        self,
        op: LogicBinaryOpF32,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorLogicTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Logic binary operation with explicit mode (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic_with_mode)]
    pub fn vector_logic_with_mode(
        self,
        op: LogicBinaryOpF32,
        mode: BinaryArgMode,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorLogicTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, Some(mode), operand.into_operands())
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Fxp>,
{
    /// Fixed-point binary operation (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fxp)]
    pub fn vector_fxp(
        self,
        op: FxpBinaryOp,
        operand: impl IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Fixed-point binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fxp_with_mode)]
    pub fn vector_fxp_with_mode(
        self,
        op: FxpBinaryOp,
        mode: BinaryArgMode,
        operand: impl IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, Some(mode), operand.into_operands())
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::FxpToFp>,
{
    /// Converts i32 to f32. Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fxp_to_fp)]
    pub fn vector_fxp_to_fp(
        self,
        int_width: u32,
    ) -> VectorFxpToFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        let op = FxpToFp::new(int_width);
        let op_fn = op.op_fn();

        let result = self.inner().map(|v| v.map(&op_fn));

        let (ctx, _inner, execution_id, ve_state) = self.into_parts();
        VectorFxpToFpTensor::from_parts(ctx, result, execution_id, ve_state)
    }
}

// ============================================================================
// Narrow operations (split / trim)
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Narrow>,
{
    /// Narrow layer (split). Requires `Way8` mode.
    ///
    /// Takes an 8-element packet, splits it into front 4 + back 4.
    /// The factor of 2 goes into Time, and the output is `Way4` with 4-element packets.
    /// Output: `Time2 = Time × 2`, `Packet2` = front 4 of Packet (size 4).
    #[primitive(VectorTensor::vector_split)]
    pub fn vector_split<Time2: M, Packet2: M>(
        self,
    ) -> VectorNarrowTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, StashD, Stash, VE_ORDER, FS, { Way4 }> {
        verify_vector_split::<Time, Packet, Time2, Packet2>();

        let (ctx, inner, execution_id, ve_state) = self.into_parts();

        let split_inner = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);
        let split_eid = execution_id.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);

        VectorNarrowTensor::from_parts(ctx, split_inner, split_eid, ve_state)
    }
}

// ============================================================================
// vector_trim_way4: strip back-4 dummy from Packet 8 → 4 (type-only, no-op at hardware level)
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Narrow>,
{
    /// Strip the back-4 dummy lanes from an 8-element packet, yielding a 4-element packet.
    /// Transitions from `Way8` to `Way4` mode and enters the `Narrow` stage.
    ///
    /// This is a type-system-only operation — no hardware instruction is emitted.
    /// Use this when the back 4 lanes are already padding (≤ 4 real elements).
    /// For packets with > 4 real elements, use `vector_split()` instead.
    #[primitive(VectorTensor::vector_trim_way4)]
    pub fn vector_trim_way4<Packet2: M>(
        self,
    ) -> VectorNarrowTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet2, StashD, Stash, VE_ORDER, FS, { Way4 }> {
        verify_vector_trim_way4::<Packet, Packet2>();

        let (ctx, inner, execution_id, ve_state) = self.into_parts();

        let stripped = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);
        let stripped_eid = execution_id.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);

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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Fp>,
{
    /// Fp unary operation (f32 only).
    #[primitive(VectorTensor::vector_fp_unary)]
    pub fn vector_fp_unary(
        self,
        op: FpUnaryOp,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.vector_fp_unary_with_mode(op, UnaryArgMode::Mode0)
    }

    /// Fp unary operation with explicit mode (f32 only).
    #[primitive(VectorTensor::vector_fp_unary_with_mode)]
    pub fn vector_fp_unary_with_mode(
        mut self,
        op: FpUnaryOp,
        mode: UnaryArgMode,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.ve_state_mut().use_alu(op.alu());
        let op_fn = op.unary_op_fn(Some(mode));
        let result = apply_unary_op(self.inner(), self.execution_id(), op_fn, &[]);
        let (ctx, _inner, execution_id, ve_state) = self.into_parts();
        VectorFpTensor::from_parts(ctx, result, execution_id, ve_state)
    }

    /// Fp binary operation (f32 only).
    #[primitive(VectorTensor::vector_fp_binary)]
    pub fn vector_fp_binary(
        self,
        op: FpBinaryOp,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Fp binary operation with explicit mode (f32 only).
    #[primitive(VectorTensor::vector_fp_binary_with_mode)]
    pub fn vector_fp_binary_with_mode(
        self,
        op: FpBinaryOp,
        mode: BinaryArgMode,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, Some(mode), operand.into_operands())
    }

    /// Fp ternary operation (f32 only).
    ///
    /// # Example
    /// ```ignore
    /// // FmaF: result = data * operand0 + operand1
    /// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 3.0f32))
    ///
    /// // With VRF as operand0
    /// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (&vrf, 3.0f32))
    ///
    /// // With stash as operand0
    /// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (Stash, 3.0f32))
    /// ```
    #[primitive(VectorTensor::vector_fp_ternary)]
    pub fn vector_fp_ternary(
        self,
        op: FpTernaryOp,
        operands: impl IntoTernaryOperands<VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.vector_fp_ternary_with_mode(op, TernaryArgMode::Mode012, operands)
    }

    /// Fp ternary operation with explicit mode (f32 only).
    #[primitive(VectorTensor::vector_fp_ternary_with_mode)]
    pub fn vector_fp_ternary_with_mode(
        mut self,
        op: FpTernaryOp,
        mode: TernaryArgMode,
        operands: impl IntoTernaryOperands<VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        let operands = operands.into_ternary_operands();
        // TODO: we should only read stash if actually used by an operand, just like apply_binary
        let stash_data: Option<Tensor<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>> =
            self.ve_state().force_clone_stash_as();
        self.ve_state_mut().use_alu(op.alu());
        let op_fn = op.ternary_op_fn(Some(mode));
        let result = apply_ternary_op(
            self.inner(),
            self.execution_id(),
            op_fn,
            operands.as_slice(),
            stash_data.as_ref(),
        );
        let (ctx, _inner, execution_id, ve_state) = self.into_parts();
        VectorFpTensor::from_parts(ctx, result, execution_id, ve_state)
    }
}

// ============================================================================
// Intra-slice reduce operations
// ============================================================================

/// Verifies that all reduced axes (quotient of input / output shape) match the expected ident.
fn verify_reduce_label<Time: M, Packet: M, OTime: M, OPacket: M>(reduce_label: &Ident) {
    let d = <m![{ Time }, { Packet }]>::to_value()
        .divide_relaxed(&<m![{ OTime }, { OPacket }]>::to_value())
        .exact()
        .expect("[Intra-slice reduce] divide failed: output shape must divide input shape");
    let quotient = d.dividend_residue;
    let division = d.division_terms;
    assert!(
        quotient.idents().iter().all(|ident| ident == reduce_label),
        "IntraSliceReduce: all reduced axes must match the specified reduce_label {}, got quotient {} with idents {:?}",
        reduce_label,
        quotient,
        quotient.idents()
    );

    assert!(
        division
            .iter()
            .all(|d| d.term.idents().iter().all(|ident| ident != reduce_label)),
        "IntraSliceReduce: all the reduce axes should be fully reduced (not present in the division terms), got reduce_label {} appearing in division {:?}",
        reduce_label,
        division,
    );

    assert!(
        Packet::to_value().factorize() == OPacket::to_value().factorize()
            || OPacket::to_value().factorize() == <m![1 # 4]>::to_value().factorize(),
        "IntraSliceReduce: Packet should be either preserved or reduced to 4 (for partial reduction), got Packet {} → OPacket {}",
        Packet::to_value().factorize(),
        OPacket::to_value().factorize()
    );
}

/// Reduces execution_id tensor by keeping the last value (hardware semantics: all reduced
/// elements share the same execution_id).
fn reduce_execution_id<Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, OTime: M, OPacket: M>(
    execution_id: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
) -> Tensor<u8, VeTensorShape<Chip, Cluster, Slice, OTime, OPacket>> {
    execution_id.reduce::<VeTensorShape<Chip, Cluster, Slice, OTime, OPacket>>(|_, y| y, Opt::Uninit)
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::IntraSliceReduce>,
{
    /// Intra-slice reduce operation (i32).
    #[primitive(VectorTensor::vector_intra_slice_reduce)]
    pub fn vector_intra_slice_reduce<Reduce: AxisName, OTime: M, OPacket: M>(
        mut self,
        op: IntraSliceReduceOpI32,
    ) -> VectorIntraSliceReduceTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        OTime,
        OPacket,
        StashD,
        Stash,
        VE_ORDER,
        stage::Standalone,
        { Way4 },
    >
// ANCHOR_END: intra_slice_reduce_i32
    {
        self.ve_state_mut().use_alu(op.alu());
        let (ctx, inner, execution_id, ve_state) = self.into_parts();
        verify_reduce_label::<Time, Packet, OTime, OPacket>(&Reduce::NAME);
        let reduced_inner =
            inner.reduce::<VeTensorShape<Chip, Cluster, Slice, OTime, OPacket>>(op.lifted_reduce_fn(), Opt::Uninit);
        let reduced_eid = reduce_execution_id::<Chip, Cluster, Slice, Time, Packet, OTime, OPacket>(execution_id);
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::IntraSliceReduce>,
{
    /// Intra-slice reduce operation (f32).
    #[primitive(VectorTensor::vector_intra_slice_reduce)]
    pub fn vector_intra_slice_reduce<Reduce: AxisName, OTime: M, OPacket: M>(
        mut self,
        op: IntraSliceReduceOpF32,
    ) -> VectorIntraSliceReduceTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        OTime,
        OPacket,
        StashD,
        Stash,
        VE_ORDER,
        stage::Standalone,
        { Way4 },
    >
// ANCHOR_END: intra_slice_reduce_f32
    {
        self.ve_state_mut().use_alu(op.alu());
        let (ctx, inner, execution_id, ve_state) = self.into_parts();
        verify_reduce_label::<Time, Packet, OTime, OPacket>(&Reduce::NAME);
        let reduced_inner =
            inner.reduce::<VeTensorShape<Chip, Cluster, Slice, OTime, OPacket>>(op.lifted_reduce_fn(), Opt::Uninit);
        let reduced_eid = reduce_execution_id::<Chip, Cluster, Slice, Time, Packet, OTime, OPacket>(execution_id);
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::FpDiv>,
{
    /// Floating-point division.
    #[primitive(VectorTensor::vector_fp_div)]
    pub fn vector_fp_div(
        self,
        op: FpDivOp,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFpDivTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Floating-point division with explicit mode.
    #[primitive(VectorTensor::vector_fp_div_with_mode)]
    pub fn vector_fp_div_with_mode(
        self,
        op: crate::vector_engine::op::FpDivBinaryOp,
        mode: BinaryArgMode,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorFpDivTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }> {
        self.do_binary(op, Some(mode), operand.into_operands())
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Widen>,
{
    /// Widen layer (concat). Requires `Way4` mode.
    ///
    /// Reverse of split. Takes 4-element packets from 2 consecutive time steps,
    /// merges them into one 8-element packet and transitions to `Way8`.
    /// `Time2 = Time / 2`, `Packet2` = Packet combined with factor of 2 from Time.
    #[primitive(VectorTensor::vector_concat)]
    pub fn vector_concat<Time2: M, Packet2: M>(
        self,
    ) -> VectorWidenTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, StashD, Stash, VE_ORDER, FS, { Way8 }> {
        verify_vector_concat::<Time, Packet, Time2, Packet2>();

        let (ctx, inner, execution_id, ve_state) = self.into_parts();

        let concat_inner = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);
        let concat_eid = execution_id.transpose::<VeTensorShape<Chip, Cluster, Slice, Time2, Packet2>>(true);

        VectorWidenTensor::from_parts(ctx, concat_inner, concat_eid, ve_state)
    }
}

// ============================================================================
// vector_pad_way8: pad Packet 4 → 8 with dummy (type-only, no-op at hardware level)
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Widen>,
{
    /// Pad a 4-element packet back to 8 by adding dummy lanes.
    /// Transitions from `Way4` to `Way8` mode and enters the `Widen` stage.
    ///
    /// This is a type-system-only operation — no hardware instruction is emitted.
    /// Reverse of `vector_trim_way4`. Use this when no time-dimension merging is needed.
    /// For merging split time steps back, use `vector_concat()` instead.
    #[primitive(VectorTensor::vector_pad_way8)]
    pub fn vector_pad_way8<Packet2: M>(
        self,
    ) -> VectorWidenTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet2, StashD, Stash, VE_ORDER, FS, { Way8 }> {
        verify_vector_pad_way8::<Packet, Packet2>();

        let (ctx, inner, execution_id, ve_state) = self.into_parts();

        let padded = inner.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);
        let padded_eid = execution_id.transpose::<VeTensorShape<Chip, Cluster, Slice, Time, Packet2>>(true);

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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::FpToFxp>,
{
    /// Converts f32 to i32. Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fp_to_fxp)]
    pub fn vector_fp_to_fxp(
        self,
        int_width: u32,
    ) -> VectorFpToFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        let op = FpToFxp::new(int_width);
        let op_fn = op.op_fn();
        let result = self.inner().map(|&v| v.map(&op_fn));

        let (ctx, _inner, execution_id, ve_state) = self.into_parts();
        VectorFpToFxpTensor::from_parts(ctx, result, execution_id, ve_state)
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Clip>,
{
    /// Clip binary operation (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip)]
    pub fn vector_clip(
        self,
        op: ClipBinaryOpI32,
        operand: impl IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorClipTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Clip binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip_with_mode)]
    pub fn vector_clip_with_mode(
        self,
        op: ClipBinaryOpI32,
        mode: BinaryArgMode,
        operand: impl IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorClipTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, Some(mode), operand.into_operands())
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Clip>,
{
    /// Clip binary operation (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip)]
    pub fn vector_clip(
        self,
        op: ClipBinaryOpF32,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorClipTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, None, operand.into_operands())
    }

    /// Clip binary operation with explicit mode (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip_with_mode)]
    pub fn vector_clip_with_mode(
        self,
        op: ClipBinaryOpF32,
        mode: BinaryArgMode,
        operand: impl IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> VectorClipTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.do_binary(op, Some(mode), operand.into_operands())
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
    StashD: VeScalar,
    Stash: TensorState<StashD>,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, stage::Standalone, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Filter>,
{
    /// Filter by branch ID. Requires `Way8` mode.
    /// NOT available after binary operations (AfterBinary state).
    #[primitive(VectorTensor::vector_filter)]
    pub fn vector_filter<Time2: M>(
        self,
        _config: ValidBranchIds,
    ) -> VectorFilterTensor<
        'l,
        T,
        D,
        Chip,
        Cluster,
        Slice,
        Time2,
        Packet,
        StashD,
        Stash,
        VE_ORDER,
        stage::Standalone,
        { Way8 },
    > {
        todo!("Implement vector_filter operation")
    }
}
