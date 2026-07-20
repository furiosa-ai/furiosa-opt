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
use crate::array_vec::ArrayVec;
use crate::backend::Backend;
use crate::context::*;
use crate::engine::vector::MAX_TAGS;
use crate::engine::vector::alu::RngdAlu;
use crate::engine::vector::branch::{TagMode, apply_branch_config};
use crate::engine::vector::layer::{FpToFxp, FxpToFp};
use crate::engine::vector::op::semantics::{HasBinaryOp, HasTernaryOp, HasUnaryOp};
use crate::engine::vector::op::{
    BinaryArgMode, ClipBinaryOpF32, ClipBinaryOpI32, FpBinaryOp, FpDivBinaryOp, FpTernaryOp, FpUnaryOp, FxpBinaryOp,
    HasAlu, InterSliceReduceOpF32, InterSliceReduceOpI32, IntraSliceReduceOpF32, IntraSliceReduceOpI32,
    LogicBinaryOpF32, LogicBinaryOpI32, TernaryArgMode,
};
use crate::engine::vector::operand::OperandTag;
use crate::prelude::TagFilter;
use crate::runtime::CurrentBackend;
use crate::scalar::Opt;
use crate::tensor::*;

use crate::engine::vector::operand::{
    BinaryOperandTag, IntoOperands, IntoTernaryOperandTags, StashTransition, TernaryOperandTag, VeRhs,
};
use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::stage::markers as stage;
use crate::engine::vector::stage::markers::CanTransitionTo;
use crate::engine::vector::stage::markers::VeOrder;
use crate::engine::vector::stage::markers::Way::{self, Way4, Way8};
use crate::engine::vector::stage::state::VeState;
use crate::engine::vector::stash_slot::{Fresh, Occupied, StashSlot};
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    const VE_ORDER: VeOrder,
    FS: stage::VeTensorContext = stage::Standalone,
    const W: Way = { Way8 },
> {
    pub(crate) inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    pub(crate) tag: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    const VE_ORDER: VeOrder,
    FS: stage::VeTensorContext = stage::Standalone,
    const W: Way = { Way8 },
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
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const W: Way,
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
        VeState<StashD, Stash>,
    ) {
        (self.inner, self.tag, self.ve_state)
    }

    /// Internal helper for binary operations.
    /// Applies a binary operation with ALU tracking and stash support.
    pub(crate) fn apply_binary<NextStage: stage::Stage, NextFS: stage::VeTensorContext>(
        mut self,
        alu: RngdAlu,
        op_fn: impl Fn(D, D) -> D + Sync,
        operands: &ArrayVec<BinaryOperandTag<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>, MAX_TAGS>,
    ) -> VeTensorData<NextStage, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, NextFS, W> {
        // Only read stash if actually used by an operand
        let uses_stash = operands.iter().any(|op| matches!(op.operand0, VeRhs::Stash));
        let stash_data: Option<Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>> = if uses_stash {
            self.ve_state.force_clone_stash_as()
        } else {
            None
        };
        self.ve_state.use_alu(alu);
        let result = apply_binary_op(&self.inner, &self.tag, op_fn, operands.as_slice(), stash_data.as_ref());
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const W: Way,
    const VE_ORDER: VeOrder,
> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>
{
    /// Re-tags the stash typestate via the operand's [`StashTransition`], after the op already
    /// ran through the ordinary binary/ternary path (so vISA lowering is unchanged).
    pub(crate) fn apply_stash_transition<Op: StashTransition<StashD, Stash>>(
        self,
    ) -> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER, FS, W> {
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const W: Way,
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
        let (inner, tag, ve_state) = self.data.into_parts();
        (self.ctx, inner, tag, ve_state)
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

    /// Returns a reference to the tag tensor.
    pub fn tag(&self) -> &Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> {
        self.data.tag()
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
        tag: Tensor<u8, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        ve_state: VeState<StashD, Stash>,
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
        data: VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>,
    ) -> Self {
        Self { ctx, data }
    }

    /// Internal helper for binary operations.
    /// Applies operation with ALU tracking and stash support, returns new VectorTensor. Mode defaulting
    /// (`None` → `Mode01`) happens in `HasBinaryOp::binary_op_fn`.
    pub(crate) fn do_binary<NextStage: stage::Stage, NextFS: stage::VeTensorContext>(
        self,
        op: impl HasAlu + HasBinaryOp<D>,
        mode: Option<BinaryArgMode>,
        operands: ArrayVec<BinaryOperandTag<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>, MAX_TAGS>,
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
    const W: Way,
    const VE_ORDER: VeOrder,
> VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, VE_ORDER, stage::Standalone, W>
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
        Occupied<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VE_ORDER,
        stage::Standalone,
        W,
    > {
        let new_ve_state = self.ve_state.stash(&self.inner);
        VeTensorData {
            inner: self.inner,
            tag: self.tag,
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
    const W: Way,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, VE_ORDER, stage::Standalone, W>
{
    /// Writes the current tensor data to the operand register (the RNGD "stash").
    ///
    /// Write-once: this op is defined only on the [`Fresh`](crate::engine::vector::stash_slot::Fresh)
    /// state and flips it to [`Occupied`](crate::engine::vector::stash_slot::Occupied), so a second
    /// `vector_stash` (whether after a read, or back-to-back with no read) has no impl and does not
    /// compile. Read-once is enforced by the [`Stash`](crate::engine::vector::operand::Stash)
    /// operand, which consumes `Occupied` into
    /// [`Spent`](crate::engine::vector::stash_slot::Spent), not back to `Fresh`, so a read never
    /// re-arms the write. See the stash-slot [module docs] for the full state machine. Only on
    /// `Stashable` stages (Tag, Logic, Fxp, Narrow, Fp, FpDiv, Clip) in the `Standalone` context.
    ///
    /// Both HW-illegal double writes are compile errors (the second `vector_stash` finds no impl):
    ///
    /// ```text
    /// t.vector_stash().vector_fp_binary(op, Stash).vector_stash();  // read then re-stash: Spent has no vector_stash
    /// t.vector_stash().vector_stash();                              // no read between: Occupied has no vector_stash
    /// ```
    ///
    /// The read-once half is locked by the `compile_fail` doctests on
    /// [`StashTransition`](crate::engine::vector::operand::StashTransition); the write-once half by
    /// the `ve_elementwise_stash_*` example kernels (a real double `vector_stash` there is `E0599`).
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
        D,
        Occupied<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        VE_ORDER,
        stage::Standalone,
        W,
    > {
        let new_ve_state = self.data.ve_state.stash(&self.data.inner);
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
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
    Stash: StashSlot<StashD>,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
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
            tag: Tensor::uninit(),
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
        branch: TagMode,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, { VeOrder::IntraFirst }> {
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
        D,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::Commitable,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, { VeOrder::InterFirst }, FS, { Way8 }>
{
    /// Enters intra-slice pipeline from inter-slice output. Requires VeOrder::InterFirst.
    /// Preserves VeOrder::InterFirst.
    #[primitive(VectorTensor::vector_intra_slice_tag)]
    pub fn vector_intra_slice_tag(
        self,
        branch: TagMode,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, { VeOrder::InterFirst }> {
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
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Tag, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, const VE_ORDER: VeOrder>
    VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, D, Fresh, VE_ORDER>
{
    /// Creates a new VectorBranchTensor from inner tensor and branch configuration.
    pub fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        branch_config: TagMode,
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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way8 },
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
    const W: Way = { Way8 },
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
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::FxpToFp, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

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
    StashD,
    Stash,
    const VE_ORDER: VeOrder,
    FS = stage::Standalone,
    const W: Way = { Way4 },
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
    const W: Way = { Way4 },
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
    const W: Way = { Way4 },
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
    const W: Way = { Way4 },
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
    const W: Way = { Way8 },
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
    const W: Way = { Way8 },
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
    const W: Way = { Way8 },
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
    const W: Way = { Way8 },
> = VectorTensor<'l, T, stage::Filter, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, W>;

// ============================================================================
// Helper functions for applying operations with tag
// ============================================================================

/// Resolves `operand0`'s rhs tensor and fuses one branch-conditional combine into a single
/// [`Backend::zip3_with`] over `(out, rhs, tag)`. `Vrf`/`Stash` borrow the operand's existing tensor
/// (no copy); only `Const` materializes a broadcast tensor.
fn blend_operand<D: VeScalar, Mapping: M>(
    out: &<CurrentBackend as Backend>::Storage<D>,
    operand0: &VeRhs<D, Mapping>,
    tag: &Tensor<u8, Mapping>,
    stash_data: Option<&Tensor<D, Mapping>>,
    blend: impl Fn(D, D, u8) -> D + Sync,
) -> <CurrentBackend as Backend>::Storage<D> {
    match operand0 {
        VeRhs::Vrf { data } => CurrentBackend::zip3_with(out, &data.inner, &tag.inner, blend),
        VeRhs::Stash => {
            let stash = stash_data.expect("VeRhs::Stash operand requires stash_data; caller must supply it");
            CurrentBackend::zip3_with(out, &stash.inner, &tag.inner, blend)
        }
        VeRhs::Const { v } => {
            let rhs = Tensor::<D, Mapping>::splat(*v);
            CurrentBackend::zip3_with(out, &rhs.inner, &tag.inner, blend)
        }
    }
}

/// Applies a binary operation with branch-conditional execution: for each operand, `op` runs only on
/// cells whose `tag` matches the operand's `TagFilter`, leaving `out` unchanged elsewhere. Stash data
/// is passed as a `Tensor` (already transposed to the current mapping).
pub(super) fn apply_binary_op<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    tag: &Tensor<u8, Mapping>,
    op: impl Fn(D, D) -> D + Sync,
    operands: &[BinaryOperandTag<D, Mapping>],
    stash_data: Option<&Tensor<D, Mapping>>,
) -> Tensor<D, Mapping> {
    let mut out = data.inner.clone();
    for operand in operands {
        let filter = operand.tag_filter();
        let blend = |o: D, r: D, t: u8| if filter.matches(Opt::Init(t)) { op(o, r) } else { o };
        out = blend_operand(&out, operand.operand0(), tag, stash_data, blend);
    }
    Tensor::from_inner(out)
}

/// Applies a unary operation to every position of `data`. Unary ops accept no branch [`OperandTag`].
pub(super) fn apply_unary_op<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    op: impl Fn(D) -> D + Sync,
) -> Tensor<D, Mapping> {
    data.map(op)
}

/// Applies a ternary operation to a tensor with branch-conditional execution.
/// Ternary operations are only supported for f32 tensors.
pub(super) fn apply_ternary_op<Mapping: M>(
    data: &Tensor<f32, Mapping>,
    tag: &Tensor<u8, Mapping>,
    op: impl Fn(f32, f32, f32) -> f32 + Sync,
    operands: &[TernaryOperandTag<Mapping>],
    stash_data: Option<&Tensor<f32, Mapping>>,
) -> Tensor<f32, Mapping> {
    let mut out = data.inner.clone();
    for operand in operands {
        let filter = operand.tag_filter();
        let rhs1 = operand.operand1();
        let blend = |o: f32, r: f32, t: u8| {
            if filter.matches(Opt::Init(t)) {
                op(o, r, rhs1)
            } else {
                o
            }
        };
        out = blend_operand(&out, operand.operand0(), tag, stash_data, blend);
    }
    Tensor::from_inner(out)
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
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
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
    ) -> VectorLogicTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Logic, stage::Standalone>(op, None, operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Logic binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic_with_mode)]
    pub fn vector_logic_with_mode<Op>(
        self,
        op: LogicBinaryOpI32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorLogicTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Logic, stage::Standalone>(op, Some(mode), operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
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
    ) -> VectorLogicTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Logic, stage::Standalone>(op, None, operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Logic binary operation with explicit mode (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_logic_with_mode)]
    pub fn vector_logic_with_mode<Op>(
        self,
        op: LogicBinaryOpF32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorLogicTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Logic, stage::Standalone>(op, Some(mode), operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
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
    ) -> VectorFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Fxp, stage::Standalone>(op, None, operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Fixed-point binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_fxp_with_mode)]
    pub fn vector_fxp_with_mode<Op>(
        self,
        op: FxpBinaryOp,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorFxpTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Fxp, stage::Standalone>(op, Some(mode), operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    Stash: StashSlot<StashD>,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
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
    #[primitive(VectorTensor::vector_narrow_split)]
    pub fn vector_narrow_split<Time2: M, Packet2: M>(
        self,
    ) -> VectorNarrowTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, StashD, Stash, VE_ORDER, FS, { Way4 }> {
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
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
    /// For packets with > 4 real elements, use `vector_narrow_split()` instead.
    #[primitive(VectorTensor::vector_narrow_trim)]
    pub fn vector_narrow_trim<Packet2: M>(
        self,
    ) -> VectorNarrowTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet2, StashD, Stash, VE_ORDER, FS, { Way4 }> {
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
where
    S: stage::Stage + CanTransitionTo<stage::Fp>,
{
    /// Fp unary operation (f32 only).
    #[primitive(VectorTensor::vector_fp_unary)]
    pub fn vector_fp_unary(
        mut self,
        op: FpUnaryOp,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER> {
        self.ve_state_mut().use_alu(op.alu());
        let result = apply_unary_op(self.inner(), op.unary_op_fn());
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
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let operands = operand.into_operands();
        let vt = self.do_binary::<stage::Fp, stage::Standalone>(op, None, operands);
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Fp binary operation with explicit mode (f32 only). See [`vector_fp_binary`](Self::vector_fp_binary).
    #[primitive(VectorTensor::vector_fp_binary_with_mode)]
    pub fn vector_fp_binary_with_mode<Op>(
        self,
        op: FpBinaryOp,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let operands = operand.into_operands();
        let vt = self.do_binary::<stage::Fp, stage::Standalone>(op, Some(mode), operands);
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoTernaryOperandTags<VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
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
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoTernaryOperandTags<VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let operands = operands.into_ternary_operands();
        // TODO: we should only read stash if actually used by an operand, just like apply_binary
        let stash_data: Option<Tensor<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>> =
            self.ve_state().force_clone_stash_as();
        self.ve_state_mut().use_alu(op.alu());
        let op_fn = op.ternary_op_fn(Some(mode));
        let result = apply_ternary_op(
            self.inner(),
            self.tag(),
            op_fn,
            operands.as_slice(),
            stash_data.as_ref(),
        );
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
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
        StashD,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
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
        StashD,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way4 }>
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
    ) -> VectorFpDivTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER, FS, { Way4 }>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::FpDiv, FS>(FpDivBinaryOp::DivF, None, operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Floating-point division with explicit mode. See [`vector_fp_div`](Self::vector_fp_div).
    #[primitive(VectorTensor::vector_fp_div_with_mode)]
    pub fn vector_fp_div_with_mode<Op>(
        self,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorFpDivTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER, FS, { Way4 }>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::FpDiv, FS>(FpDivBinaryOp::DivF, Some(mode), operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    Stash: StashSlot<StashD>,
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
    #[primitive(VectorTensor::vector_widen_concat)]
    pub fn vector_widen_concat<Time2: M, Packet2: M>(
        self,
    ) -> VectorWidenTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, StashD, Stash, VE_ORDER, FS, { Way8 }> {
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
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
    /// Reverse of `vector_narrow_trim`. Use this when no time-dimension merging is needed.
    /// For merging split time steps back, use `vector_widen_concat()` instead.
    #[primitive(VectorTensor::vector_widen_pad)]
    pub fn vector_widen_pad<Packet2: M>(
        self,
    ) -> VectorWidenTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet2, StashD, Stash, VE_ORDER, FS, { Way8 }> {
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
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
    StashD: VeScalar,
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, i32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
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
    ) -> VectorClipTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Clip, stage::Standalone>(op, None, operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Clip binary operation with explicit mode (i32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip_with_mode)]
    pub fn vector_clip_with_mode<Op>(
        self,
        op: ClipBinaryOpI32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorClipTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<i32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Clip, stage::Standalone>(op, Some(mode), operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    Stash: StashSlot<StashD>,
    FS: stage::VeTensorContext,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, f32, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, FS, { Way8 }>
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
    ) -> VectorClipTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Clip, stage::Standalone>(op, None, operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
    }

    /// Clip binary operation with explicit mode (f32 only). Requires `Way8` mode.
    #[primitive(VectorTensor::vector_clip_with_mode)]
    pub fn vector_clip_with_mode<Op>(
        self,
        op: ClipBinaryOpF32,
        mode: BinaryArgMode,
        operand: Op,
    ) -> VectorClipTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, StashD, Op::Next, VE_ORDER>
    where
        Op: IntoOperands<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> + StashTransition<StashD, Stash>,
    {
        let vt = self.do_binary::<stage::Clip, stage::Standalone>(op, Some(mode), operand.into_operands());
        VectorTensor {
            ctx: vt.ctx,
            data: vt.data.apply_stash_transition::<Op>(),
        }
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
    Stash: StashSlot<StashD>,
    const VE_ORDER: VeOrder,
> VectorTensor<'l, T, S, D, Chip, Cluster, Slice, Time, Packet, StashD, Stash, VE_ORDER, stage::Standalone, { Way8 }>
where
    S: stage::Stage + CanTransitionTo<stage::Filter>,
{
    /// Filter by branch ID. Requires `Way8` mode and the `Standalone` context (not `Group` or `Zipped`).
    #[primitive(VectorTensor::vector_filter)]
    pub fn vector_filter<Time2: M>(
        self,
        _config: TagFilter,
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
