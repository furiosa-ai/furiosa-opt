//! Vector Tensor Pair API for two-group operations.
//!
//! This module provides a type-safe API for processing two groups of interleaved data
//! with different preprocessing operations before combining them with a binary operation.
//!
//! # Key Types
//! - `VectorTensorPair`: Manages both groups together with unified operations
//! - `GroupOperand`: Optional operand for each group (`None` skips the operation)
//!
//! # Pipeline Flow
//! ```text
//! CollectTensor
//!   └─ vector_init()
//!        └─ vector_intra_slice_unzip() → VectorTensorPair<Branch>
//!                                        │
//!                                        ├─ vector_fxp(...) → VectorTensorPair<Fxp>
//!                                        ├─ fxp_to_fp() → VectorTensorPair<FxpToFp>
//!                                        │
//!                                        └─ vector_clip_zip() → VectorClipTensor (Zipped)
//!                                             └─ vector_final() → commit
//! ```
//!
//! # Example
//! ```ignore
//! // Simple binary add (no preprocessing)
//! ctx.main
//!     .begin_interleaved(lhs.view(), rhs.view())
//!     .fetch::<i32, m![I], m![A % 8]>()
//!     .switch::<m![A / 8], m![I]>(config)
//!     .collect::<m![I], m![A % 8]>()
//!     .vector_init()
//!     .vector_intra_slice_unzip::<{ Ident::I }, m![1 # 2], m![1]>()
//!     .vector_clip_zip(ClipBinaryOpI32::AddFxp)
//!     .vector_final()
//!     .commit(addr)
//!
//! // With group preprocessing (both groups at once)
//! ctx.main
//!     .begin_interleaved(lhs.view(), rhs.view())
//!     .fetch::<i32, m![I], m![A % 8]>()
//!     .switch::<m![A / 8], m![I]>(config)
//!     .collect::<m![I], m![A % 8]>()
//!     .vector_init()
//!     .vector_intra_slice_unzip::<{ Ident::I }, m![1 # 2], m![1]>()
//!     .vector_fxp(FxpBinaryOp::MulInt, Some(operands0), Some(operands1))
//!     .vector_clip_zip(ClipBinaryOpI32::AddFxp)
//!     .vector_final()
//!     .commit(addr)
//! ```

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_mapping_macro::primitive;
use furiosa_opt_macro::m;

use super::VeTensorShape;

use crate::context::*;
use crate::prelude::RngdAlu;
use crate::prelude::semantics::HasConversionOp;
use crate::scalar::Opt;
use crate::tensor::*;
use crate::tensor_state::NoTensor;
use crate::vector_engine::layer::{FpToFxp, FxpToFp};
use crate::vector_engine::op::{
    BinaryArgMode, ClipBinaryOpF32, ClipBinaryOpI32, FpBinaryOp, FpDivBinaryOp, FpTernaryOp, FpUnaryOp, FxpBinaryOp,
    HasAlu, HasBinaryOp, HasTernaryOp, HasUnaryOp, LogicBinaryOpF32, LogicBinaryOpI32, TernaryArgMode, UnaryArgMode,
};
use crate::vector_engine::operand::{GroupOperand, IntoGroupOperand, IntoGroupTernaryOperand, TernaryOperand};
use crate::vector_engine::scalar::VeScalar;
use crate::vector_engine::stage::markers as stage;
use crate::vector_engine::stage::markers::{
    CanTransitionTo,
    PacketMode::{self, Way4, Way8},
    VeOrder,
};
use crate::vector_engine::stage::state::VeState;
use crate::vector_engine::tensor::verify::{verify_vector_concat, verify_vector_split};

use super::vector_tensor::{
    VeTensorData, VectorClipTensor, VectorFpTensor, VectorFxpTensor, VectorLogicTensor, apply_binary_op,
    apply_ternary_op,
};

/// Combines two tensors using a binary operation.
/// Used by VectorTensorPair::combine operations.
/// - When both values are Init, applies the operation
/// - Otherwise returns Uninit
fn zip_groups<D: VeScalar, Mapping: M>(
    op_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
    lhs: &Tensor<D, Mapping>,
    rhs: &Tensor<D, Mapping>,
) -> Tensor<D, Mapping> {
    lhs.zip_with(rhs, |l, r| match (l, r) {
        (Opt::Init(_), Opt::Init(_)) => op_fn(l, r),
        _ => Opt::Uninit,
    })
}

// ============================================================================
// Type aliases
// ============================================================================

/// Type alias for group tensor data with Group state.
type GroupTensorData<S, D, Chip, Cluster, Slice, Time, Packet, const W: PacketMode = { Way8 }> =
    VeTensorData<S, D, Chip, Cluster, Slice, Time, Packet, D, NoTensor, { VeOrder::IntraFirst }, stage::Group, W>;

// ============================================================================
// VectorTensorPair - Manages both groups together
// ============================================================================

/// Pair of tensors that manages both groups together.
///
/// Unified operations apply to both groups simultaneously:
/// - `vector_fxp()`, `vector_logic()`, `vector_clip()`, etc.
/// - Each takes `GroupOperand` for both groups; `None` skips the operation for that group
///
/// Common operations (FxpToFp, Narrow, Widen, FpToFxp) apply to both sides simultaneously.
///
/// After a binary operation (`vector_clip_zip`, `vector_fxp_zip`, etc.), the pair is combined
/// into a single `VectorTensor` with `Zipped` state.
///
/// Each group uses `VeTensorData` with `Group` state, which prevents:
/// - `stash()` operations on individual groups
/// - Common operations (fxp_to_fp, split, concat, fp_to_fxp) on individual groups
#[derive(Debug)]
pub struct VectorTensorPair<
    'l,
    const T: Tu,
    D: VeScalar,
    S: stage::Stage,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
    const W: PacketMode = { Way8 },
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,

    /// Group 0 tensor data (with Group state)
    pub(crate) group0: GroupTensorData<S, D, Chip, Cluster, Slice, SplitTime, Packet, W>,

    /// Group 1 tensor data (with Group state)
    pub(crate) group1: GroupTensorData<S, D, Chip, Cluster, Slice, SplitTime, Packet, W>,
}

// ============================================================================
// VectorTensorPair - Constructor (Branch stage)
// ============================================================================

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Packet: M, SplitTime: M, const W: PacketMode>
    VectorTensorPair<'l, T, D, stage::Branch, Chip, Cluster, Slice, SplitTime, Packet, W>
{
    /// Creates a new VectorTensorPair by tiling the input tensor along the interleaved group axis.
    /// Group 0 gets tile index 0, Group 1 gets tile index 1.
    /// Both groups start at Branch stage.
    #[primitive(VectorTensorPair::new)]
    pub(crate) fn new<I: AxisName, Time: M, TileTime: M>(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    ) -> Self {
        assert_eq!(
            Packet::SIZE,
            8,
            "VectorTensorPair requires Packet of 8 elements (one flit) in Way8 mode, got {}",
            Packet::SIZE,
        );
        // SAFETY: We're transmuting between tensor shapes that differ only in the Time dimension.
        // The tile operation produces TileTime (with the interleaved group axis replaced by 1 # 2), and we transmute
        // to SplitTime which should have the same memory layout.
        let (g0_inner, g1_inner): (
            Tensor<D, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
            Tensor<D, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        ) = unsafe {
            let g0 = inner
                .view()
                .tile::<Symbol<I>, m![{ Chip }, { Cluster }, { Slice }, { TileTime }, { Packet }], 1>(0)
                .read()
                .transmute();
            let g1 = inner
                .view()
                .tile::<Symbol<I>, m![{ Chip }, { Cluster }, { Slice }, { TileTime }, { Packet }], 1>(1)
                .read()
                .transmute();
            (g0, g1)
        };

        // Create default execution_id (all zeros)
        let g0_execution_id = g0_inner.map(|_| Opt::Init(0u8));
        let g1_execution_id = g1_inner.map(|_| Opt::Init(0u8));

        VectorTensorPair {
            ctx,
            group0: VeTensorData {
                inner: g0_inner,
                execution_id: g0_execution_id,
                ve_state: VeState::new(),
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
            group1: VeTensorData {
                inner: g1_inner,
                execution_id: g1_execution_id,
                ve_state: VeState::new(),
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }
}

// ============================================================================
// VectorTensorPair - Internal helpers for binary operations (generic D type)
// ============================================================================

impl<
    'l,
    const T: Tu,
    D: VeScalar,
    S: stage::Stage,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
    const W: PacketMode,
> VectorTensorPair<'l, T, D, S, Chip, Cluster, Slice, SplitTime, Packet, W>
{
    /// Internal helper for applying binary operations to both groups.
    /// The target stage is determined by the caller.
    /// Uses trait bounds to get ALU and operation function from the operation type.
    ///
    /// Note: ALU state is updated for both groups when either group has an operand,
    /// because ALU is a global resource shared between groups.
    fn apply_binary_to_both<TargetStage, Op>(
        self,
        op: Op,
        mode: Option<BinaryArgMode>,
        group0_operand: GroupOperand<D, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: GroupOperand<D, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, D, TargetStage, Chip, Cluster, Slice, SplitTime, Packet, W>
    where
        TargetStage: stage::Stage,
        Op: HasAlu + HasBinaryOp<D>,
    {
        let alu = op.alu();
        let group0_execution_id = self.group0.execution_id.clone();
        let group1_execution_id = self.group1.execution_id.clone();
        let op_fn = op.binary_op_fn(mode);

        let fn_group0 = group0_operand.map(|operand| {
            |data: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
                apply_binary_op(&data, &group0_execution_id, &op_fn, &[operand], None)
            }
        });

        let fn_group1 = group1_operand.map(|operand| {
            |data: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
                apply_binary_op(&data, &group1_execution_id, &op_fn, &[operand], None)
            }
        });

        self.apply_op_to_both(alu, fn_group0, fn_group1)
    }

    /// Internal helper for applying custom operations to both groups.
    fn apply_op_to_both<TargetStage, F0, F1>(
        mut self,
        alu: RngdAlu,
        fn_group0: Option<F0>,
        fn_group1: Option<F1>,
    ) -> VectorTensorPair<'l, T, D, TargetStage, Chip, Cluster, Slice, SplitTime, Packet, W>
    where
        TargetStage: stage::Stage,
        F0: FnOnce(
            Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>,
        ) -> Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>,
        F1: FnOnce(
            Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>,
        ) -> Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>,
    {
        // ALU Resource Management: when any group operation is present, mark ALU as used
        if fn_group0.is_some() || fn_group1.is_some() {
            self.group0.ve_state.use_alu(alu);
            self.group1.ve_state.use_alu(alu);
        }

        let group0_inner = match fn_group0 {
            Some(f) => f(self.group0.inner),
            None => self.group0.inner,
        };

        let group1_inner = match fn_group1 {
            Some(f) => f(self.group1.inner),
            None => self.group1.inner,
        };

        VectorTensorPair {
            ctx: self.ctx,
            group0: VeTensorData {
                inner: group0_inner,
                execution_id: self.group0.execution_id,
                ve_state: self.group0.ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
            group1: VeTensorData {
                inner: group1_inner,
                execution_id: self.group1.execution_id,
                ve_state: self.group1.ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }

    fn apply_conversion_op_to_both<TargetStage, Op, D2: VeScalar>(
        self,
        op: Op,
    ) -> VectorTensorPair<'l, T, D2, TargetStage, Chip, Cluster, Slice, SplitTime, Packet, W>
    where
        TargetStage: stage::Stage,
        Op: HasConversionOp<D, D2>,
    {
        let op_fn = op.conversion_op_fn();

        let fn_group0 = |data: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
            data.map(|&v| v.map(&op_fn))
        };

        let fn_group1 = |data: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
            data.map(|&v| v.map(&op_fn))
        };

        VectorTensorPair {
            ctx: self.ctx,
            group0: VeTensorData {
                inner: fn_group0(self.group0.inner),
                execution_id: self.group0.execution_id,
                ve_state: self.group0.ve_state.retype(),
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
            group1: VeTensorData {
                inner: fn_group1(self.group1.inner),
                execution_id: self.group1.execution_id,
                ve_state: self.group1.ve_state.retype(),
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }
}

// ============================================================================
// VectorTensorPair - Internal helpers for f32-specific operations (unary, ternary)
// ============================================================================

impl<'l, const T: Tu, S: stage::Stage, Chip: M, Cluster: M, Slice: M, SplitTime: M, Packet: M, const W: PacketMode>
    VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, SplitTime, Packet, W>
{
    /// Internal helper for applying unary operations to both groups.
    /// Uses trait bounds to get ALU and operation function from the operation type.
    ///
    /// Note: ALU state is updated for both groups when either group applies the operation,
    /// because ALU is a global resource shared between groups.
    fn apply_unary_to_both<TargetStage, Op>(
        self,
        op: Op,
        mode: Option<UnaryArgMode>,
        group0_apply: bool,
        group1_apply: bool,
    ) -> VectorTensorPair<'l, T, f32, TargetStage, Chip, Cluster, Slice, SplitTime, Packet, W>
    where
        TargetStage: stage::Stage,
        Op: HasAlu + HasUnaryOp<f32>,
    {
        let op_fn = op.unary_op_fn(mode);
        let alu = op.alu();

        let fn_group0 = group0_apply.then_some(
            |data: Tensor<f32, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
                data.map(|&v| op_fn(v))
            },
        );

        let fn_group1 = group1_apply.then_some(
            |data: Tensor<f32, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
                data.map(|&v| op_fn(v))
            },
        );

        self.apply_op_to_both(alu, fn_group0, fn_group1)
    }

    /// Internal helper for applying ternary operations to both groups.
    /// Uses trait bounds to get ALU and operation function from the operation type.
    ///
    /// Note: ALU state is updated for both groups when either group has an operand,
    /// because ALU is a global resource shared between groups.
    fn apply_ternary_to_both<TargetStage, Op>(
        self,
        op: Op,
        mode: Option<TernaryArgMode>,
        group0_operand: Option<TernaryOperand<VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>>,
        group1_operand: Option<TernaryOperand<VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>>,
    ) -> VectorTensorPair<'l, T, f32, TargetStage, Chip, Cluster, Slice, SplitTime, Packet, W>
    where
        TargetStage: stage::Stage,
        Op: HasAlu + HasTernaryOp<f32>,
    {
        let alu = op.alu();
        let group0_execution_id = self.group0.execution_id.clone();
        let group1_execution_id = self.group1.execution_id.clone();
        let op_fn = op.ternary_op_fn(mode);

        let fn_group0 = group0_operand.map(|operand| {
            |data: Tensor<f32, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
                apply_ternary_op(&data, &group0_execution_id, &op_fn, &[operand], None)
            }
        });

        let fn_group1 = group1_operand.map(|operand| {
            |data: Tensor<f32, m![{ Chip }, { Cluster }, { Slice }, { SplitTime }, { Packet }]>| {
                apply_ternary_op(&data, &group1_execution_id, &op_fn, &[operand], None)
            }
        });

        self.apply_op_to_both(alu, fn_group0, fn_group1)
    }
}

// ============================================================================
// VectorTensorPair - Logic operations (i32/f32)
// Stage order: Logic comes after Branch
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Logic>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Logic binary operation on both groups simultaneously. Requires `Way8` mode.
    ///
    /// # Arguments
    /// * `op` - The logic binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `i32` for constant
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `i32` for constant
    ///
    /// # Stage Transition
    /// Both groups transition to `stage::Logic` regardless of whether operands are provided.
    #[primitive(VectorTensorPair::vector_logic)]
    pub fn vector_logic(
        self,
        op: LogicBinaryOpI32,
        group0_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, i32, stage::Logic, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Logic binary operation on both groups simultaneously with explicit mode. Requires `Way8` mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_logic_with_mode)]
    pub fn vector_logic_with_mode(
        self,
        op: LogicBinaryOpI32,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, i32, stage::Logic, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }
}

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Logic>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Logic binary operation on both groups simultaneously. Requires `Way8` mode.
    ///
    /// # Arguments
    /// * `op` - The logic binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `f32` for constant
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `f32` for constant
    ///
    /// # Stage Transition
    /// Both groups transition to `stage::Logic` regardless of whether operands are provided.
    #[primitive(VectorTensorPair::vector_logic)]
    pub fn vector_logic(
        self,
        op: LogicBinaryOpF32,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Logic, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Logic binary operation on both groups simultaneously with explicit mode. Requires `Way8` mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_logic_with_mode)]
    pub fn vector_logic_with_mode(
        self,
        op: LogicBinaryOpF32,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Logic, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }
}

// ============================================================================
// VectorTensorPair - Fxp operations (i32 only)
// Stage order: Fxp comes after Logic
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Fxp>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Fixed-point binary operation on both groups simultaneously. Requires `Way8` mode.
    ///
    /// # Arguments
    /// * `op` - The fixed-point binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `i32` for constant
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `i32` for constant
    ///
    /// # Stage Transition
    /// Both groups transition to `stage::Fxp` regardless of whether operands are provided.
    ///
    /// # Example
    /// ```ignore
    /// .vector_fxp(FxpBinaryOp::MulInt, 16384, ())  // Apply to group0 only
    /// .vector_fxp(FxpBinaryOp::MulInt, (), 16384)  // Apply to group1 only
    /// .vector_fxp(FxpBinaryOp::MulInt, 16384, 32768)  // Apply to both
    /// ```
    #[primitive(VectorTensorPair::vector_fxp)]
    pub fn vector_fxp(
        self,
        op: FxpBinaryOp,
        group0_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, i32, stage::Fxp, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Fixed-point binary operation on both groups simultaneously with explicit mode. Requires `Way8` mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_fxp_with_mode)]
    pub fn vector_fxp_with_mode(
        self,
        op: FxpBinaryOp,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, i32, stage::Fxp, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }
}

// ============================================================================
// VectorTensorPair - FxpToFp conversion (i32 -> f32)
// Stage order: FxpToFp comes after Fxp
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::FxpToFp>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Converts i32 to f32 for both groups simultaneously. Requires `Way8` mode.
    /// This is a common operation that must be applied to both groups.
    #[primitive(VectorTensorPair::vector_fxp_to_fp)]
    pub fn vector_fxp_to_fp(
        self,
        int_width: u32,
    ) -> VectorTensorPair<'l, T, f32, stage::FxpToFp, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_conversion_op_to_both(FxpToFp::new(int_width))
    }
}

// ============================================================================
// VectorTensorPair - Narrow operations (split / trim)
// Stage order: Narrow comes after FxpToFp
// ============================================================================

impl<
    'l,
    const T: Tu,
    D: VeScalar,
    S: stage::Stage + CanTransitionTo<stage::Narrow>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, D, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Narrow layer (split) operation for both groups simultaneously. Requires `Way8` mode.
    ///
    /// Takes an 8-element packet in each group, splits it into front 4 + back 4.
    /// The factor of 2 goes into SplitTime, output is `Way4` with 4-element packets.
    /// Output: `SplitTime2 = SplitTime × 2`, `Packet2` = front 4 of Packet (size 4).
    #[primitive(VectorTensorPair::vector_split)]
    pub fn vector_split<SplitTime2: M, Packet2: M>(
        self,
    ) -> VectorTensorPair<'l, T, D, stage::Narrow, Chip, Cluster, Slice, SplitTime2, Packet2, { Way4 }> {
        verify_vector_split::<SplitTime, Packet, SplitTime2, Packet2>();

        let (g0_inner, g0_eid, g0_ve_state) = self.group0.into_parts();
        let (g1_inner, g1_eid, g1_ve_state) = self.group1.into_parts();

        type OutShape<Chip, Cluster, Slice, SplitTime2, Packet2> =
            VeTensorShape<Chip, Cluster, Slice, SplitTime2, Packet2>;

        VectorTensorPair {
            ctx: self.ctx,
            group0: VeTensorData {
                inner: g0_inner.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                execution_id: g0_eid.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                ve_state: g0_ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
            group1: VeTensorData {
                inner: g1_inner.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                execution_id: g1_eid.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                ve_state: g1_ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }
}

// ============================================================================
// vector_trim_way4 is intentionally NOT available on `VectorTensorPair`.
//
// `vector_trim_way4` is the bypass (no-op) form of Split: it only reinterprets
// the Way8 flit's front 4 lanes as Way4 without emitting any hardware
// instruction. In combination with a later `vector_pad_way8` (which also
// bypass-pads in LIR as `Concat(Bypass)`), the group execution-id tensor gets
// zero-padded on the back 4 lanes, so the final `filter: ValidGroup::One`
// compacts *only* the 4 originally-valid lanes per packet — silently producing
// a different shape than the VE pipeline's declared output.
//
// Pair tensors must therefore use the buffered `vector_split` / `vector_concat`
// pair (which actually moves data between the front and back halves rather
// than just reinterpreting lane shape) before running float-side per-group
// operations. Users that only need a single stream can still call
// `vector_fp_zip` / `vector_clip_zip` / `vector_fxp_zip` to consume the pair
// and then `vector_trim_way4` on the resulting `VectorTensor`.
// ============================================================================

// ============================================================================
// VectorTensorPair - Floating-point operations (f32 only)
// Stage order: Fp comes after Narrow
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Fp>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }>
{
    /// Floating-point unary operation on both groups simultaneously.
    ///
    /// # Arguments
    /// * `op` - The floating-point unary operation to apply
    /// * `group0_apply` - Whether to apply to Group 0
    /// * `group1_apply` - Whether to apply to Group 1
    #[primitive(VectorTensorPair::vector_fp_unary)]
    pub fn vector_fp_unary(
        self,
        op: FpUnaryOp,
        group0_apply: bool,
        group1_apply: bool,
    ) -> VectorTensorPair<'l, T, f32, stage::Fp, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_unary_to_both(op, None, group0_apply, group1_apply)
    }

    /// Floating-point unary operation on both groups with explicit mode.
    #[primitive(VectorTensorPair::vector_fp_unary_with_mode)]
    pub fn vector_fp_unary_with_mode(
        self,
        op: FpUnaryOp,
        mode: UnaryArgMode,
        group0_apply: bool,
        group1_apply: bool,
    ) -> VectorTensorPair<'l, T, f32, stage::Fp, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_unary_to_both(op, Some(mode), group0_apply, group1_apply)
    }

    /// Floating-point binary operation on both groups simultaneously.
    ///
    /// # Arguments
    /// * `op` - The floating-point binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip
    #[primitive(VectorTensorPair::vector_fp_binary)]
    pub fn vector_fp_binary(
        self,
        op: FpBinaryOp,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Fp, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Floating-point binary operation on both groups with explicit mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_fp_binary_with_mode)]
    pub fn vector_fp_binary_with_mode(
        self,
        op: FpBinaryOp,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Fp, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Floating-point ternary operation on both groups simultaneously.
    ///
    /// # Arguments
    /// * `op` - The floating-point ternary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `(f32, f32)` for constants
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `(f32, f32)` for constants
    ///
    /// # Example
    /// ```ignore
    /// // Apply ternary op only to group0
    /// pair.vector_fp_ternary(op, (2.0f32, 3.0f32), ())
    ///
    /// // Apply to both groups with different operands
    /// pair.vector_fp_ternary(op, (2.0f32, 3.0f32), (4.0f32, 5.0f32))
    ///
    /// // With stash as operand0 for group0
    /// pair.vector_fp_ternary(op, (Stash, 3.0f32), ())
    /// ```
    #[primitive(VectorTensorPair::vector_fp_ternary)]
    pub fn vector_fp_ternary(
        self,
        op: FpTernaryOp,
        group0_operand: impl IntoGroupTernaryOperand<VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupTernaryOperand<VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Fp, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_ternary_to_both(
            op,
            None,
            group0_operand.into_group_ternary_operand(),
            group1_operand.into_group_ternary_operand(),
        )
    }

    /// Floating-point ternary operation on both groups with explicit mode.
    #[primitive(VectorTensorPair::vector_fp_ternary_with_mode)]
    pub fn vector_fp_ternary_with_mode(
        self,
        op: FpTernaryOp,
        mode: TernaryArgMode,
        group0_operand: impl IntoGroupTernaryOperand<VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupTernaryOperand<VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Fp, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_ternary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_ternary_operand(),
            group1_operand.into_group_ternary_operand(),
        )
    }
}

// ============================================================================
// VectorTensorPair - FpDiv operations (f32 only)
// Stage order: FpDiv comes after Fp (skipping Reduce)
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::FpDiv>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }>
{
    /// Floating-point division on both groups simultaneously.
    ///
    /// # Arguments
    /// * `op` - The fp div binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `f32` for constant
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `f32` for constant
    ///
    /// # Stage Transition
    /// Both groups transition to `stage::FpDiv` regardless of whether operands are provided.
    #[primitive(VectorTensorPair::vector_fp_div)]
    pub fn vector_fp_div(
        self,
        op: FpDivBinaryOp,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::FpDiv, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Floating-point division on both groups simultaneously with explicit mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_fp_div_with_mode)]
    pub fn vector_fp_div_with_mode(
        self,
        op: FpDivBinaryOp,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::FpDiv, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }
}

// ============================================================================
// VectorTensorPair - Widen operations (concat / pad)
// Stage order: Widen comes after FpDiv
// ============================================================================

impl<
    'l,
    const T: Tu,
    D: VeScalar,
    S: stage::Stage + CanTransitionTo<stage::Widen>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, D, S, Chip, Cluster, Slice, SplitTime, Packet, { Way4 }>
{
    /// Widen layer (concat) operation for both groups simultaneously. Requires `Way4` mode.
    ///
    /// Reverse of split. Takes 4-element packets from 2 consecutive time steps,
    /// merges them into one 8-element packet and transitions to `Way8`.
    /// `SplitTime2 = SplitTime / 2`, `Packet2` = Packet combined with factor of 2 from Time.
    #[primitive(VectorTensorPair::vector_concat)]
    pub fn vector_concat<SplitTime2: M, Packet2: M>(
        self,
    ) -> VectorTensorPair<'l, T, D, stage::Widen, Chip, Cluster, Slice, SplitTime2, Packet2, { Way8 }> {
        verify_vector_concat::<SplitTime, Packet, SplitTime2, Packet2>();

        let (g0_inner, g0_eid, g0_ve_state) = self.group0.into_parts();
        let (g1_inner, g1_eid, g1_ve_state) = self.group1.into_parts();

        type OutShape<Chip, Cluster, Slice, SplitTime2, Packet2> =
            VeTensorShape<Chip, Cluster, Slice, SplitTime2, Packet2>;

        VectorTensorPair {
            ctx: self.ctx,
            group0: VeTensorData {
                inner: g0_inner.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                execution_id: g0_eid.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                ve_state: g0_ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
            group1: VeTensorData {
                inner: g1_inner.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                execution_id: g1_eid.transpose::<OutShape<Chip, Cluster, Slice, SplitTime2, Packet2>>(true),
                ve_state: g1_ve_state,
                _stage: PhantomData,
                _filter_state: PhantomData,
            },
        }
    }
}

// ============================================================================
// vector_pad_way8 is intentionally NOT available on `VectorTensorPair`.
//
// Symmetric to the `vector_trim_way4` block above: the LIR realization of
// `vector_pad_way8` is `Concat(Bypass)`, which zero-pads the group
// execution-id tensor on the back 4 Way8 lanes. Any downstream op that
// filters by `ValidGroup` on Way8 then only matches the front 4 lanes while
// the cache is still laid out with 8 lanes per partition — the same
// Way8-filter-vs-Way8-cache misalignment that motivates blocking
// `vector_trim_way4`. Pair tensors must use the buffered `vector_concat`
// instead (it actually moves data between the front and back halves, so
// exec-ids stay meaningful). Zipping the pair first (`vector_fp_zip`,
// `vector_clip_zip`, `vector_fxp_zip`) and then calling `vector_pad_way8`
// on the resulting single `VectorTensor` remains fine.
// ============================================================================

// ============================================================================
// VectorTensorPair - FpToFxp conversion (f32 -> i32)
// Stage order: FpToFxp comes after Widen
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::FpToFxp>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Converts f32 to i32 for both groups simultaneously. Requires `Way8` mode.
    /// This is a common operation that must be applied to both groups.
    #[primitive(VectorTensorPair::vector_fp_to_fxp)]
    pub fn vector_fp_to_fxp(
        self,
        int_width: u32,
    ) -> VectorTensorPair<'l, T, i32, stage::FpToFxp, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_conversion_op_to_both(FpToFxp::new(int_width))
    }
}

// ============================================================================
// VectorTensorPair - Clip operations (i32/f32)
// Stage order: Clip comes after FpToFxp
// ============================================================================

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Clip>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Clip binary operation on both groups simultaneously. Requires `Way8` mode.
    ///
    /// # Arguments
    /// * `op` - The clip binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `i32` for constant
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `i32` for constant
    ///
    /// # Stage Transition
    /// Both groups transition to `stage::Clip` regardless of whether operands are provided.
    #[primitive(VectorTensorPair::vector_clip)]
    pub fn vector_clip(
        self,
        op: ClipBinaryOpI32,
        group0_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, i32, stage::Clip, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Clip binary operation on both groups simultaneously with explicit mode. Requires `Way8` mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_clip_with_mode)]
    pub fn vector_clip_with_mode(
        self,
        op: ClipBinaryOpI32,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<i32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, i32, stage::Clip, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }
}

impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Clip>,
    Chip: M,
    Cluster: M,
    Slice: M,
    SplitTime: M,
    Packet: M,
> VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }>
{
    /// Clip binary operation on both groups simultaneously. Requires `Way8` mode.
    ///
    /// # Arguments
    /// * `op` - The clip binary operation to apply
    /// * `group0_operand` - Operand for Group 0. Use `()` to skip, or `f32` for constant
    /// * `group1_operand` - Operand for Group 1. Use `()` to skip, or `f32` for constant
    ///
    /// # Stage Transition
    /// Both groups transition to `stage::Clip` regardless of whether operands are provided.
    #[primitive(VectorTensorPair::vector_clip)]
    pub fn vector_clip(
        self,
        op: ClipBinaryOpF32,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Clip, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            None,
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }

    /// Clip binary operation on both groups simultaneously with explicit mode. Requires `Way8` mode.
    /// In this paired form, `BinaryArgMode` is interpreted independently inside each group:
    /// `0` means that group's stream and `1` means that group's operand.
    #[primitive(VectorTensorPair::vector_clip_with_mode)]
    pub fn vector_clip_with_mode(
        self,
        op: ClipBinaryOpF32,
        mode: BinaryArgMode,
        group0_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
        group1_operand: impl IntoGroupOperand<f32, VeTensorShape<Chip, Cluster, Slice, SplitTime, Packet>>,
    ) -> VectorTensorPair<'l, T, f32, stage::Clip, Chip, Cluster, Slice, SplitTime, Packet, { Way8 }> {
        self.apply_binary_to_both(
            op,
            Some(mode),
            group0_operand.into_group_operand(),
            group1_operand.into_group_operand(),
        )
    }
}

// ============================================================================
// VectorTensorPair - Combine operations (returns VectorTensor)
// These operations combine both groups into a single tensor.
// Ordered by target stage: Logic → Fxp → Fp → Clip
// ============================================================================

// Logic combine (i32)
impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Logic>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
> VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, Time, Packet, { Way8 }>
{
    /// Binary logic operation merging Group 0 and Group 1. Requires `Way8` mode.
    /// Result = op(group0, group1), result is placed in Group 1 positions.
    /// Returns VectorTensor with Zipped state (filter/stash not available).
    #[primitive(VectorTensorPair::vector_logic_zip)]
    pub fn vector_logic_zip(
        mut self,
        op: LogicBinaryOpI32,
    ) -> VectorLogicTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        i32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(None), &self.group0.inner, &self.group1.inner);
        VectorLogicTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }

    /// Binary logic operation merging Group 0 and Group 1 with explicit mode. Requires `Way8` mode.
    /// In this zipped form, `BinaryArgMode` uses the two grouped streams directly:
    /// `0` means Group 0 and `1` means Group 1.
    #[primitive(VectorTensorPair::vector_logic_zip_with_mode)]
    pub fn vector_logic_zip_with_mode(
        mut self,
        op: LogicBinaryOpI32,
        mode: BinaryArgMode,
    ) -> VectorLogicTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        i32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(Some(mode)), &self.group0.inner, &self.group1.inner);
        VectorLogicTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }
}

// Logic combine (f32)
impl<
    'l,
    const T: Tu,
    S: stage::Stage + CanTransitionTo<stage::Logic>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
> VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, Time, Packet, { Way8 }>
{
    /// Binary logic operation merging Group 0 and Group 1. Requires `Way8` mode.
    /// Result = op(group0, group1), result is placed in Group 1 positions.
    /// Returns VectorTensor with Zipped state (filter/stash not available).
    #[primitive(VectorTensorPair::vector_logic_zip)]
    pub fn vector_logic_zip(
        mut self,
        op: LogicBinaryOpF32,
    ) -> VectorLogicTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        f32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(None), &self.group0.inner, &self.group1.inner);
        VectorLogicTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }

    /// Binary logic operation merging Group 0 and Group 1 with explicit mode. Requires `Way8` mode.
    /// In this zipped form, `BinaryArgMode` uses the two grouped streams directly:
    /// `0` means Group 0 and `1` means Group 1.
    #[primitive(VectorTensorPair::vector_logic_zip_with_mode)]
    pub fn vector_logic_zip_with_mode(
        mut self,
        op: LogicBinaryOpF32,
        mode: BinaryArgMode,
    ) -> VectorLogicTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        f32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(Some(mode)), &self.group0.inner, &self.group1.inner);
        VectorLogicTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }
}

// Fxp combine
impl<'l, const T: Tu, S: stage::Stage + CanTransitionTo<stage::Fxp>, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, Time, Packet, { Way8 }>
{
    /// Binary fxp operation merging Group 0 and Group 1. Requires `Way8` mode.
    /// Result = op(group0, group1), result is placed in Group 1 positions.
    /// Returns VectorTensor with Zipped state (filter/stash not available).
    #[primitive(VectorTensorPair::vector_fxp_zip)]
    pub fn vector_fxp_zip(
        mut self,
        op: FxpBinaryOp,
    ) -> VectorFxpTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        i32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(None), &self.group0.inner, &self.group1.inner);
        VectorFxpTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }

    /// Binary fxp operation merging Group 0 and Group 1 with explicit mode. Requires `Way8` mode.
    /// In this zipped form, `BinaryArgMode` uses the two grouped streams directly:
    /// `0` means Group 0 and `1` means Group 1.
    #[primitive(VectorTensorPair::vector_fxp_zip_with_mode)]
    pub fn vector_fxp_zip_with_mode(
        mut self,
        op: FxpBinaryOp,
        mode: BinaryArgMode,
    ) -> VectorFxpTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        i32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(Some(mode)), &self.group0.inner, &self.group1.inner);
        VectorFxpTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }
}

// Fp combine
impl<'l, const T: Tu, S: stage::Stage + CanTransitionTo<stage::Fp>, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, Time, Packet, { Way4 }>
{
    /// Binary fp operation merging Group 0 and Group 1.
    /// Result = op(group0, group1), result is placed in Group 1 positions.
    /// Returns VectorTensor with Zipped state (filter/stash not available).
    #[primitive(VectorTensorPair::vector_fp_zip)]
    pub fn vector_fp_zip(
        mut self,
        op: FpBinaryOp,
    ) -> VectorFpTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        f32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way4 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(None), &self.group0.inner, &self.group1.inner);
        VectorFpTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }

    /// Binary fp operation merging Group 0 and Group 1 with explicit mode.
    /// In this zipped form, `BinaryArgMode` uses the two grouped streams directly:
    /// `0` means Group 0 and `1` means Group 1.
    #[primitive(VectorTensorPair::vector_fp_zip_with_mode)]
    pub fn vector_fp_zip_with_mode(
        mut self,
        op: FpBinaryOp,
        mode: BinaryArgMode,
    ) -> VectorFpTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        f32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way4 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(Some(mode)), &self.group0.inner, &self.group1.inner);
        VectorFpTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }
}

// Clip combine (i32)
impl<'l, const T: Tu, S: stage::Stage + CanTransitionTo<stage::Clip>, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorTensorPair<'l, T, i32, S, Chip, Cluster, Slice, Time, Packet, { Way8 }>
{
    /// Binary clip operation merging Group 0 and Group 1. Requires `Way8` mode.
    /// Result = op(group0, group1), result is placed in Group 1 positions.
    /// Returns VectorTensor with Zipped state (filter/stash not available).
    #[primitive(VectorTensorPair::vector_clip_zip)]
    pub fn vector_clip_zip(
        mut self,
        op: ClipBinaryOpI32,
    ) -> VectorClipTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        i32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(None), &self.group0.inner, &self.group1.inner);
        VectorClipTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }

    /// Binary clip operation merging Group 0 and Group 1 with explicit mode. Requires `Way8` mode.
    /// In this zipped form, `BinaryArgMode` uses the two grouped streams directly:
    /// `0` means Group 0 and `1` means Group 1.
    #[primitive(VectorTensorPair::vector_clip_zip_with_mode)]
    pub fn vector_clip_zip_with_mode(
        mut self,
        op: ClipBinaryOpI32,
        mode: BinaryArgMode,
    ) -> VectorClipTensor<
        'l,
        T,
        i32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        i32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(Some(mode)), &self.group0.inner, &self.group1.inner);
        VectorClipTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }
}

// Clip combine (f32)
impl<'l, const T: Tu, S: stage::Stage + CanTransitionTo<stage::Clip>, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorTensorPair<'l, T, f32, S, Chip, Cluster, Slice, Time, Packet, { Way8 }>
{
    /// Binary clip operation merging Group 0 and Group 1. Requires `Way8` mode.
    /// Result = op(group0, group1), result is placed in Group 1 positions.
    /// Returns VectorTensor with Zipped state (filter/stash not available).
    #[primitive(VectorTensorPair::vector_clip_zip)]
    pub fn vector_clip_zip(
        mut self,
        op: ClipBinaryOpF32,
    ) -> VectorClipTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        f32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(None), &self.group0.inner, &self.group1.inner);
        VectorClipTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }

    /// Binary clip operation merging Group 0 and Group 1 with explicit mode. Requires `Way8` mode.
    /// In this zipped form, `BinaryArgMode` uses the two grouped streams directly:
    /// `0` means Group 0 and `1` means Group 1.
    #[primitive(VectorTensorPair::vector_clip_zip_with_mode)]
    pub fn vector_clip_zip_with_mode(
        mut self,
        op: ClipBinaryOpF32,
        mode: BinaryArgMode,
    ) -> VectorClipTensor<
        'l,
        T,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        f32,
        NoTensor,
        { VeOrder::IntraFirst },
        stage::Zipped,
        { Way8 },
    > {
        self.group0.ve_state.merge(self.group1.ve_state);
        self.group0.ve_state.use_alu(op.alu());
        let result = zip_groups(op.binary_op_fn(Some(mode)), &self.group0.inner, &self.group1.inner);
        VectorClipTensor::from_parts(self.ctx, result, self.group1.execution_id, self.group0.ve_state)
    }
}
