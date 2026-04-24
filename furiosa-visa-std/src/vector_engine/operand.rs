//! Operand types for Vector Engine operations.
//!
//! This module provides types for specifying operands in VE binary and ternary operations:
//! - [`VeRhs`]: RHS operand (constant or VRF data) with type safety
//! - [`StashOperand`]: Stash operand with branch validity (requires matching D type)
//! - [`TernaryOperand`]: Operand for ternary operations
//! - [`VeOperand`]: Unified operand type with automatic conversion
//! - [`IntoOperands`]: Trait for converting operands to ArrayVec
//! - [`Stash`]: Type-inferred stash marker (compile-time type checked)

use std::marker::PhantomData;

use furiosa_mapping::{M, Pair};
use furiosa_mapping_macro::primitive;

use crate::{
    array_vec::ArrayVec,
    prelude::{GroupId, ValidBranchIds, VrfTensor},
    tensor::Tensor,
    vector_engine::{MAX_BRANCHES, scalar::VeScalar},
};

// ============================================================================
// VeRhs - Constant or VRF operand (type-safe)
// ============================================================================

/// RHS operand for Vector Engine operations.
///
/// Generic over:
/// - `D`: Data type (i32 or f32) - ensures type safety with tensor operations
/// - `TargetMapping`: Target tensor shape for VRF transpose
#[primitive(op::VeRhs)]
#[derive(Debug, Clone)]
pub enum VeRhs<D: VeScalar, TargetMapping: M> {
    /// Constant value.
    Const {
        /// The constant value.
        v: D,
    },
    /// VRF data that has been transposed to match the target tensor shape.
    Vrf {
        /// The transposed VRF tensor.
        data: Tensor<D, TargetMapping>,
    },
    /// Read from stash (previously written value).
    Stash,
}

impl<D: VeScalar, TargetMapping: M> VeRhs<D, TargetMapping> {
    /// Creates a constant operand.
    #[primitive(op::VeRhs::constant)]
    pub fn constant(v: D) -> Self {
        VeRhs::Const { v }
    }

    /// Creates a VeRhs from a VrfTensor, automatically transposing to match the target tensor shape.
    #[primitive(op::VeRhs::vrf)]
    pub fn vrf<Chip: M, Cluster: M, Slice: M, Element: M>(vrf: &VrfTensor<D, Chip, Cluster, Slice, Element>) -> Self {
        let transposed = vrf.inner.transpose::<TargetMapping>(true);
        VeRhs::Vrf { data: transposed }
    }
}

impl<TargetMapping: M> From<i32> for VeRhs<i32, TargetMapping> {
    fn from(v: i32) -> Self {
        VeRhs::Const { v }
    }
}

impl<TargetMapping: M> From<f32> for VeRhs<f32, TargetMapping> {
    fn from(v: f32) -> Self {
        VeRhs::Const { v }
    }
}

impl<D: VeScalar, TargetMapping: M> From<Stash> for VeRhs<D, TargetMapping> {
    fn from(_: Stash) -> Self {
        VeRhs::Stash
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, TargetMapping: M>
    From<&VrfTensor<D, Chip, Cluster, Slice, Element>> for VeRhs<D, TargetMapping>
{
    fn from(vrf: &VrfTensor<D, Chip, Cluster, Slice, Element>) -> Self {
        VeRhs::vrf(vrf)
    }
}

// ============================================================================
// StashOperand - Stash read with branch validity (type-safe)
// ============================================================================

/// Stash operand for Vector Engine operations.
#[derive(Debug, Clone)]
pub struct StashOperand<D: VeScalar> {
    pub(crate) valid_branch_ids: ValidBranchIds,
    _phantom: PhantomData<D>,
}

impl<D: VeScalar> StashOperand<D> {
    pub(crate) fn always() -> Self {
        Self {
            valid_branch_ids: ValidBranchIds::ValidAlways,
            _phantom: PhantomData,
        }
    }

    #[expect(dead_code)]
    pub(crate) fn group(id: GroupId) -> Self {
        Self {
            valid_branch_ids: ValidBranchIds::ValidGroup { id },
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// VeBranchOperand - Operand with branch validity
// ============================================================================

/// Operand with branch validity for multi-operand cases.
///
/// Combines a VeRhs (constant, VRF, or stash) with branch validity.
#[primitive(op::VeBranchOperand)]
#[derive(Debug, Clone)]
pub struct VeBranchOperand<D: VeScalar, TargetMapping: M> {
    /// The operand value.
    pub operand: VeRhs<D, TargetMapping>,
    /// Valid branch IDs for this operand.
    pub valid_branch_ids: ValidBranchIds,
}

impl<D: VeScalar, TargetMapping: M> VeBranchOperand<D, TargetMapping> {
    /// Creates an always-valid operand.
    #[primitive(op::VeBranchOperand::always)]
    pub fn always(operand: VeRhs<D, TargetMapping>) -> Self {
        Self {
            operand,
            valid_branch_ids: ValidBranchIds::ValidAlways,
        }
    }

    /// Creates a group-specific operand.
    pub fn group(operand: VeRhs<D, TargetMapping>, id: GroupId) -> Self {
        Self {
            operand,
            valid_branch_ids: ValidBranchIds::ValidGroup { id },
        }
    }

    /// Creates an always-valid stash operand.
    pub fn stash_always() -> Self {
        Self {
            operand: VeRhs::Stash,
            valid_branch_ids: ValidBranchIds::ValidAlways,
        }
    }

    /// Creates a group-specific stash operand.
    pub fn stash_group(id: GroupId) -> Self {
        Self {
            operand: VeRhs::Stash,
            valid_branch_ids: ValidBranchIds::ValidGroup { id },
        }
    }

    /// Returns true if this operand uses stash.
    pub fn is_stash(&self) -> bool {
        matches!(self.operand, VeRhs::Stash)
    }

    /// Returns the valid branch IDs for this operand.
    pub fn valid_branch_ids(&self) -> &ValidBranchIds {
        &self.valid_branch_ids
    }
}

// ============================================================================
// TernaryOperand - For ternary operations (f32 only)
// ============================================================================

/// User-facing operand for ternary operations.
/// Generic over `Mapping` to match the target tensor's mapping from creation time.
/// Ternary operations are only supported for f32 tensors.
#[derive(Debug, Clone)]
pub struct TernaryOperand<Mapping: M> {
    /// First operand as VeRhs.
    pub operand0: VeRhs<f32, Mapping>,
    /// Second operand as f32.
    pub operand1: f32,
    /// Valid branch IDs.
    pub valid_branch_ids: ValidBranchIds,
}

impl<Mapping: M> TernaryOperand<Mapping> {
    /// Creates a TernaryOperand always valid.
    pub fn always(operand0: VeRhs<f32, Mapping>, operand1: f32) -> Self {
        Self {
            operand0,
            operand1,
            valid_branch_ids: ValidBranchIds::ValidAlways,
        }
    }

    /// Creates a TernaryOperand valid for a specific group.
    pub fn group(operand0: VeRhs<f32, Mapping>, operand1: f32, id: GroupId) -> Self {
        Self {
            operand0,
            operand1,
            valid_branch_ids: ValidBranchIds::ValidGroup { id },
        }
    }
}

// From implementations for TernaryOperand (enables blanket impl for IntoGroupTernaryOperand)

/// `(Into<VeRhs<f32, Mapping>>, f32)` - VeRhs and constant become TernaryOperand.
impl<R, Mapping: M> From<(R, f32)> for TernaryOperand<Mapping>
where
    R: Into<VeRhs<f32, Mapping>>,
{
    fn from((operand0, operand1): (R, f32)) -> Self {
        TernaryOperand::always(operand0.into(), operand1)
    }
}

impl<R, B, Mapping: M> From<((R, f32), B)> for TernaryOperand<Mapping>
where
    R: Into<VeRhs<f32, Mapping>>,
    B: Into<ValidBranchIds>,
{
    fn from(((operand0, operand1), branch): ((R, f32), B)) -> Self {
        TernaryOperand {
            operand0: operand0.into(),
            operand1,
            valid_branch_ids: branch.into(),
        }
    }
}

// ============================================================================
// IntoTernaryOperands trait (for ternary operations, f32 only)
// ============================================================================

/// Trait for converting various operand types into an ArrayVec of TernaryOperand.
///
/// # Supported operand types
///
/// - `(f32, f32)` - two constant values (operand0, operand1)
/// - `(VeRhs<f32, Mapping>, f32)` - VeRhs and constant
/// - `TernaryOperand<Mapping>` - single ternary operand
/// - `[TernaryOperand<Mapping>; N]` - array of ternary operands for multi-branch operations
///
/// # Example
/// ```ignore
/// // Simple usage with tuple (operand0, operand1)
/// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 3.0f32))
///
/// // With VRF as operand0
/// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (&vrf, 3.0f32))
///
/// // With stash as operand0
/// tensor.vector_fp_ternary(FpTernaryOp::FmaF, (Stash, 3.0f32))
///
/// // Explicit TernaryOperand for branch control
/// tensor.vector_fp_ternary(
///     FpTernaryOp::FmaF,
///     TernaryOperand::always(VeRhs::constant(2.0f32), 3.0f32)
/// )
/// ```
pub trait IntoTernaryOperands<TargetMapping: M> {
    /// Converts into an ArrayVec of TernaryOperand.
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperand<TargetMapping>, MAX_BRANCHES>;
}

// Blanket impl: Into<TernaryOperand> automatically provides IntoTernaryOperands
impl<T, TargetMapping: M> IntoTernaryOperands<TargetMapping> for T
where
    T: Into<TernaryOperand<TargetMapping>>,
{
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperand<TargetMapping>, MAX_BRANCHES> {
        ArrayVec::new([self.into()])
    }
}

/// Array of `TernaryOperand` for multi-branch operations.
impl<TargetMapping: M, const N: usize> IntoTernaryOperands<TargetMapping> for [TernaryOperand<TargetMapping>; N] {
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperand<TargetMapping>, MAX_BRANCHES> {
        // Validate: at most one ValidAlways operand is allowed
        let always_count = self
            .iter()
            .filter(|op| matches!(op.valid_branch_ids, ValidBranchIds::ValidAlways))
            .count();
        assert!(
            always_count <= 1,
            "Multiple ValidAlways operands are not allowed (found {always_count})"
        );
        ArrayVec::new(self)
    }
}

/// `ArrayVec<TernaryOperand, MAX_BRANCHES>` passes through.
impl<TargetMapping: M> IntoTernaryOperands<TargetMapping> for ArrayVec<TernaryOperand<TargetMapping>, MAX_BRANCHES> {
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperand<TargetMapping>, MAX_BRANCHES> {
        self
    }
}

// ============================================================================
// From implementations for VeBranchOperand (enables .into() conversion)
// ============================================================================
//
// These implementations allow ergonomic conversion to VeBranchOperand using `.into()`.
//
// # Usage for heterogeneous multi-branch operands
//
// When you need multiple operands of different types (e.g., constant + stash),
// use `.into()` to convert each to `VeBranchOperand`, then pass as array:
//
// ```ignore
// // Single operand (homogeneous) - direct usage
// tensor.vector_fxp(op, 16384i32)
// tensor.vector_fxp(op, Stash)
// tensor.vector_fxp(op, &vrf)
//
// // Multiple operands of same type
// tensor.vector_fxp(op, [
//     VeBranchOperand::group(VeRhs::constant(100), GroupId::Group0),
//     VeBranchOperand::group(VeRhs::constant(200), GroupId::Group1),
// ])
//
// // Multiple operands of different types (heterogeneous)
// // Use .into() to convert each type
// tensor.vector_fxp(op, [
//     16384i32.into(),
//     Stash.into(),
// ])
//
// // With branch control
// tensor.vector_fxp(op, [
//     VeBranchOperand::group(VeRhs::constant(100), GroupId::Group0),
//     StashOperand::group(GroupId::Group1).into(),
// ])
// ```

impl<R, D: VeScalar, Mapping: M> From<R> for VeBranchOperand<D, Mapping>
where
    R: Into<VeRhs<D, Mapping>>,
{
    fn from(rhs: R) -> Self {
        VeBranchOperand::always(rhs.into())
    }
}

impl<R, B, D: VeScalar, Mapping: M> From<(R, B)> for VeBranchOperand<D, Mapping>
where
    R: Into<VeRhs<D, Mapping>>,
    B: Into<ValidBranchIds>,
{
    fn from((rhs, branch): (R, B)) -> Self {
        VeBranchOperand {
            operand: rhs.into(),
            valid_branch_ids: branch.into(),
        }
    }
}

// ============================================================================
// IntoOperands trait - Multiple operands conversion
// ============================================================================

/// Trait for converting various operand types into an ArrayVec.
///
/// Types implementing `Into<VeBranchOperand>` automatically get this via blanket impl.
/// Array types `[VeBranchOperand; N]` and `ArrayVec` implement this directly.
///
/// # Supported operand types
///
/// **Single operand** (via `Into<VeBranchOperand>`, auto-wrapped in ArrayVec):
/// - `i32`, `f32` - constant value
/// - `Stash` - stash read marker
/// - `StashOperand<D>` - stash read with branch validity
/// - `VeBranchOperand<D, _>` - explicit operand (pass through)
/// - `&VrfTensor<D, ...>` - VRF tensor reference
///
/// **Multiple operands** (direct implementations):
/// - `[VeBranchOperand<D, _>; N]` - array of operands for multi-branch operations
/// - `ArrayVec<VeBranchOperand<D, _>, MAX_BRANCHES>` - pass through
///
/// # Examples
///
/// ```ignore
/// // Single operand - direct usage
/// tensor.vector_fxp(op, 16384i32)
///
/// // Multiple homogeneous operands
/// tensor.vector_fxp(op, [
///     VeBranchOperand::group(VeRhs::constant(100), GroupId::Group0),
///     VeBranchOperand::group(VeRhs::constant(200), GroupId::Group1),
/// ])
///
/// // Multiple heterogeneous operands - use .into()
/// tensor.vector_fxp(op, [
///     16384i32.into(),
///     Stash.into(),
/// ])
/// ```
pub trait IntoOperands<D: VeScalar, TargetMapping: M> {
    /// Converts into an ArrayVec of operands.
    fn into_operands(self) -> ArrayVec<VeBranchOperand<D, TargetMapping>, MAX_BRANCHES>;
}

// Blanket impl: Into<VeBranchOperand> automatically provides IntoOperands
impl<T, D: VeScalar, TargetMapping: M> IntoOperands<D, TargetMapping> for T
where
    T: Into<VeBranchOperand<D, TargetMapping>>,
{
    fn into_operands(self) -> ArrayVec<VeBranchOperand<D, TargetMapping>, MAX_BRANCHES> {
        ArrayVec::new([self.into()])
    }
}

impl<D: VeScalar, TargetMapping: M> IntoOperands<D, TargetMapping>
    for ArrayVec<VeBranchOperand<D, TargetMapping>, MAX_BRANCHES>
{
    fn into_operands(self) -> ArrayVec<VeBranchOperand<D, TargetMapping>, MAX_BRANCHES> {
        self
    }
}

impl<D: VeScalar, TargetMapping: M, const N: usize> IntoOperands<D, TargetMapping>
    for [VeBranchOperand<D, TargetMapping>; N]
{
    fn into_operands(self) -> ArrayVec<VeBranchOperand<D, TargetMapping>, MAX_BRANCHES> {
        // Validate: at most one ValidAlways operand is allowed
        let always_count = self
            .iter()
            .filter(|op| matches!(op.valid_branch_ids(), ValidBranchIds::ValidAlways))
            .count();
        assert!(
            always_count <= 1,
            "Multiple ValidAlways operands are not allowed (found {always_count})"
        );
        ArrayVec::new(self)
    }
}

// ============================================================================
// Stash - Type-inferred marker for stash operands (compile-time type checked)
// ============================================================================

/// Type-inferred stash marker for compile-time type checking.
///
/// When used as an operand, the stash data type must match the operation's data type.
///
/// # Example
/// ```ignore
/// // f32 tensor with f32 stash -> OK
/// tensor
///     .vector_stash()
///     .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.0f32)
///     .vector_clip(ClipBinaryOpF32::Max, Stash)  // Compiles: D == StashD == f32
///
/// ```
#[primitive(op::Stash)]
#[derive(Debug, Clone, Copy)]
pub struct Stash;

// ============================================================================
// VeOperand - Unified operand type with automatic conversion
// ============================================================================

/// Unified operand type for Vector Engine operations.
///
/// Supports automatic conversion from:
/// - `D` (i32/f32) - constant value
/// - `&VrfTensor<D, ...>` - VRF tensor reference
/// - `Stash` - stash read (always valid)
///
/// Use with `impl Into<VeOperand<D, ...>>` for ergonomic API:
/// ```ignore
/// .vector_fxp(op, 16384i32)   // i32 auto-converted
/// .vector_fxp(op, &vrf)       // VRF auto-converted
/// .vector_fxp(op, Stash)      // Stash (always valid)
/// ```
#[derive(Debug)]
pub enum VeOperand<'a, D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> {
    /// Constant value (always valid).
    Const(D),
    /// VRF tensor reference.
    Vrf(&'a VrfTensor<D, Chip, Cluster, Slice, VrfMapping>),
    /// Stash operand.
    Stash(StashOperand<D>),
}

// From<i32> for VeOperand<i32, ...>
impl<Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<i32> for VeOperand<'_, i32, Chip, Cluster, Slice, VrfMapping> {
    fn from(v: i32) -> Self {
        VeOperand::Const(v)
    }
}

// From<f32> for VeOperand<f32, ...>
impl<Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<f32> for VeOperand<'_, f32, Chip, Cluster, Slice, VrfMapping> {
    fn from(v: f32) -> Self {
        VeOperand::Const(v)
    }
}

// From<&VrfTensor<D, ...>> for VeOperand<D, ...>
impl<'a, D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M>
    From<&'a VrfTensor<D, Chip, Cluster, Slice, VrfMapping>> for VeOperand<'a, D, Chip, Cluster, Slice, VrfMapping>
{
    fn from(vrf: &'a VrfTensor<D, Chip, Cluster, Slice, VrfMapping>) -> Self {
        VeOperand::Vrf(vrf)
    }
}

// From<StashOperand<D>> for VeOperand<D, ...>
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<StashOperand<D>>
    for VeOperand<'_, D, Chip, Cluster, Slice, VrfMapping>
{
    fn from(stash: StashOperand<D>) -> Self {
        VeOperand::Stash(stash)
    }
}

// From<Stash> for VeOperand<D, ...> - enables using Stash marker directly
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<Stash>
    for VeOperand<'_, D, Chip, Cluster, Slice, VrfMapping>
{
    fn from(_: Stash) -> Self {
        VeOperand::Stash(StashOperand::always())
    }
}

impl<'a, D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> VeOperand<'a, D, Chip, Cluster, Slice, VrfMapping> {
    /// Converts VeOperand to an ArrayVec of VeBranchOperand with the target tensor mapping.
    pub fn into_branch_operands<Time: M, Packet: M>(
        self,
    ) -> ArrayVec<
        VeBranchOperand<D, furiosa_opt_macro::m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }]>,
        MAX_BRANCHES,
    > {
        type TargetShape<Chip, Cluster, Slice, Time, Packet> =
            furiosa_opt_macro::m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

        match self {
            VeOperand::Const(v) => ArrayVec::new([VeBranchOperand::always(VeRhs::Const { v })]),
            VeOperand::Vrf(vrf) => {
                let vrf_operand = VeRhs::<D, TargetShape<Chip, Cluster, Slice, Time, Packet>>::vrf(vrf);
                ArrayVec::new([VeBranchOperand::always(vrf_operand)])
            }
            VeOperand::Stash(stash) => ArrayVec::new([VeBranchOperand {
                operand: VeRhs::Stash,
                valid_branch_ids: stash.valid_branch_ids,
            }]),
        }
    }
}

// ============================================================================
// IntoGroupOperand - Ergonomic operand conversion for VectorTensorPair
// ============================================================================

/// Type alias for group operand in VectorTensorPair operations.
/// Generic over `Mapping` to match the target tensor's mapping from creation time.
pub type GroupOperand<D, Mapping> = Option<VeBranchOperand<D, Mapping>>;

/// Trait for converting various types into a group operand.
///
/// Uses `Into<VeBranchOperand>` blanket impl for automatic conversion from
/// types that implement `From` for `VeBranchOperand` (i32, f32, Stash, etc.).
///
/// # Supported types
/// - `()` - skip operation for this group
/// - `i32`, `f32` - constant value (via `Into<VeBranchOperand>`)
/// - `Stash` - stash read marker (via `Into<VeBranchOperand>`)
/// - `StashOperand<D>` - stash with branch validity (via `Into<VeBranchOperand>`)
/// - `VeBranchOperand<D, Mapping>` - explicit operand (via `Into<VeBranchOperand>`)
/// - `Option<VeBranchOperand<D, Mapping>>` - pass through
pub trait IntoGroupOperand<D: VeScalar, Mapping: M> {
    /// Converts into a GroupOperand with the specified mapping.
    fn into_group_operand(self) -> GroupOperand<D, Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for () {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        None
    }
}

/// `Option<VeBranchOperand<D, Mapping>>` passes through.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for Option<VeBranchOperand<D, Mapping>> {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        self
    }
}

/// Blanket impl: any type that implements `Into<VeBranchOperand>` automatically
/// implements `IntoGroupOperand` by wrapping in `Some`.
impl<T, D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for T
where
    T: Into<VeBranchOperand<D, Mapping>>,
{
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        Some(self.into())
    }
}

// ============================================================================
// IntoGroupTernaryOperand - Ergonomic ternary operand conversion for VectorTensorPair
// ============================================================================

/// Type alias for group ternary operand in VectorTensorPair operations.
pub type GroupTernaryOperand<Mapping> = Option<TernaryOperand<Mapping>>;

/// Trait for converting various types into a group ternary operand.
///
/// Uses `Into<TernaryOperand>` blanket impl for automatic conversion from
/// types that implement `From` for `TernaryOperand` ((f32, f32), (VeRhs, f32), etc.).
///
/// # Supported types
/// - `()` - skip operation for this group
/// - `(f32, f32)` - two constant values (via `Into<TernaryOperand>`)
/// - `(VeRhs<f32, Mapping>, f32)` - VeRhs and constant (via `Into<TernaryOperand>`)
/// - `TernaryOperand<Mapping>` - explicit ternary operand (via `Into<TernaryOperand>`)
/// - `Option<TernaryOperand<Mapping>>` - pass through
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
pub trait IntoGroupTernaryOperand<Mapping: M> {
    /// Converts into a GroupTernaryOperand with the specified mapping.
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<Mapping: M> IntoGroupTernaryOperand<Mapping> for () {
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping> {
        None
    }
}

/// `Option<TernaryOperand<Mapping>>` passes through.
impl<Mapping: M> IntoGroupTernaryOperand<Mapping> for Option<TernaryOperand<Mapping>> {
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping> {
        self
    }
}

/// Blanket impl: any type that implements `Into<TernaryOperand>` automatically
/// implements `IntoGroupTernaryOperand` by wrapping in `Some`.
impl<T, Mapping: M> IntoGroupTernaryOperand<Mapping> for T
where
    T: Into<TernaryOperand<Mapping>>,
{
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping> {
        Some(self.into())
    }
}
