//! Operand types for Vector Engine operations.
//!
//! This module provides types for specifying operands in VE binary and ternary operations:
//! - [`VeRhs`]: RHS operand (constant or VRF data) with type safety
//! - [`StashOperand`]: Stash operand with branch validity (requires matching D type)
//! - [`TernaryOperandTag`]: Operand for ternary operations
//! - [`VeOperand`]: Unified operand type with automatic conversion
//! - [`IntoOperands`]: Trait for converting operands to ArrayVec
//! - [`Stash`]: Type-inferred stash marker (compile-time type checked)

use std::marker::PhantomData;

use furiosa_mapping::{M, Pair, m};
use furiosa_opt_macro::primitive;

use crate::{
    array_vec::ArrayVec,
    engine::vector::{MAX_TAGS, scalar::VeScalar},
    prelude::{GroupId, TagFilter, VrfTensor},
    tensor::Tensor,
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
    pub(crate) tag_filter: TagFilter,
    _phantom: PhantomData<D>,
}

impl<D: VeScalar> StashOperand<D> {
    pub(crate) fn always() -> Self {
        Self {
            tag_filter: TagFilter::All,
            _phantom: PhantomData,
        }
    }

    #[expect(dead_code)]
    pub(crate) fn group(id: GroupId) -> Self {
        Self {
            tag_filter: TagFilter::Group { id },
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// OperandTagValue - Operand carrying branch-gating predicate plus rhs values
// ============================================================================

/// Per-branch operand for VE binary / ternary operations.
///
/// VE operations can configure their **rhs operand(s) per branch id**: e.g.
/// `x < 0 → add(x, 1)`, `x ≥ 0 → add(x, 2)`. Up to two rhs values are supported,
/// stored as `operand0` and `operand1`:
///
/// - `operand0` is the primary rhs and is settable for *both* binary and ternary
///   ops — even unary-shaped invocations may want to override mainstream with a
///   custom rhs (e.g. `exp(0)` instead of `exp(mainstream)`).
/// - `operand1` is only meaningful for ternary ops; the binary alias parameterizes
///   it with `()` so the field carries no extra cost.
///
/// `tag_filter` gates which branch ids actually apply this operand at the
/// position's execution id; positions whose execution id is `Uninit` skip entirely.
///
/// Type aliases [`BinaryOperandTag`] (binary, no `operand1`) and
/// [`TernaryOperandTag`] (ternary, `operand1: f32`) name the two specializations.
#[derive(Debug, Clone)]
pub struct OperandTagValue<D: VeScalar, TargetMapping: M, Operand1: Copy> {
    /// Primary rhs (always present). Replaces mainstream at this branch.
    pub operand0: VeRhs<D, TargetMapping>,
    /// Secondary rhs scalar. `()` for binary, `f32` for ternary.
    pub operand1: Operand1,
    /// Predicate gating which branch ids actually apply this operand.
    pub tag_filter: TagFilter,
}

/// Per-branch operand for binary VE operations: `operand0` only, `operand1 = ()`.
#[primitive(op::BinaryOperandTag)]
pub type BinaryOperandTag<D, TargetMapping> = OperandTagValue<D, TargetMapping, ()>;

/// Per-branch operand for ternary VE operations: `operand0` (rhs) plus `operand1: f32`.
/// Ternary ops are only supported for f32 tensors.
pub type TernaryOperandTag<Mapping> = OperandTagValue<f32, Mapping, f32>;

impl<D: VeScalar, TargetMapping: M> OperandTagValue<D, TargetMapping, ()> {
    /// Creates an always-valid operand.
    #[primitive(op::BinaryOperandTag::always)]
    pub fn always(operand0: VeRhs<D, TargetMapping>) -> Self {
        Self {
            operand0,
            operand1: (),
            tag_filter: TagFilter::All,
        }
    }

    /// Creates a group-specific operand.
    pub fn group(operand0: VeRhs<D, TargetMapping>, id: GroupId) -> Self {
        Self {
            operand0,
            operand1: (),
            tag_filter: TagFilter::Group { id },
        }
    }

    /// Creates an always-valid stash operand.
    pub fn stash_always() -> Self {
        Self {
            operand0: VeRhs::Stash,
            operand1: (),
            tag_filter: TagFilter::All,
        }
    }

    /// Creates a group-specific stash operand.
    pub fn stash_group(id: GroupId) -> Self {
        Self {
            operand0: VeRhs::Stash,
            operand1: (),
            tag_filter: TagFilter::Group { id },
        }
    }

    /// Returns true if this operand uses stash.
    pub fn is_stash(&self) -> bool {
        matches!(self.operand0, VeRhs::Stash)
    }
}

// ============================================================================
// TernaryOperandTag - For ternary operations (f32 only)
// ============================================================================

impl<Mapping: M> OperandTagValue<f32, Mapping, f32> {
    /// Creates a TernaryOperandTag always valid.
    pub fn always(operand0: VeRhs<f32, Mapping>, operand1: f32) -> Self {
        Self {
            operand0,
            operand1,
            tag_filter: TagFilter::All,
        }
    }

    /// Creates a TernaryOperandTag valid for a specific group.
    pub fn group(operand0: VeRhs<f32, Mapping>, operand1: f32, id: GroupId) -> Self {
        Self {
            operand0,
            operand1,
            tag_filter: TagFilter::Group { id },
        }
    }
}

/// Shared view over VE operand types that carry a branch-gating predicate plus the rhs
/// value(s) used at that branch. Implemented by [`BinaryOperandTag`] (one rhs) and
/// [`TernaryOperandTag`] (two rhs) so VE apply helpers can iterate either kind through one
/// code path.
///
/// **Mainstream is not part of this trait.** A ternary op takes three positional inputs
/// (mainstream, operand0, operand1), but the trait only exposes the two rhs values that vary
/// per branch — mainstream is a tensor-level input passed separately to the apply helper.
///
/// `Operand1` is the type of the secondary rhs: `()` for [`BinaryOperandTag`] and `f32` for
/// [`TernaryOperandTag`], so callers that know the concrete type get a typed value without
/// `Option`/`expect`.
pub trait OperandTag<D: VeScalar, Mapping: M> {
    /// Type of the secondary rhs scalar (`()` when the operand kind doesn't carry one).
    type Operand1: Copy;
    /// Primary rhs value (always present).
    fn operand0(&self) -> &VeRhs<D, Mapping>;
    /// Secondary rhs scalar.
    fn operand1(&self) -> Self::Operand1;
    /// Predicate gating whether this operand applies at a given execution id.
    fn tag_filter(&self) -> &TagFilter;
}

impl<D: VeScalar, Mapping: M, Operand1: Copy> OperandTag<D, Mapping> for OperandTagValue<D, Mapping, Operand1> {
    type Operand1 = Operand1;
    fn operand0(&self) -> &VeRhs<D, Mapping> {
        &self.operand0
    }
    fn operand1(&self) -> Operand1 {
        self.operand1
    }
    fn tag_filter(&self) -> &TagFilter {
        &self.tag_filter
    }
}

// From implementations for TernaryOperandTag (enables blanket impl for IntoGroupTernaryOperandTag)

/// `(Into<VeRhs<f32, Mapping>>, f32)` - VeRhs and constant become TernaryOperandTag.
impl<R, Mapping: M> From<(R, f32)> for TernaryOperandTag<Mapping>
where
    R: Into<VeRhs<f32, Mapping>>,
{
    fn from((operand0, operand1): (R, f32)) -> Self {
        TernaryOperandTag::always(operand0.into(), operand1)
    }
}

impl<R, B, Mapping: M> From<((R, f32), B)> for TernaryOperandTag<Mapping>
where
    R: Into<VeRhs<f32, Mapping>>,
    B: Into<TagFilter>,
{
    fn from(((operand0, operand1), branch): ((R, f32), B)) -> Self {
        TernaryOperandTag {
            operand0: operand0.into(),
            operand1,
            tag_filter: branch.into(),
        }
    }
}

// ============================================================================
// IntoTernaryOperandTags trait (for ternary operations, f32 only)
// ============================================================================

/// Trait for converting various operand types into an ArrayVec of TernaryOperandTag.
///
/// # Supported operand types
///
/// - `(f32, f32)` - two constant values (operand0, operand1)
/// - `(VeRhs<f32, Mapping>, f32)` - VeRhs and constant
/// - `TernaryOperandTag<Mapping>` - single ternary operand
/// - `[TernaryOperandTag<Mapping>; N]` - array of ternary operands for multi-branch operations
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
/// // Explicit TernaryOperandTag for branch control
/// tensor.vector_fp_ternary(
///     FpTernaryOp::FmaF,
///     TernaryOperandTag::always(VeRhs::constant(2.0f32), 3.0f32)
/// )
/// ```
pub trait IntoTernaryOperandTags<TargetMapping: M> {
    /// Converts into an ArrayVec of TernaryOperandTag.
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS>;
}

// Blanket impl: Into<TernaryOperandTag> automatically provides IntoTernaryOperandTags
impl<T, TargetMapping: M> IntoTernaryOperandTags<TargetMapping> for T
where
    T: Into<TernaryOperandTag<TargetMapping>>,
{
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
        ArrayVec::new([self.into()])
    }
}

/// Array of `TernaryOperandTag` for multi-branch operations.
impl<TargetMapping: M, const N: usize> IntoTernaryOperandTags<TargetMapping> for [TernaryOperandTag<TargetMapping>; N] {
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
        // Validate: at most one All operand is allowed
        let always_count = self.iter().filter(|op| matches!(op.tag_filter, TagFilter::All)).count();
        assert!(
            always_count <= 1,
            "Multiple All operands are not allowed (found {always_count})"
        );
        ArrayVec::new(self)
    }
}

/// `ArrayVec<TernaryOperandTag, MAX_TAGS>` passes through.
impl<TargetMapping: M> IntoTernaryOperandTags<TargetMapping> for ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
        self
    }
}

// ============================================================================
// From implementations for BinaryOperandTag (enables .into() conversion)
// ============================================================================
//
// These implementations allow ergonomic conversion to BinaryOperandTag using `.into()`.
//
// # Usage for heterogeneous multi-branch operands
//
// When you need multiple operands of different types (e.g., constant + stash),
// use `.into()` to convert each to `BinaryOperandTag`, then pass as array:
//
// ```ignore
// // Single operand (homogeneous) - direct usage
// tensor.vector_fxp(op, 16384i32)
// tensor.vector_fxp(op, Stash)
// tensor.vector_fxp(op, &vrf)
//
// // Multiple operands of same type
// tensor.vector_fxp(op, [
//     BinaryOperandTag::group(VeRhs::constant(100), GroupId::Group0),
//     BinaryOperandTag::group(VeRhs::constant(200), GroupId::Group1),
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
//     BinaryOperandTag::group(VeRhs::constant(100), GroupId::Group0),
//     StashOperand::group(GroupId::Group1).into(),
// ])
// ```

impl<R, D: VeScalar, Mapping: M> From<R> for BinaryOperandTag<D, Mapping>
where
    R: Into<VeRhs<D, Mapping>>,
{
    fn from(rhs: R) -> Self {
        BinaryOperandTag::always(rhs.into())
    }
}

impl<R, B, D: VeScalar, Mapping: M> From<(R, B)> for BinaryOperandTag<D, Mapping>
where
    R: Into<VeRhs<D, Mapping>>,
    B: Into<TagFilter>,
{
    fn from((rhs, branch): (R, B)) -> Self {
        BinaryOperandTag {
            operand0: rhs.into(),
            operand1: (),
            tag_filter: branch.into(),
        }
    }
}

// ============================================================================
// IntoOperands trait - Multiple operands conversion
// ============================================================================

/// Trait for converting various operand types into an ArrayVec.
///
/// Types implementing `Into<BinaryOperandTag>` automatically get this via blanket impl.
/// Array types `[BinaryOperandTag; N]` and `ArrayVec` implement this directly.
///
/// # Supported operand types
///
/// **Single operand** (via `Into<BinaryOperandTag>`, auto-wrapped in ArrayVec):
/// - `i32`, `f32` - constant value
/// - `Stash` - stash read marker
/// - `StashOperand<D>` - stash read with branch validity
/// - `BinaryOperandTag<D, _>` - explicit operand (pass through)
/// - `&VrfTensor<D, ...>` - VRF tensor reference
///
/// **Multiple operands** (direct implementations):
/// - `[BinaryOperandTag<D, _>; N]` - array of operands for multi-branch operations
/// - `ArrayVec<BinaryOperandTag<D, _>, MAX_TAGS>` - pass through
///
/// # Examples
///
/// ```ignore
/// // Single operand - direct usage
/// tensor.vector_fxp(op, 16384i32)
///
/// // Multiple homogeneous operands
/// tensor.vector_fxp(op, [
///     BinaryOperandTag::group(VeRhs::constant(100), GroupId::Group0),
///     BinaryOperandTag::group(VeRhs::constant(200), GroupId::Group1),
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
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS>;
}

// Blanket impl: Into<BinaryOperandTag> automatically provides IntoOperands
impl<T, D: VeScalar, TargetMapping: M> IntoOperands<D, TargetMapping> for T
where
    T: Into<BinaryOperandTag<D, TargetMapping>>,
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        ArrayVec::new([self.into()])
    }
}

impl<D: VeScalar, TargetMapping: M> IntoOperands<D, TargetMapping>
    for ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS>
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        self
    }
}

impl<D: VeScalar, TargetMapping: M, const N: usize> IntoOperands<D, TargetMapping>
    for [BinaryOperandTag<D, TargetMapping>; N]
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        // Validate: at most one All operand is allowed
        let always_count = self
            .iter()
            .filter(|op| matches!(op.tag_filter(), TagFilter::All))
            .count();
        assert!(
            always_count <= 1,
            "Multiple All operands are not allowed (found {always_count})"
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
    /// Converts VeOperand to an ArrayVec of BinaryOperandTag with the target tensor mapping.
    pub fn into_branch_operands<Time: M, Packet: M>(
        self,
    ) -> ArrayVec<BinaryOperandTag<D, m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }]>, MAX_TAGS> {
        type TargetShape<Chip, Cluster, Slice, Time, Packet> =
            m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

        match self {
            VeOperand::Const(v) => ArrayVec::new([BinaryOperandTag::always(VeRhs::Const { v })]),
            VeOperand::Vrf(vrf) => {
                let vrf_operand = VeRhs::<D, TargetShape<Chip, Cluster, Slice, Time, Packet>>::vrf(vrf);
                ArrayVec::new([BinaryOperandTag::always(vrf_operand)])
            }
            VeOperand::Stash(stash) => ArrayVec::new([BinaryOperandTag {
                operand0: VeRhs::Stash,
                operand1: (),
                tag_filter: stash.tag_filter,
            }]),
        }
    }
}

// ============================================================================
// IntoGroupOperand - Ergonomic operand conversion for VectorTensorPair
// ============================================================================

/// Optional per-group operand for VectorTensorPair operations. `None` skips the operation for
/// that group; `Some(operand)` applies it.
pub type GroupOperand<D, Mapping> = Option<BinaryOperandTag<D, Mapping>>;

/// Trait for converting various types into a [`GroupOperand`].
///
/// Uses `Into<BinaryOperandTag>` blanket impl so callers can pass i32/f32/Stash/etc. directly.
///
/// # Supported types
/// - `()` - skip operation for this group (returns `None`)
/// - `i32`, `f32` - constant value (via `Into<BinaryOperandTag>`)
/// - `Stash` - stash read marker (via `Into<BinaryOperandTag>`)
/// - `StashOperand<D>` - stash with branch validity (via `Into<BinaryOperandTag>`)
/// - `BinaryOperandTag<D, Mapping>` - explicit operand (via `Into<BinaryOperandTag>`)
/// - `Option<BinaryOperandTag<D, Mapping>>` - pass through
pub trait IntoGroupOperand<D: VeScalar, Mapping: M> {
    /// Converts into a [`GroupOperand`]. `None` skips the operation for this group.
    fn into_group_operand(self) -> GroupOperand<D, Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for () {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        None
    }
}

/// `Option<BinaryOperandTag<D, Mapping>>` passes through.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for Option<BinaryOperandTag<D, Mapping>> {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        self
    }
}

/// Blanket impl: any type that implements `Into<BinaryOperandTag>` automatically
/// implements `IntoGroupOperand` by wrapping in `Some`.
impl<T, D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for T
where
    T: Into<BinaryOperandTag<D, Mapping>>,
{
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        Some(self.into())
    }
}

// ============================================================================
// IntoGroupTernaryOperandTag - Ergonomic ternary operand conversion for VectorTensorPair
// ============================================================================

/// Type alias for group ternary operand in VectorTensorPair operations.
pub type GroupTernaryOperandTag<Mapping> = Option<TernaryOperandTag<Mapping>>;

/// Trait for converting various types into a group ternary operand.
///
/// Uses `Into<TernaryOperandTag>` blanket impl for automatic conversion from
/// types that implement `From` for `TernaryOperandTag` ((f32, f32), (VeRhs, f32), etc.).
///
/// # Supported types
/// - `()` - skip operation for this group
/// - `(f32, f32)` - two constant values (via `Into<TernaryOperandTag>`)
/// - `(VeRhs<f32, Mapping>, f32)` - VeRhs and constant (via `Into<TernaryOperandTag>`)
/// - `TernaryOperandTag<Mapping>` - explicit ternary operand (via `Into<TernaryOperandTag>`)
/// - `Option<TernaryOperandTag<Mapping>>` - pass through
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
pub trait IntoGroupTernaryOperandTag<Mapping: M> {
    /// Converts into a GroupTernaryOperandTag with the specified mapping.
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<Mapping: M> IntoGroupTernaryOperandTag<Mapping> for () {
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping> {
        None
    }
}

/// `Option<TernaryOperandTag<Mapping>>` passes through.
impl<Mapping: M> IntoGroupTernaryOperandTag<Mapping> for Option<TernaryOperandTag<Mapping>> {
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping> {
        self
    }
}

/// Blanket impl: any type that implements `Into<TernaryOperandTag>` automatically
/// implements `IntoGroupTernaryOperandTag` by wrapping in `Some`.
impl<T, Mapping: M> IntoGroupTernaryOperandTag<Mapping> for T
where
    T: Into<TernaryOperandTag<Mapping>>,
{
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping> {
        Some(self.into())
    }
}
