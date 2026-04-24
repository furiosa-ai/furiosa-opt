//! Semantic implementations for VE operations.
//!
//! This module provides the actual operation logic (apply functions, operation functions)
//! separated from type definitions in `op.rs`.

use super::*;
use crate::prelude::VeScalar;
use crate::scalar::Opt;
use crate::vector_engine::layer::{FpToFxp, FxpToFp};

// ============================================================================
// Operation functions - Logic
// ============================================================================

impl LogicBinaryOpI32 {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            LogicBinaryOpI32::BitAnd => |a, b| a & b,
            LogicBinaryOpI32::BitOr => |a, b| a | b,
            LogicBinaryOpI32::BitXor => |a, b| a ^ b,
            LogicBinaryOpI32::LeftShift => |a, b| a << (b as u32),
            LogicBinaryOpI32::LogicRightShift => |a, b| ((a as u32) >> (b as u32)) as i32,
            LogicBinaryOpI32::ArithRightShift => |a, b| a >> (b as u32),
        }
    }
}

impl LogicBinaryOpF32 {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            LogicBinaryOpF32::BitAnd => |a, b| f32::from_bits(a.to_bits() & b.to_bits()),
            LogicBinaryOpF32::BitOr => |a, b| f32::from_bits(a.to_bits() | b.to_bits()),
            LogicBinaryOpF32::BitXor => |a, b| f32::from_bits(a.to_bits() ^ b.to_bits()),
        }
    }
}

impl LogicOpI {
    /// Returns the binary operation with arg mode applied (Opt version).
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<i32>, Opt<i32>) -> Opt<i32>> {
        let op = self.op.op_fn();
        self.arg_mode.apply_opt(op)
    }
}

impl LogicOpF {
    /// Returns the binary operation with arg mode applied (Opt version).
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>) -> Opt<f32>> {
        let op = self.op.op_fn();
        self.arg_mode.apply_opt(op)
    }
}

// ============================================================================
// Operation functions - Fxp
// ============================================================================

impl FxpBinaryOp {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            FxpBinaryOp::AddFxp => |a, b| a.wrapping_add(b),
            FxpBinaryOp::AddFxpSat => |a, b| a.saturating_add(b),
            FxpBinaryOp::SubFxp => |a, b| a.wrapping_sub(b),
            FxpBinaryOp::SubFxpSat => |a, b| a.saturating_sub(b),
            FxpBinaryOp::LeftShift => |a, b| a << (b as u32),
            FxpBinaryOp::LeftShiftSat => |a, b| a.saturating_mul(1 << (b as u32)),
            FxpBinaryOp::MulFxp => |a, b| {
                // Q31 fixed-point multiply with rounding, matching npu-ir BinOp::MulFxp.
                // Operands are interpreted as Q31 (2^31 ≈ 1.0), so the raw product is
                // shifted right by 31 with a round-to-nearest step. The sole overflow
                // case is MIN × MIN, which saturates to MAX.
                if a == i32::MIN && b == i32::MIN {
                    i32::MAX
                } else {
                    let product = i64::from(a) * i64::from(b);
                    (((product >> 30) + 1) >> 1) as i32
                }
            },
            FxpBinaryOp::MulInt => |a, b| a.wrapping_mul(b),
            FxpBinaryOp::LogicRightShift => |a, b| ((a as u32) >> (b as u32)) as i32,
            FxpBinaryOp::ArithRightShift => |a, b| a >> (b as u32),
            FxpBinaryOp::ArithRightShiftRound => todo!("ArithRightShiftRound not implemented"),
        }
    }
}

impl FxpOp {
    /// Returns the binary operation with arg mode applied (Opt version).
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<i32>, Opt<i32>) -> Opt<i32>> {
        let op = self.op.op_fn();
        self.arg_mode.apply_opt(op)
    }
}

// ============================================================================
// Operation functions - Fp
// ============================================================================

impl FpUnaryOp {
    /// Returns the raw unary operation function.
    pub fn op_fn(&self) -> fn(f32) -> f32 {
        match self {
            FpUnaryOp::Exp => |x| x.exp(),
            FpUnaryOp::NegExp => |x| (-x).exp(),
            FpUnaryOp::Sqrt => |x| x.sqrt(),
            FpUnaryOp::Tanh => |x| x.tanh(),
            FpUnaryOp::Sigmoid => |x| 1.0 / (1.0 + (-x).exp()),
            FpUnaryOp::Erf => |_x| todo!("Erf not implemented"),
            FpUnaryOp::Log => |x| x.ln(),
            FpUnaryOp::Sin => |x| x.sin(),
            FpUnaryOp::Cos => |x| x.cos(),
        }
    }
}

impl FpBinaryOp {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            FpBinaryOp::AddF => |a, b| a + b,
            FpBinaryOp::SubF => |a, b| a - b,
            FpBinaryOp::MulF(_) => |a, b| a * b,
            FpBinaryOp::MaskMulF(_) => |_a, _b| todo!("MaskMulF not implemented"),
            FpBinaryOp::DivF => |a, b| a / b,
        }
    }
}

impl FpTernaryOp {
    /// Returns the raw ternary operation function.
    pub fn op_fn(&self) -> fn(f32, f32, f32) -> f32 {
        match self {
            FpTernaryOp::FmaF => |a, b, c| a.mul_add(b, c),
            FpTernaryOp::MaskFmaF => |_a, _b, _c| todo!("MaskFmaF not implemented"),
        }
    }
}

impl FpOp {
    /// Returns the unary operation with arg mode applied (Opt version).
    /// Panics if not a unary operation.
    pub fn unary_op_opt(&self) -> Box<dyn Fn(Opt<f32>) -> Opt<f32>> {
        match self {
            FpOp::UnaryOp { op, mode } => mode.apply_opt(op.op_fn()),
            _ => panic!("unary_op_opt called on non-unary FpOp"),
        }
    }

    /// Returns the binary operation with arg mode applied (Opt version).
    /// Panics if not a binary operation.
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>) -> Opt<f32>> {
        match self {
            FpOp::BinaryOp { op, mode } => mode.apply_opt(op.op_fn()),
            _ => panic!("binary_op_opt called on non-binary FpOp"),
        }
    }

    /// Returns the ternary operation with arg mode applied (Opt version).
    /// Panics if not a ternary operation.
    pub fn ternary_op_opt(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>, Opt<f32>) -> Opt<f32>> {
        match self {
            FpOp::TernaryOp { op, mode } => mode.apply_opt(op.op_fn()),
            _ => panic!("ternary_op_opt called on non-ternary FpOp"),
        }
    }
}

// ============================================================================
// Operation functions - Clip
// ============================================================================

impl ClipBinaryOpI32 {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            ClipBinaryOpI32::AddFxp => |a, b| a.wrapping_add(b),
            ClipBinaryOpI32::AddFxpSat => |a, b| a.saturating_add(b),
            ClipBinaryOpI32::Min => |a, b| a.min(b),
            ClipBinaryOpI32::Max => |a, b| a.max(b),
            ClipBinaryOpI32::AbsMin => |a, b| if a.abs() < b.abs() { a } else { b },
            ClipBinaryOpI32::AbsMax => |a, b| if a.abs() > b.abs() { a } else { b },
        }
    }
}

impl ClipBinaryOpF32 {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            ClipBinaryOpF32::Add => |a, b| a + b,
            ClipBinaryOpF32::Min => |a, b| a.min(b),
            ClipBinaryOpF32::Max => |a, b| a.max(b),
            ClipBinaryOpF32::AbsMin => |a, b| if a.abs() < b.abs() { a } else { b },
            ClipBinaryOpF32::AbsMax => |a, b| if a.abs() > b.abs() { a } else { b },
        }
    }
}

impl ClipOpI {
    /// Returns the binary operation with arg mode applied (Opt version).
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<i32>, Opt<i32>) -> Opt<i32>> {
        let op = self.op.op_fn();
        self.mode.apply_opt(op)
    }
}

impl ClipOpF {
    /// Returns the binary operation with arg mode applied (Opt version).
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>) -> Opt<f32>> {
        let op = self.op.op_fn();
        self.mode.apply_opt(op)
    }
}

// ============================================================================
// Operation functions - FxpToFp / FpToFxp conversions
// ============================================================================

impl FxpToFp {
    /// Returns the conversion function.
    pub fn op_fn(&self) -> impl Fn(i32) -> f32 {
        let int_width = self.int_width();
        move |x| crate::float::fixedpoint_to_float(x, int_width)
    }
}

impl FpToFxp {
    /// Returns the conversion function.
    pub fn op_fn(&self) -> impl Fn(f32) -> i32 {
        let int_width = self.int_width();
        move |x| crate::float::float_to_fixedpoint(x, int_width)
    }
}

/// Trait for ops that provide conversion operation.
pub trait HasConversionOp<D: VeScalar, D2: VeScalar>: Clone + Copy {
    /// Returns the conversion function.
    fn conversion_op_fn(&self) -> impl Fn(D) -> D2;
}

impl HasConversionOp<i32, f32> for FxpToFp {
    fn conversion_op_fn(&self) -> impl Fn(i32) -> f32 {
        self.op_fn()
    }
}

impl HasConversionOp<f32, i32> for FpToFxp {
    fn conversion_op_fn(&self) -> impl Fn(f32) -> i32 {
        self.op_fn()
    }
}

// ============================================================================
// Operation functions - Intra-Slice Reduce
// ============================================================================

/// Lifts a binary reduction function on D to operate on Opt<D>, treating Uninit as the identity element.
/// TODO: this should be replaced by valid count generator, and Opt<D> should be removed.
fn lift_reduce_fn<D: Copy>(reduce_fn: impl Fn(D, D) -> D + 'static) -> impl Fn(Opt<D>, Opt<D>) -> Opt<D> {
    move |a: Opt<D>, b: Opt<D>| match (a, b) {
        (Opt::Uninit, _) => b,
        (_, Opt::Uninit) => a,
        (Opt::Init(x), Opt::Init(y)) => Opt::Init(reduce_fn(x, y)),
    }
}

impl IntraSliceReduceOpI32 {
    /// Returns the raw binary reduction function.
    pub fn reduce_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            IntraSliceReduceOpI32::AddSat => |a, b| a.saturating_add(b),
            IntraSliceReduceOpI32::Max => |a, b| a.max(b),
            IntraSliceReduceOpI32::Min => |a, b| a.min(b),
        }
    }

    /// Returns a reduction function lifted to [`Opt`], treating `Uninit` as the identity.
    pub fn lifted_reduce_fn(&self) -> Box<dyn Fn(Opt<i32>, Opt<i32>) -> Opt<i32>> {
        Box::new(lift_reduce_fn(self.reduce_fn()))
    }

    /// Returns the identity element for reduction.
    pub fn identity(&self) -> i32 {
        match self {
            IntraSliceReduceOpI32::AddSat => 0,
            IntraSliceReduceOpI32::Max => i32::MIN,
            IntraSliceReduceOpI32::Min => i32::MAX,
        }
    }
}

impl IntraSliceReduceOpF32 {
    /// Returns the raw binary reduction function.
    pub fn reduce_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            IntraSliceReduceOpF32::Add => |a, b| a + b,
            IntraSliceReduceOpF32::Max => |a, b| a.max(b),
            IntraSliceReduceOpF32::Min => |a, b| a.min(b),
        }
    }

    /// Returns a reduction function lifted to [`Opt`], treating `Uninit` as the identity.
    pub fn lifted_reduce_fn(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>) -> Opt<f32>> {
        Box::new(lift_reduce_fn(self.reduce_fn()))
    }

    /// Returns the identity element for reduction.
    pub fn identity(&self) -> f32 {
        match self {
            IntraSliceReduceOpF32::Add => 0.0,
            IntraSliceReduceOpF32::Max => f32::NEG_INFINITY,
            IntraSliceReduceOpF32::Min => f32::INFINITY,
        }
    }
}

// ============================================================================
// Operation functions - Inter-Slice Reduce (VRU)
// ============================================================================

impl InterSliceReduceOpI32 {
    /// Returns the raw binary reduction function.
    pub fn reduce_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            InterSliceReduceOpI32::Add => |a, b| a.wrapping_add(b),
            InterSliceReduceOpI32::AddSat => |a, b| a.saturating_add(b),
            InterSliceReduceOpI32::Max => |a, b| a.max(b),
            InterSliceReduceOpI32::Min => |a, b| a.min(b),
        }
    }

    /// Returns a reduction function lifted to [`Opt`], treating `Uninit` as the identity.
    pub fn lifted_reduce_fn(&self) -> Box<dyn Fn(Opt<i32>, Opt<i32>) -> Opt<i32>> {
        Box::new(lift_reduce_fn(self.reduce_fn()))
    }
}

impl InterSliceReduceOpF32 {
    /// Returns the raw binary reduction function.
    pub fn reduce_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            InterSliceReduceOpF32::Add => |a, b| a + b,
            InterSliceReduceOpF32::Max => |a, b| a.max(b),
            InterSliceReduceOpF32::Min => |a, b| a.min(b),
            InterSliceReduceOpF32::Mul => |a, b| a * b,
        }
    }

    /// Returns a reduction function lifted to [`Opt`], treating `Uninit` as the identity.
    pub fn lifted_reduce_fn(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>) -> Opt<f32>> {
        Box::new(lift_reduce_fn(self.reduce_fn()))
    }
}

// ============================================================================
// Operation functions - FpDiv
// ============================================================================

impl FpDivBinaryOp {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            FpDivBinaryOp::DivF => |a, b| a / b,
        }
    }
}

impl FpDivOp {
    /// Returns the binary operation with arg mode applied (Opt version).
    pub fn binary_op_opt(&self) -> Box<dyn Fn(Opt<f32>, Opt<f32>) -> Opt<f32>> {
        let op = self.op.op_fn();
        self.mode.apply_opt(op)
    }
}

/// Trait for ops that provide unary operation function.
pub trait HasUnaryOp<D>: Clone + Copy {
    /// Returns a function that applies this unary operation with the given mode.
    /// If mode is None, uses the default mode (Mode0).
    fn unary_op_fn(self, mode: Option<UnaryArgMode>) -> impl Fn(Opt<D>) -> Opt<D>;
}

/// Trait for ops that provide binary operation function.
pub trait HasBinaryOp<D>: Clone + Copy {
    /// Returns a function that applies this binary operation with the given mode.
    /// If mode is None, uses the default mode (Mode01).
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<D>, Opt<D>) -> Opt<D>;
}

/// Trait for ops that provide ternary operation function.
pub trait HasTernaryOp<D>: Clone + Copy {
    /// Returns a function that applies this ternary operation.
    fn ternary_op_fn(self, mode: Option<TernaryArgMode>) -> impl Fn(Opt<D>, Opt<D>, Opt<D>) -> Opt<D>;
}

// ============================================================================
// Op implementations
// ============================================================================

impl HasBinaryOp<i32> for LogicBinaryOpI32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<i32>, Opt<i32>) -> Opt<i32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}

impl HasBinaryOp<f32> for LogicBinaryOpF32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<f32>, Opt<f32>) -> Opt<f32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}

impl HasBinaryOp<i32> for FxpBinaryOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<i32>, Opt<i32>) -> Opt<i32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}

impl HasUnaryOp<f32> for FpUnaryOp {
    fn unary_op_fn(self, mode: Option<UnaryArgMode>) -> impl Fn(Opt<f32>) -> Opt<f32> {
        mode.unwrap_or(UnaryArgMode::Mode0).apply_opt(self.op_fn())
    }
}

impl HasBinaryOp<f32> for FpBinaryOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<f32>, Opt<f32>) -> Opt<f32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}

impl HasTernaryOp<f32> for FpTernaryOp {
    fn ternary_op_fn(self, mode: Option<TernaryArgMode>) -> impl Fn(Opt<f32>, Opt<f32>, Opt<f32>) -> Opt<f32> {
        mode.unwrap_or(TernaryArgMode::Mode012).apply_opt(self.op_fn())
    }
}

impl HasBinaryOp<f32> for FpDivBinaryOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<f32>, Opt<f32>) -> Opt<f32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}

impl HasBinaryOp<f32> for FpDivOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<f32>, Opt<f32>) -> Opt<f32> {
        match mode {
            Some(mode) => mode.apply_opt(self.op.op_fn()),
            None => self.binary_op_opt(),
        }
    }
}

impl HasBinaryOp<i32> for ClipBinaryOpI32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<i32>, Opt<i32>) -> Opt<i32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}

impl HasBinaryOp<f32> for ClipBinaryOpF32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(Opt<f32>, Opt<f32>) -> Opt<f32> {
        mode.unwrap_or(BinaryArgMode::Mode01).apply_opt(self.op_fn())
    }
}
