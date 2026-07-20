//! Semantic implementations for VE operations.
//!
//! This module provides the actual operation logic (apply functions, operation functions)
//! separated from type definitions in `op.rs`.

use super::*;
use crate::engine::vector::layer::{FpToFxp, FxpToFp};
use crate::prelude::VeScalar;

// ============================================================================
// Operation functions - Logic
// ============================================================================

impl LogicBinaryOpI32 {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            Self::BitAnd => |a, b| a & b,
            Self::BitOr => |a, b| a | b,
            Self::BitXor => |a, b| a ^ b,
            Self::LeftShift => |a, b| a << (b as u32),
            Self::LogicRightShift => |a, b| ((a as u32) >> (b as u32)) as i32,
            Self::ArithRightShift => |a, b| a >> (b as u32),
        }
    }
}

impl LogicBinaryOpF32 {
    /// Returns the raw binary operation function.
    pub(crate) fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            Self::BitAnd => |a, b| f32::from_bits(a.to_bits() & b.to_bits()),
            Self::BitOr => |a, b| f32::from_bits(a.to_bits() | b.to_bits()),
            Self::BitXor => |a, b| f32::from_bits(a.to_bits() ^ b.to_bits()),
        }
    }
}

// ============================================================================
// Operation functions - Fxp
// ============================================================================

impl FxpBinaryOp {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            Self::AddFxp => |a, b| a.wrapping_add(b),
            Self::AddFxpSat => |a, b| a.saturating_add(b),
            Self::SubFxp => |a, b| a.wrapping_sub(b),
            Self::SubFxpSat => |a, b| a.saturating_sub(b),
            Self::LeftShift => |a, b| a << (b as u32),
            Self::LeftShiftSat => |a, b| a.saturating_mul(1 << (b as u32)),
            Self::MulFxp => |a, b| {
                // Q31 fixed-point multiply with rounding (hardware MulFxp).
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
            Self::MulInt => |a, b| a.wrapping_mul(b),
            Self::LogicRightShift => |a, b| ((a as u32) >> (b as u32)) as i32,
            Self::ArithRightShift => |a, b| a >> (b as u32),
            Self::ArithRightShiftRound => todo!("ArithRightShiftRound not implemented"),
        }
    }
}

// ============================================================================
// Operation functions - Fp
// ============================================================================

impl FpUnaryOp {
    /// Returns the raw unary operation function.
    pub fn op_fn(&self) -> fn(f32) -> f32 {
        match self {
            Self::Exp => |x| x.exp(),
            Self::NegExp => |x| (-x).exp(),
            Self::Sqrt => |x| x.sqrt(),
            Self::Tanh => |x| x.tanh(),
            Self::Sigmoid => |x| 1.0 / (1.0 + (-x).exp()),
            Self::Erf => |_x| todo!("Erf not implemented"),
            Self::Log => |x| x.ln(),
            Self::Sin => |x| x.sin(),
            Self::Cos => |x| x.cos(),
        }
    }
}

impl FpBinaryOp {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            Self::AddF => |a, b| a + b,
            Self::SubF => |a, b| a - b,
            Self::MulF(_) => |a, b| a * b,
            Self::DivF => |a, b| a / b,
        }
    }
}

impl FpTernaryOp {
    /// Returns the raw ternary operation function.
    pub fn op_fn(&self) -> fn(f32, f32, f32) -> f32 {
        match self {
            Self::FmaF => |a, b, c| a.mul_add(b, c),
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
            Self::AddFxp => |a, b| a.wrapping_add(b),
            Self::AddFxpSat => |a, b| a.saturating_add(b),
            Self::Min => |a, b| a.min(b),
            Self::Max => |a, b| a.max(b),
            Self::AbsMin => |a, b| if a.abs() < b.abs() { a } else { b },
            Self::AbsMax => |a, b| if a.abs() > b.abs() { a } else { b },
        }
    }
}

impl ClipBinaryOpF32 {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            Self::Add => |a, b| a + b,
            Self::Min => |a, b| a.min(b),
            Self::Max => |a, b| a.max(b),
            Self::AbsMin => |a, b| if a.abs() < b.abs() { a } else { b },
            Self::AbsMax => |a, b| if a.abs() > b.abs() { a } else { b },
        }
    }
}

// ============================================================================
// Operation functions - FxpToFp / FpToFxp conversions
// ============================================================================

impl FxpToFp {
    /// Returns the conversion function.
    pub(crate) fn op_fn(&self) -> impl Fn(i32) -> f32 + Sync {
        let int_width = self.int_width();
        move |x| crate::float::fixedpoint_to_float(x, int_width)
    }
}

impl FpToFxp {
    /// Returns the conversion function.
    pub(crate) fn op_fn(&self) -> impl Fn(f32) -> i32 + Sync {
        let int_width = self.int_width();
        move |x| crate::float::float_to_fixedpoint(x, int_width)
    }
}

/// Trait for ops that provide conversion operation.
pub trait HasConversionOp<D: VeScalar, D2: VeScalar>: Clone + Copy {
    /// Returns the conversion function.
    fn conversion_op_fn(&self) -> impl Fn(D) -> D2 + Sync;
}

impl HasConversionOp<i32, f32> for FxpToFp {
    fn conversion_op_fn(&self) -> impl Fn(i32) -> f32 + Sync {
        self.op_fn()
    }
}

impl HasConversionOp<f32, i32> for FpToFxp {
    fn conversion_op_fn(&self) -> impl Fn(f32) -> i32 + Sync {
        self.op_fn()
    }
}

// ============================================================================
// Operation functions - Intra-Slice Reduce
// ============================================================================

impl IntraSliceReduceOpI32 {
    /// Returns the raw binary reduction function.
    pub(crate) fn reduce_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            Self::AddSat => |a, b| a.saturating_add(b),
            Self::Max => |a, b| a.max(b),
            Self::Min => |a, b| a.min(b),
        }
    }

    /// Returns the identity element for reduction.
    pub(crate) fn identity(&self) -> i32 {
        match self {
            Self::AddSat => 0,
            Self::Max => i32::MIN,
            Self::Min => i32::MAX,
        }
    }
}

impl IntraSliceReduceOpF32 {
    /// Returns the raw binary reduction function.
    pub(crate) fn reduce_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            Self::Add => |a, b| a + b,
            Self::Max => |a, b| a.max(b),
            Self::Min => |a, b| a.min(b),
        }
    }

    /// Returns the identity element for reduction.
    pub(crate) fn identity(&self) -> f32 {
        match self {
            Self::Add => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
        }
    }
}

// ============================================================================
// Operation functions - Inter-Slice Reduce
// ============================================================================

impl InterSliceReduceOpI32 {
    /// Returns the raw binary reduction function.
    pub(crate) fn reduce_fn(&self) -> fn(i32, i32) -> i32 {
        match self {
            Self::Add => |a, b| a.wrapping_add(b),
            Self::AddSat => |a, b| a.saturating_add(b),
            Self::Max => |a, b| a.max(b),
            Self::Min => |a, b| a.min(b),
        }
    }

    /// Returns the identity element for reduction.
    pub(crate) fn identity(&self) -> i32 {
        match self {
            Self::Add | Self::AddSat => 0,
            Self::Max => i32::MIN,
            Self::Min => i32::MAX,
        }
    }
}

impl InterSliceReduceOpF32 {
    /// Returns the raw binary reduction function.
    pub(crate) fn reduce_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            Self::Add => |a, b| a + b,
            Self::Max => |a, b| a.max(b),
            Self::Min => |a, b| a.min(b),
            Self::Mul => |a, b| a * b,
        }
    }

    /// Returns the identity element for reduction.
    pub(crate) fn identity(&self) -> f32 {
        match self {
            Self::Add => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
            Self::Mul => 1.0,
        }
    }
}

// ============================================================================
// Operation functions - FpDiv
// ============================================================================

impl FpDivBinaryOp {
    /// Returns the raw binary operation function.
    pub fn op_fn(&self) -> fn(f32, f32) -> f32 {
        match self {
            Self::DivF => |a, b| a / b,
        }
    }
}

/// Trait for ops that provide unary operation function.
pub trait HasUnaryOp<D>: Clone + Copy {
    /// Returns a function that applies this unary operation to mainstream values.
    fn unary_op_fn(self) -> impl Fn(D) -> D + Sync;
}

/// Trait for ops that provide binary operation function.
pub trait HasBinaryOp<D>: Clone + Copy {
    /// Returns a function that applies this binary operation with the given mode.
    /// If mode is None, uses the default mode (Mode01).
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(D, D) -> D + Sync;
}

/// Trait for ops that provide ternary operation function.
pub trait HasTernaryOp<D>: Clone + Copy {
    /// Returns a function that applies this ternary operation.
    fn ternary_op_fn(self, mode: Option<TernaryArgMode>) -> impl Fn(D, D, D) -> D + Sync;
}

// ============================================================================
// Op implementations
// ============================================================================

impl HasBinaryOp<i32> for LogicBinaryOpI32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(i32, i32) -> i32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}

impl HasBinaryOp<f32> for LogicBinaryOpF32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(f32, f32) -> f32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}

impl HasBinaryOp<i32> for FxpBinaryOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(i32, i32) -> i32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}

impl HasUnaryOp<f32> for FpUnaryOp {
    fn unary_op_fn(self) -> impl Fn(f32) -> f32 + Sync {
        self.op_fn()
    }
}

impl HasBinaryOp<f32> for FpBinaryOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(f32, f32) -> f32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}

impl HasTernaryOp<f32> for FpTernaryOp {
    fn ternary_op_fn(self, mode: Option<TernaryArgMode>) -> impl Fn(f32, f32, f32) -> f32 + Sync {
        mode.unwrap_or(TernaryArgMode::Mode012).apply(self.op_fn())
    }
}

impl HasBinaryOp<f32> for FpDivBinaryOp {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(f32, f32) -> f32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}

impl HasBinaryOp<i32> for ClipBinaryOpI32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(i32, i32) -> i32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}

impl HasBinaryOp<f32> for ClipBinaryOpF32 {
    fn binary_op_fn(self, mode: Option<BinaryArgMode>) -> impl Fn(f32, f32) -> f32 + Sync {
        mode.unwrap_or(BinaryArgMode::Mode01).apply(self.op_fn())
    }
}
