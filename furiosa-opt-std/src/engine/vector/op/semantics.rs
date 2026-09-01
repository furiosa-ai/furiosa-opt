//! Semantic implementations for VE operations.
//!
//! This module provides the actual operation logic (apply functions, operation functions)
//! separated from type definitions in `op.rs`.

use super::*;
use crate::engine::vector::layer::{FpToFxp, FxpToFp, Reinterpret};
use crate::prelude::VeScalar;

fn rounding_divide_by_pot(value: i32, exponent: i32) -> i32 {
    let exponent = exponent as usize;
    assert!(exponent < i32::BITS as usize);

    let mask = (1_i32 << exponent).wrapping_sub(1);
    let remainder = value & mask;
    let threshold = (mask >> 1) + i32::from(value < 0);

    (value >> exponent) + i32::from(remainder > threshold)
}

fn normalize_float(value: f32) -> f32 {
    if value.is_subnormal() {
        if value.is_sign_positive() { 0.0 } else { -0.0 }
    } else if value.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        value
    }
}

fn apply_float_operation(value: f32, operation: impl FnOnce(f32) -> f32) -> f32 {
    normalize_float(operation(normalize_float(value)))
}

fn erf(value: f32) -> f32 {
    apply_float_operation(value, f32::erf)
}

fn exp(value: f32) -> f32 {
    apply_float_operation(value, f32::exp)
}

fn neg_exp(value: f32) -> f32 {
    apply_float_operation(value, |value| (-value).exp())
}

fn sqrt(value: f32) -> f32 {
    apply_float_operation(value, f32::sqrt)
}

fn tanh(value: f32) -> f32 {
    apply_float_operation(value, f32::tanh)
}

fn sigmoid(value: f32) -> f32 {
    apply_float_operation(value, |value| 1.0 / (1.0 + (-value).exp()))
}

fn log(value: f32) -> f32 {
    apply_float_operation(value, f32::ln)
}

fn sin(value: f32) -> f32 {
    apply_float_operation(value, |value| {
        if value.to_bits() & 0x7fff_ffff > 0x40c9_0fda {
            f32::NAN
        } else {
            value.sin()
        }
    })
}

fn cos(value: f32) -> f32 {
    apply_float_operation(value, |value| {
        if value.to_bits() & 0x7fff_ffff > 0x40c9_0fda {
            f32::NAN
        } else {
            value.cos()
        }
    })
}

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
            Self::ArithRightShiftRound => rounding_divide_by_pot,
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
            Self::Exp => exp,
            Self::NegExp => neg_exp,
            Self::Sqrt => sqrt,
            Self::Tanh => tanh,
            Self::Sigmoid => sigmoid,
            Self::Erf => erf,
            Self::Log => log,
            Self::Sin => sin,
            Self::Cos => cos,
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

/// Unlike [`FxpToFp`] / [`FpToFxp`], this holds the bits and changes only how they are read, in
/// either direction (including `D == D2`, where it is the identity). See
/// [`VeScalar::reinterpret`](crate::prelude::VeScalar::reinterpret).
impl<D: VeScalar, D2: VeScalar> HasConversionOp<D, D2> for Reinterpret {
    fn conversion_op_fn(&self) -> impl Fn(D) -> D2 + Sync {
        |x| x.reinterpret::<D2>()
    }
}

// ============================================================================
// Operation functions - Intra-Slice Reduce
// ============================================================================

impl IntraSliceReduceOpI32 {
    /// Returns the raw binary reduction function.
    pub fn reduce_fn(&self) -> fn(i32, i32) -> i32 {
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
    pub fn reduce_fn(&self) -> fn(f32, f32) -> f32 {
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
    pub fn reduce_fn(&self) -> fn(i32, i32) -> i32 {
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
    pub fn reduce_fn(&self) -> fn(f32, f32) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arith_right_shift_round_uses_nearest_with_ties_away_from_zero() {
        let op = FxpBinaryOp::ArithRightShiftRound.op_fn();

        for (value, expected) in [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 2),
            (7, 2),
            (-1, 0),
            (-2, -1),
            (-3, -1),
            (-4, -1),
            (-5, -1),
            (-6, -2),
            (-7, -2),
            (-8, -2),
        ] {
            assert_eq!(op(value, 2), expected, "value={value}");
        }
    }

    #[test]
    fn arith_right_shift_round_handles_shift_range_boundaries() {
        let op = FxpBinaryOp::ArithRightShiftRound.op_fn();

        assert_eq!(op(123, 0), 123);
        assert_eq!(op(i32::MAX, 31), 1);
        assert_eq!(op(i32::MIN, 31), -1);
    }

    #[test]
    fn erf_matches_reference_values() {
        let op = FpUnaryOp::Erf.op_fn();

        for (value, expected) in [
            (-3.0, -0.999_977_9),
            (-1.0, -0.842_700_8),
            (-0.5, -0.520_499_9),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (3.0, 0.999_977_9),
        ] {
            assert!((op(value) - expected).abs() <= f32::EPSILON, "value={value}");
        }

        assert_eq!(op(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(op(0.0), 0.0);
        assert_eq!(op(f32::NEG_INFINITY), -1.0);
        assert_eq!(op(f32::INFINITY), 1.0);
    }

    #[test]
    fn erf_canonicalizes_nan() {
        let op = FpUnaryOp::Erf.op_fn();

        for bits in [0x7f80_0001, 0x7fc1_2345, 0xff80_0001, 0xffc1_2345] {
            assert_eq!(op(f32::from_bits(bits)).to_bits(), 0x7fc0_0000);
        }
    }

    #[test]
    fn erf_applies_daz_and_preserves_zero_sign() {
        let op = FpUnaryOp::Erf.op_fn();

        for bits in [0x0000_0001, 0x007f_ffff, 0x8000_0001, 0x807f_ffff] {
            let input = f32::from_bits(bits);
            assert_ne!(f32::erf(input).to_bits() & 0x7fff_ffff, 0, "input={bits:#010x}");
            assert_eq!(op(input).to_bits(), bits & 0x8000_0000, "input={bits:#010x}");
        }
    }

    #[test]
    fn fp_unary_ops_apply_daz() {
        let positive = f32::from_bits(0x007f_ffff);
        let negative = f32::from_bits(0x807f_ffff);

        for (op, expected_positive, expected_negative) in [
            (FpUnaryOp::Exp, 1.0, 1.0),
            (FpUnaryOp::NegExp, 1.0, 1.0),
            (FpUnaryOp::Sqrt, 0.0, -0.0),
            (FpUnaryOp::Tanh, 0.0, -0.0),
            (FpUnaryOp::Sigmoid, 0.5, 0.5),
            (FpUnaryOp::Erf, 0.0, -0.0),
            (FpUnaryOp::Log, f32::NEG_INFINITY, f32::NEG_INFINITY),
            (FpUnaryOp::Sin, 0.0, -0.0),
            (FpUnaryOp::Cos, 1.0, 1.0),
        ] {
            let op = op.op_fn();
            assert_eq!(op(positive).to_bits(), expected_positive.to_bits());
            assert_eq!(op(negative).to_bits(), expected_negative.to_bits());
        }
    }

    #[test]
    fn fp_unary_ops_canonicalize_nan() {
        for op in [
            FpUnaryOp::Exp,
            FpUnaryOp::NegExp,
            FpUnaryOp::Sqrt,
            FpUnaryOp::Tanh,
            FpUnaryOp::Sigmoid,
            FpUnaryOp::Erf,
            FpUnaryOp::Log,
            FpUnaryOp::Sin,
            FpUnaryOp::Cos,
        ] {
            assert_eq!(op.op_fn()(f32::from_bits(0xffc1_2345)).to_bits(), 0x7fc0_0000);
        }
    }

    #[test]
    fn exp_uses_x86_rounding_within_hardware_bound() {
        let input = f32::from_bits(0xb54c_da26);
        let exp = FpUnaryOp::Exp.op_fn()(input);
        let neg_exp = FpUnaryOp::NegExp.op_fn()(-input);

        assert_eq!(exp.to_bits(), input.exp().to_bits());
        assert_eq!(neg_exp.to_bits(), input.exp().to_bits());
        assert_eq!(exp.to_bits().abs_diff(0x3f7f_fff4), 1);
        assert_eq!(neg_exp.to_bits().abs_diff(0x3f7f_fff4), 1);
    }

    #[test]
    fn exp_and_sigmoid_apply_ftz() {
        assert_eq!(FpUnaryOp::Exp.op_fn()(-90.0).to_bits(), 0);
        assert_eq!(FpUnaryOp::NegExp.op_fn()(90.0).to_bits(), 0);
        assert_eq!(FpUnaryOp::Sigmoid.op_fn()(-90.0).to_bits(), 0);
    }

    #[test]
    fn sin_and_cos_reject_inputs_outside_hardware_domain() {
        let last_in_domain = f32::from_bits(0x40c9_0fda);
        let first_out_of_domain = f32::from_bits(0x40c9_0fdb);

        assert!(!FpUnaryOp::Sin.op_fn()(last_in_domain).is_nan());
        assert!(!FpUnaryOp::Cos.op_fn()(last_in_domain).is_nan());
        assert_eq!(FpUnaryOp::Sin.op_fn()(first_out_of_domain).to_bits(), 0x7fc0_0000);
        assert_eq!(FpUnaryOp::Cos.op_fn()(-first_out_of_domain).to_bits(), 0x7fc0_0000);
    }
}
