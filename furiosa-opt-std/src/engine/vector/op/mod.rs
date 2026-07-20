//! Vector Engine operation types and configurations.
//!
//! This module defines the operation types for the Vector Engine pipeline.
//! Semantic implementations (operation functions) are in `op::semantics.rs`.

mod arg_mode;
mod has_alu;
pub mod semantics;

pub use arg_mode::{ArgMode, BinaryArgMode, TernaryArgMode};
pub use has_alu::HasAlu;
pub use semantics::{HasBinaryOp, HasTernaryOp, HasUnaryOp};

use std::fmt::{self, Display, Formatter};

use super::alu::{FpMulAlu, RngdAlu};
use furiosa_opt_macro::primitive;

// ============================================================================
// Logic cluster
// ============================================================================

/// Logic cluster binary operations for i32 (user-facing, mode-free).
#[primitive(op::LogicBinaryOpI32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicBinaryOpI32 {
    /// Bitwise AND operation.
    BitAnd,
    /// Bitwise OR operation.
    BitOr,
    /// Bitwise XOR operation.
    BitXor,
    /// Left shift operation.
    LeftShift,
    /// Logical right shift operation.
    LogicRightShift,
    /// Arithmetic right shift operation.
    ArithRightShift,
}

impl Display for LogicBinaryOpI32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitAnd => write!(f, "LogicBinaryOpI32::BitAnd"),
            Self::BitOr => write!(f, "LogicBinaryOpI32::BitOr"),
            Self::BitXor => write!(f, "LogicBinaryOpI32::BitXor"),
            Self::LeftShift => write!(f, "LogicBinaryOpI32::LeftShift"),
            Self::LogicRightShift => write!(f, "LogicBinaryOpI32::LogicRightShift"),
            Self::ArithRightShift => write!(f, "LogicBinaryOpI32::ArithRightShift"),
        }
    }
}

impl LogicBinaryOpI32 {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::BitAnd => RngdAlu::LogicAnd,
            Self::BitOr => RngdAlu::LogicOr,
            Self::BitXor => RngdAlu::LogicXor,
            Self::LeftShift => RngdAlu::LogicLshift,
            Self::LogicRightShift | Self::ArithRightShift => RngdAlu::LogicRshift,
        }
    }
}

/// Logic cluster binary operations for f32 (user-facing, mode-free).
#[primitive(op::LogicBinaryOpF32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicBinaryOpF32 {
    /// Bitwise AND operation.
    BitAnd,
    /// Bitwise OR operation.
    BitOr,
    /// Bitwise XOR operation.
    BitXor,
}

impl Display for LogicBinaryOpF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitAnd => write!(f, "LogicBinaryOpF32::BitAnd"),
            Self::BitOr => write!(f, "LogicBinaryOpF32::BitOr"),
            Self::BitXor => write!(f, "LogicBinaryOpF32::BitXor"),
        }
    }
}

impl LogicBinaryOpF32 {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::BitAnd => RngdAlu::LogicAnd,
            Self::BitOr => RngdAlu::LogicOr,
            Self::BitXor => RngdAlu::LogicXor,
        }
    }
}

// ============================================================================
// Fxp cluster
// ============================================================================

/// Fxp cluster binary operations (user-facing, mode-free).
#[primitive(op::FxpBinaryOp)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FxpBinaryOp {
    /// Fixed-point add (wrapping).
    AddFxp,
    /// Fixed-point add (saturating).
    AddFxpSat,
    /// Fixed-point subtract (wrapping).
    SubFxp,
    /// Fixed-point subtract (saturating).
    SubFxpSat,
    /// Left shift operation.
    LeftShift,
    /// Left shift (saturating).
    LeftShiftSat,
    /// Fixed-point multiply.
    MulFxp,
    /// Integer multiply.
    MulInt,
    /// Logical right shift.
    LogicRightShift,
    /// Arithmetic right shift.
    ArithRightShift,
    /// Arithmetic right shift with rounding.
    ArithRightShiftRound,
}

impl Display for FxpBinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddFxp => write!(f, "FxpBinaryOp::AddFxp"),
            Self::AddFxpSat => write!(f, "FxpBinaryOp::AddFxpSat"),
            Self::SubFxp => write!(f, "FxpBinaryOp::SubFxp"),
            Self::SubFxpSat => write!(f, "FxpBinaryOp::SubFxpSat"),
            Self::LeftShift => write!(f, "FxpBinaryOp::LeftShift"),
            Self::LeftShiftSat => write!(f, "FxpBinaryOp::LeftShiftSat"),
            Self::MulFxp => write!(f, "FxpBinaryOp::MulFxp"),
            Self::MulInt => write!(f, "FxpBinaryOp::MulInt"),
            Self::LogicRightShift => write!(f, "FxpBinaryOp::LogicRightShift"),
            Self::ArithRightShift => write!(f, "FxpBinaryOp::ArithRightShift"),
            Self::ArithRightShiftRound => write!(f, "FxpBinaryOp::ArithRightShiftRound"),
        }
    }
}

impl FxpBinaryOp {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::AddFxp | Self::AddFxpSat | Self::SubFxp | Self::SubFxpSat => RngdAlu::FxpAdd,
            Self::LeftShift | Self::LeftShiftSat => RngdAlu::FxpLshift,
            Self::MulFxp | Self::MulInt => RngdAlu::FxpMul,
            Self::LogicRightShift | Self::ArithRightShift | Self::ArithRightShiftRound => RngdAlu::FxpRshift,
        }
    }
}

// ============================================================================
// Fp cluster
// ============================================================================

/// Fp unary operations (user-facing, mode-free).
#[primitive(op::FpUnaryOp)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpUnaryOp {
    /// Exponential function (e^x).
    Exp,
    /// Negative exponential function (e^(-x)).
    NegExp,
    /// Square root.
    Sqrt,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid function.
    Sigmoid,
    /// Error function.
    Erf,
    /// Natural logarithm.
    Log,
    /// Sine function.
    Sin,
    /// Cosine function.
    Cos,
}

impl Display for FpUnaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exp => write!(f, "FpUnaryOp::Exp"),
            Self::NegExp => write!(f, "FpUnaryOp::NegExp"),
            Self::Sqrt => write!(f, "FpUnaryOp::Sqrt"),
            Self::Tanh => write!(f, "FpUnaryOp::Tanh"),
            Self::Sigmoid => write!(f, "FpUnaryOp::Sigmoid"),
            Self::Erf => write!(f, "FpUnaryOp::Erf"),
            Self::Log => write!(f, "FpUnaryOp::Log"),
            Self::Sin => write!(f, "FpUnaryOp::Sin"),
            Self::Cos => write!(f, "FpUnaryOp::Cos"),
        }
    }
}

impl FpUnaryOp {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::Exp | Self::NegExp => RngdAlu::FpExp,
            _ => RngdAlu::FpFpu,
        }
    }
}

/// Fp binary operations (user-facing, mode-free).
#[primitive(op::FpBinaryOp)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpBinaryOp {
    /// Floating-point addition.
    AddF,
    /// Floating-point subtraction.
    SubF,
    /// Floating-point multiplication.
    MulF(FpMulAlu),
    /// Floating-point division.
    DivF,
}

impl FpBinaryOp {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::AddF | Self::SubF => RngdAlu::FpFma,
            Self::MulF(alu) => alu.to_alu(),
            Self::DivF => RngdAlu::FpFpu,
        }
    }
}

/// Fp ternary operations (user-facing, mode-free).
#[primitive(op::FpTernaryOp)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpTernaryOp {
    /// Fused multiply-add: a * b + c.
    FmaF,
}

impl Display for FpTernaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FmaF => write!(f, "FpTernaryOp::FmaF"),
        }
    }
}

impl FpTernaryOp {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        RngdAlu::FpFma
    }
}

// ============================================================================
// Intra-Slice Reduce
// ============================================================================

/// Intra-Slice Reduce operations for i32.
#[primitive(op::IntraSliceReduceOpI32)]
#[derive(Debug, Clone, Copy)]
pub enum IntraSliceReduceOpI32 {
    /// Saturating addition reduction.
    AddSat,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}

impl IntraSliceReduceOpI32 {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        RngdAlu::ReduceAccTree
    }
}

/// Intra-Slice Reduce operations for f32.
#[primitive(op::IntraSliceReduceOpF32)]
#[derive(Debug, Clone, Copy)]
pub enum IntraSliceReduceOpF32 {
    /// Floating-point addition reduction.
    Add,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}

impl IntraSliceReduceOpF32 {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        RngdAlu::ReduceAccTree
    }
}

// ============================================================================
// Inter-Slice Reduce
// ============================================================================

/// Inter-slice reduce operations for i32.
#[primitive(op::InterSliceReduceOpI32)]
#[derive(Debug, Clone, Copy)]
pub enum InterSliceReduceOpI32 {
    /// Addition reduction.
    Add,
    /// Saturating addition reduction.
    AddSat,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}

/// Inter-slice reduce operations for f32.
#[primitive(op::InterSliceReduceOpF32)]
#[derive(Debug, Clone, Copy)]
pub enum InterSliceReduceOpF32 {
    /// Floating-point addition reduction.
    Add,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
    /// Floating-point multiplication reduction.
    Mul,
}

// ============================================================================
// FpDiv
// ============================================================================

/// FpDiv binary operations (user-facing, mode-free).
#[primitive(op::FpDivBinaryOp)]
#[derive(Debug, Clone, Copy)]
pub enum FpDivBinaryOp {
    /// Floating-point division.
    DivF,
}

impl Display for FpDivBinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivF => write!(f, "FpDivBinaryOp::DivF"),
        }
    }
}

impl FpDivBinaryOp {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        RngdAlu::ReduceFpDiv
    }
}

// ============================================================================
// Clip cluster
// ============================================================================

/// Clip binary operations for i32 (user-facing, mode-free).
#[primitive(op::ClipBinaryOpI32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipBinaryOpI32 {
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Absolute minimum value.
    AbsMin,
    /// Absolute maximum value.
    AbsMax,
    /// Fixed-point add (wrapping).
    AddFxp,
    /// Fixed-point add (saturating).
    AddFxpSat,
}

impl Display for ClipBinaryOpI32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Min => write!(f, "ClipBinaryOpI32::Min"),
            Self::Max => write!(f, "ClipBinaryOpI32::Max"),
            Self::AbsMin => write!(f, "ClipBinaryOpI32::AbsMin"),
            Self::AbsMax => write!(f, "ClipBinaryOpI32::AbsMax"),
            Self::AddFxp => write!(f, "ClipBinaryOpI32::AddFxp"),
            Self::AddFxpSat => write!(f, "ClipBinaryOpI32::AddFxpSat"),
        }
    }
}

impl ClipBinaryOpI32 {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::AddFxp | Self::AddFxpSat => RngdAlu::ClipAdd,
            Self::Max | Self::AbsMax => RngdAlu::ClipMax,
            Self::Min | Self::AbsMin => RngdAlu::ClipMin,
        }
    }
}

/// Clip binary operations for f32 (user-facing, mode-free).
#[primitive(op::ClipBinaryOpF32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipBinaryOpF32 {
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Absolute minimum value.
    AbsMin,
    /// Absolute maximum value.
    AbsMax,
    /// Floating-point addition.
    Add,
}

impl Display for ClipBinaryOpF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Min => write!(f, "ClipBinaryOpF32::Min"),
            Self::Max => write!(f, "ClipBinaryOpF32::Max"),
            Self::AbsMin => write!(f, "ClipBinaryOpF32::AbsMin"),
            Self::AbsMax => write!(f, "ClipBinaryOpF32::AbsMax"),
            Self::Add => write!(f, "ClipBinaryOpF32::Add"),
        }
    }
}

impl ClipBinaryOpF32 {
    /// Returns the ALU type for this operation.
    pub(crate) fn alu(&self) -> RngdAlu {
        match self {
            Self::Add => RngdAlu::ClipAdd,
            Self::Max | Self::AbsMax => RngdAlu::ClipMax,
            Self::Min | Self::AbsMin => RngdAlu::ClipMin,
        }
    }
}
