//! Vector Engine operation types and configurations.
//!
//! This module defines the operation types for the Vector Engine pipeline.
//! Semantic implementations (operation functions) are in `op::semantics.rs`.

mod arg_mode;
mod has_alu;
pub mod semantics;

pub use arg_mode::{ArgMode, BinaryArgMode, TernaryArgMode, UnaryArgMode};
pub use has_alu::HasAlu;
pub use semantics::{HasBinaryOp, HasTernaryOp, HasUnaryOp};

use std::fmt::{self, Display, Formatter};

use super::alu::{FpMulAlu, RngdAlu};
use furiosa_mapping_macro::primitive;

// ============================================================================
// Common traits
// ============================================================================

/// Common trait for Vector Engine operations.
pub trait VeOperation {
    /// Returns the argument mode for this operation.
    fn arg_mode(&self) -> ArgMode;
}

// ============================================================================
// Logic cluster
// ============================================================================

/// Logic cluster operations for i32 (internal, with mode).
#[derive(Debug, Clone, Copy)]
pub struct LogicOpI {
    pub(crate) op: LogicBinaryOpI32,
    pub(crate) arg_mode: BinaryArgMode,
}

impl LogicOpI {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl VeOperation for LogicOpI {
    fn arg_mode(&self) -> ArgMode {
        ArgMode::Binary(self.arg_mode)
    }
}

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
    /// Converts to LogicOpI with default mode (Mode01).
    pub fn into_logic_op(self) -> LogicOpI {
        self.with_mode(BinaryArgMode::Mode01)
    }

    /// Converts to LogicOpI with the specified arg mode.
    pub fn with_mode(self, mode: BinaryArgMode) -> LogicOpI {
        LogicOpI {
            op: self,
            arg_mode: mode,
        }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        match self {
            Self::BitAnd => RngdAlu::LogicAnd,
            Self::BitOr => RngdAlu::LogicOr,
            Self::BitXor => RngdAlu::LogicXor,
            Self::LeftShift => RngdAlu::LogicLshift,
            Self::LogicRightShift | Self::ArithRightShift => RngdAlu::LogicRshift,
        }
    }
}

/// Logic cluster operations for f32 (internal, with mode).
#[derive(Debug, Clone, Copy)]
pub struct LogicOpF {
    pub(crate) op: LogicBinaryOpF32,
    pub(crate) arg_mode: BinaryArgMode,
}

impl LogicOpF {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl VeOperation for LogicOpF {
    fn arg_mode(&self) -> ArgMode {
        ArgMode::Binary(self.arg_mode)
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
    /// Converts to LogicOpF with default mode (Mode01).
    pub fn into_logic_op(self) -> LogicOpF {
        self.with_mode(BinaryArgMode::Mode01)
    }

    /// Converts to LogicOpF with the specified arg mode.
    pub fn with_mode(self, mode: BinaryArgMode) -> LogicOpF {
        LogicOpF {
            op: self,
            arg_mode: mode,
        }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
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

/// Fxp cluster operations (internal, with mode).
#[derive(Debug, Clone, Copy)]
pub struct FxpOp {
    pub(crate) op: FxpBinaryOp,
    pub(crate) arg_mode: BinaryArgMode,
}

impl FxpOp {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl VeOperation for FxpOp {
    fn arg_mode(&self) -> ArgMode {
        ArgMode::Binary(self.arg_mode)
    }
}

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
    /// Converts to FxpOp with default mode (Mode01).
    pub fn into_fxp_op(self) -> FxpOp {
        self.with_mode(BinaryArgMode::Mode01)
    }

    /// Converts to FxpOp with the specified arg mode.
    pub fn with_mode(self, mode: BinaryArgMode) -> FxpOp {
        FxpOp {
            op: self,
            arg_mode: mode,
        }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
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

/// Fp cluster operations (internal, with mode).
#[derive(Debug, Clone)]
pub enum FpOp {
    /// Unary fp operation with op and mode.
    UnaryOp {
        /// The unary operation.
        op: FpUnaryOp,
        /// The argument mode.
        mode: UnaryArgMode,
    },
    /// Binary fp operation with op and mode.
    BinaryOp {
        /// The binary operation.
        op: FpBinaryOp,
        /// The argument mode.
        mode: BinaryArgMode,
    },
    /// Ternary fp operation with op and mode.
    TernaryOp {
        /// The ternary operation.
        op: FpTernaryOp,
        /// The argument mode.
        mode: TernaryArgMode,
    },
}

impl FpOp {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        match self {
            FpOp::UnaryOp { op, .. } => op.alu(),
            FpOp::BinaryOp { op, .. } => op.alu(),
            FpOp::TernaryOp { op, .. } => op.alu(),
        }
    }
}

impl VeOperation for FpOp {
    fn arg_mode(&self) -> ArgMode {
        match self {
            FpOp::UnaryOp { mode, .. } => ArgMode::Unary(*mode),
            FpOp::BinaryOp { mode, .. } => ArgMode::Binary(*mode),
            FpOp::TernaryOp { mode, .. } => ArgMode::Ternary(*mode),
        }
    }
}

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
    /// Converts to FpOp with default mode (Mode0).
    pub fn into_fp_op(self) -> FpOp {
        self.with_mode(UnaryArgMode::Mode0)
    }

    /// Converts to FpOp with the specified arg mode.
    pub fn with_mode(self, mode: UnaryArgMode) -> FpOp {
        FpOp::UnaryOp { op: self, mode }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
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
    /// Masked floating-point multiplication.
    MaskMulF(FpMulAlu),
    /// Floating-point division.
    DivF,
}

impl FpBinaryOp {
    /// Converts to FpOp with default mode (Mode01).
    pub fn into_fp_op(self) -> FpOp {
        self.with_mode(BinaryArgMode::Mode01)
    }

    /// Converts to FpOp with the specified arg mode.
    pub fn with_mode(self, mode: BinaryArgMode) -> FpOp {
        FpOp::BinaryOp { op: self, mode }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        match self {
            Self::AddF | Self::SubF => RngdAlu::FpFma,
            Self::MulF(alu) | Self::MaskMulF(alu) => alu.to_alu(),
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
    /// Masked fused multiply-add.
    MaskFmaF,
}

impl Display for FpTernaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FmaF => write!(f, "FpTernaryOp::FmaF"),
            Self::MaskFmaF => write!(f, "FpTernaryOp::MaskFmaF"),
        }
    }
}

impl FpTernaryOp {
    /// Converts to FpOp with default mode (Mode012).
    pub fn into_fp_op(self) -> FpOp {
        self.with_mode(TernaryArgMode::Mode012)
    }

    /// Converts to FpOp with the specified arg mode.
    pub fn with_mode(self, mode: TernaryArgMode) -> FpOp {
        FpOp::TernaryOp { op: self, mode }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
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
    pub fn alu(&self) -> RngdAlu {
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
    pub fn alu(&self) -> RngdAlu {
        RngdAlu::ReduceAccTree
    }
}

// ============================================================================
// Inter-Slice Reduce (VRU)
// ============================================================================

/// Inter-slice reduce operations for i32 (VRU).
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

/// Inter-slice reduce operations for f32 (VRU).
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

/// Floating Point Division operation (internal, with mode).
#[primitive(op::FpDivOp)]
#[derive(Debug, Clone, Copy)]
pub struct FpDivOp {
    pub(crate) op: FpDivBinaryOp,
    pub(crate) mode: BinaryArgMode,
}

impl FpDivOp {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl VeOperation for FpDivOp {
    fn arg_mode(&self) -> ArgMode {
        ArgMode::Binary(self.mode)
    }
}

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
    /// Converts to FpDivOp with the specified arg mode.
    #[primitive(op::FpDivBinaryOp::with_mode)]
    pub fn with_mode(self, mode: BinaryArgMode) -> FpDivOp {
        FpDivOp { op: self, mode }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        RngdAlu::ReduceFpDiv
    }
}

// ============================================================================
// Clip cluster
// ============================================================================

/// Clip cluster operations for i32 (internal, with mode).
#[derive(Debug, Clone, Copy)]
pub struct ClipOpI {
    pub(crate) op: ClipBinaryOpI32,
    pub(crate) mode: BinaryArgMode,
}

impl ClipOpI {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl VeOperation for ClipOpI {
    fn arg_mode(&self) -> ArgMode {
        ArgMode::Binary(self.mode)
    }
}

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
    /// Converts to ClipOpI with default mode (Mode01).
    pub fn into_clip_op(self) -> ClipOpI {
        self.with_mode(BinaryArgMode::Mode01)
    }

    /// Converts to ClipOpI with the specified arg mode.
    pub fn with_mode(self, mode: BinaryArgMode) -> ClipOpI {
        ClipOpI { op: self, mode }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        match self {
            Self::AddFxp | Self::AddFxpSat => RngdAlu::ClipAdd,
            Self::Max | Self::AbsMax => RngdAlu::ClipMax,
            Self::Min | Self::AbsMin => RngdAlu::ClipMin,
        }
    }
}

/// Clip cluster operations for f32 (internal, with mode).
#[derive(Debug, Clone, Copy)]
pub struct ClipOpF {
    pub(crate) op: ClipBinaryOpF32,
    pub(crate) mode: BinaryArgMode,
}

impl ClipOpF {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl VeOperation for ClipOpF {
    fn arg_mode(&self) -> ArgMode {
        ArgMode::Binary(self.mode)
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
    /// Converts to ClipOpF with default mode (Mode01).
    pub fn into_clip_op(self) -> ClipOpF {
        self.with_mode(BinaryArgMode::Mode01)
    }

    /// Converts to ClipOpF with the specified arg mode.
    pub fn with_mode(self, mode: BinaryArgMode) -> ClipOpF {
        ClipOpF { op: self, mode }
    }

    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        match self {
            Self::Add => RngdAlu::ClipAdd,
            Self::Max | Self::AbsMax => RngdAlu::ClipMax,
            Self::Min | Self::AbsMin => RngdAlu::ClipMin,
        }
    }
}

// ============================================================================
// Unified VeOp enum
// ============================================================================

/// Vector Engine operations (unified enum for runtime storage).
#[derive(Debug, Clone)]
pub enum VeOp {
    /// Logic operations (integer)
    LogicOpI(LogicOpI),
    /// Logic operations (fp)
    LogicOpF(LogicOpF),
    /// Fxp operations
    FxpOp(FxpOp),
    /// Fp operations
    FpOp(FpOp),
    /// Intra-Slice Reduce operations (integer)
    IntraSliceReduceOpI32(IntraSliceReduceOpI32),
    /// Intra-Slice Reduce operations (fp)
    IntraSliceReduceOpF32(IntraSliceReduceOpF32),
    /// Fp Division operation
    FpDivOp(FpDivOp),
    /// Clip operations (integer)
    ClipOpI(ClipOpI),
    /// Clip operations (fp)
    ClipOpF(ClipOpF),
}

impl VeOp {
    /// Returns the ALU type for this operation.
    pub fn alu(&self) -> RngdAlu {
        match self {
            VeOp::LogicOpI(op) => op.alu(),
            VeOp::LogicOpF(op) => op.alu(),
            VeOp::FxpOp(op) => op.alu(),
            VeOp::FpOp(op) => op.alu(),
            VeOp::IntraSliceReduceOpI32(op) => op.alu(),
            VeOp::IntraSliceReduceOpF32(op) => op.alu(),
            VeOp::FpDivOp(op) => op.alu(),
            VeOp::ClipOpI(op) => op.alu(),
            VeOp::ClipOpF(op) => op.alu(),
        }
    }
}

impl VeOperation for VeOp {
    fn arg_mode(&self) -> ArgMode {
        match self {
            VeOp::LogicOpI(op) => op.arg_mode(),
            VeOp::LogicOpF(op) => op.arg_mode(),
            VeOp::FxpOp(op) => op.arg_mode(),
            VeOp::FpOp(op) => op.arg_mode(),
            VeOp::IntraSliceReduceOpI32(_) | VeOp::IntraSliceReduceOpF32(_) => ArgMode::Unary(UnaryArgMode::Mode0),
            VeOp::FpDivOp(op) => op.arg_mode(),
            VeOp::ClipOpI(op) => op.arg_mode(),
            VeOp::ClipOpF(op) => op.arg_mode(),
        }
    }
}
