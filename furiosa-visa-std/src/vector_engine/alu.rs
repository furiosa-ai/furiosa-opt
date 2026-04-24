//! ALU definitions for Renegade vector engine.

use std::fmt::{self, Display, Formatter};

use furiosa_mapping_macro::primitive;

/// Renegade ALU units.
///
/// Each ALU can only be used once per VE configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngdAlu {
    // Logic cluster (max 5 ops)
    /// Logic AND ALU
    LogicAnd,
    /// Logic OR ALU
    LogicOr,
    /// Logic XOR ALU
    LogicXor,
    /// Logic left shift ALU
    LogicLshift,
    /// Logic right shift ALU
    LogicRshift,

    // Fxp cluster (max 4 ops)
    /// Fixed-point add ALU
    FxpAdd,
    /// Fixed-point left shift ALU
    FxpLshift,
    /// Fixed-point multiply ALU
    FxpMul,
    /// Fixed-point right shift ALU
    FxpRshift,

    // Float cluster (max 5 ops)
    /// Floating-point FMA ALU
    FpFma,
    /// Floating-point FPU ALU (sqrt, tanh, sigmoid, etc.)
    FpFpu,
    /// Floating-point exp ALU
    FpExp,
    /// Floating-point multiply ALU 0
    FpMul0,
    /// Floating-point multiply ALU 1
    FpMul1,

    // Clip cluster (max 3 ops)
    /// Clip add ALU
    ClipAdd,
    /// Clip max ALU
    ClipMax,
    /// Clip min ALU
    ClipMin,

    // Reduce
    /// Intra-slice reduce accumulator tree ALU
    ReduceAccTree,
    /// Reduce FP division ALU
    ReduceFpDiv,
}

/// Float cluster Mul ALU selection.
///
/// Used to specify which ALU to use for MulF/MaskMulF operations,
/// since they can use FpMul0, FpMul1, or FpFma.
#[primitive(ve::FpMulAlu)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpMulAlu {
    /// Use FpMul0 ALU
    Mul0,
    /// Use FpMul1 ALU
    Mul1,
    /// Use FpFma ALU
    Fma,
}

impl Display for FpMulAlu {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mul0 => write!(f, "FpMulAlu::Mul0"),
            Self::Mul1 => write!(f, "FpMulAlu::Mul1"),
            Self::Fma => write!(f, "FpMulAlu::Fma"),
        }
    }
}

impl FpMulAlu {
    /// Converts to the corresponding RngdAlu.
    pub fn to_alu(self) -> RngdAlu {
        match self {
            FpMulAlu::Mul0 => RngdAlu::FpMul0,
            FpMulAlu::Mul1 => RngdAlu::FpMul1,
            FpMulAlu::Fma => RngdAlu::FpFma,
        }
    }
}
