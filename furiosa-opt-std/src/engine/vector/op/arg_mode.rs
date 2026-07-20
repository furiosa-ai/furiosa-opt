//! Argument mode types for VE operations.
//!
//! Defines how operands are mapped to operation arguments.

use std::fmt::{self, Display, Formatter};

use crate::engine::vector::scalar::VeScalar;
use furiosa_opt_macro::primitive;

// ============================================================================
// ArgMode types
// ============================================================================

/// Arg mode: what operand to use as each argument of the operator.
///
/// Unary ops carry no mode — they always run as `op(mainstream)`. The previous "Mode1"
/// (`op(operand0)`) shape is now expressed by adding a dedicated op variant rather than
/// by overloading the unary path with an operand list.
#[derive(Debug, Clone, Copy)]
pub enum ArgMode {
    /// Unary argument mode: `op(mainstream)`.
    Unary,
    /// Binary argument mode.
    Binary(BinaryArgMode),
    /// Ternary argument mode.
    Ternary(TernaryArgMode),
}

/// Binary arg mode.
/// ModeXY: op(argX, argY) where 0=mainstream, 1=operand0
#[primitive(op::BinaryArgMode)]
#[derive(Debug, Clone, Copy)]
pub enum BinaryArgMode {
    /// op(mainstream, mainstream).
    Mode00,
    /// op(mainstream, operand0).
    Mode01,
    /// op(operand0, mainstream).
    Mode10,
    /// op(operand0, operand0).
    Mode11,
}

impl Display for BinaryArgMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode00 => write!(f, "BinaryArgMode::Mode00"),
            Self::Mode01 => write!(f, "BinaryArgMode::Mode01"),
            Self::Mode10 => write!(f, "BinaryArgMode::Mode10"),
            Self::Mode11 => write!(f, "BinaryArgMode::Mode11"),
        }
    }
}

impl BinaryArgMode {
    /// Applies arg mode (operand selection) to a bare binary operation. `Uninit` propagation is the
    /// caller's `zip_with`'s responsibility, this only picks which operands feed `op`.
    pub(crate) fn apply<D: VeScalar>(self, op: impl Fn(D, D) -> D) -> impl Fn(D, D) -> D {
        move |a, b| match self {
            Self::Mode00 => op(a, a),
            Self::Mode01 => op(a, b),
            Self::Mode10 => op(b, a),
            Self::Mode11 => op(b, b),
        }
    }
}

/// Ternary arg mode.
/// ModeXYZ: op(argX, argY, argZ) where 0=mainstream, 1=operand0, 2=operand1
#[derive(Debug, Clone, Copy)]
pub enum TernaryArgMode {
    /// op(mainstream, operand0, operand1).
    Mode012,
    /// op(mainstream, mainstream, operand1).
    Mode002,
    /// op(operand0, mainstream, operand1).
    Mode102,
    /// op(operand0, operand0, operand1).
    Mode112,
    /// op(mainstream, operand1, mainstream).
    Mode020,
    /// op(mainstream, operand1, operand0).
    Mode021,
    /// op(operand0, operand1, mainstream).
    Mode120,
}

impl Display for TernaryArgMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode012 => write!(f, "TernaryArgMode::Mode012"),
            Self::Mode002 => write!(f, "TernaryArgMode::Mode002"),
            Self::Mode102 => write!(f, "TernaryArgMode::Mode102"),
            Self::Mode112 => write!(f, "TernaryArgMode::Mode112"),
            Self::Mode020 => write!(f, "TernaryArgMode::Mode020"),
            Self::Mode021 => write!(f, "TernaryArgMode::Mode021"),
            Self::Mode120 => write!(f, "TernaryArgMode::Mode120"),
        }
    }
}

impl TernaryArgMode {
    /// Applies arg mode (operand selection) to a bare ternary operation. `Uninit` propagation is the
    /// caller's responsibility, this only picks which of `(m, o0, o1)` feed `op`.
    pub(crate) fn apply<D: VeScalar>(self, op: impl Fn(D, D, D) -> D) -> impl Fn(D, D, D) -> D {
        move |m, o0, o1| match self {
            Self::Mode012 => op(m, o0, o1),
            Self::Mode002 => op(m, m, o1),
            Self::Mode102 => op(o0, m, o1),
            Self::Mode112 => op(o0, o0, o1),
            Self::Mode020 => op(m, o1, m),
            Self::Mode021 => op(m, o1, o0),
            Self::Mode120 => op(o0, o1, m),
        }
    }
}
