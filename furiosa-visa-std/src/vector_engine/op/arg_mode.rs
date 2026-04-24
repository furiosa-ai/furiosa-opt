//! Argument mode types for VE operations.
//!
//! Defines how operands are mapped to operation arguments.

use std::fmt::{self, Display, Formatter};

use crate::scalar::Opt;
use crate::vector_engine::scalar::VeScalar;
use furiosa_mapping_macro::primitive;

// ============================================================================
// ArgMode types
// ============================================================================

/// Arg mode: what operand to use as each argument of the operator.
#[derive(Debug, Clone, Copy)]
pub enum ArgMode {
    /// Unary argument mode.
    Unary(UnaryArgMode),
    /// Binary argument mode.
    Binary(BinaryArgMode),
    /// Ternary argument mode.
    Ternary(TernaryArgMode),
}

/// Unary arg mode.
/// Mode0: op(mainstream), Mode1: op(operand0)
#[derive(Debug, Clone, Copy)]
pub enum UnaryArgMode {
    /// Use mainstream as the argument: op(mainstream).
    Mode0,
    /// Use operand0 as the argument: op(operand0).
    Mode1,
}

impl Display for UnaryArgMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode0 => write!(f, "UnaryArgMode::Mode0"),
            Self::Mode1 => write!(f, "UnaryArgMode::Mode1"),
        }
    }
}

impl UnaryArgMode {
    /// Applies arg mode to a unary operation (Opt version).
    pub fn apply_opt<D: VeScalar>(&self, op: impl Fn(D) -> D + 'static) -> Box<dyn Fn(Opt<D>) -> Opt<D>> {
        Box::new(move |x| match x {
            Opt::Init(x) => Opt::Init(op(x)),
            Opt::Uninit => Opt::Uninit,
        })
    }
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
    /// Applies arg mode to a binary operation (Opt version).
    pub fn apply_opt<D: VeScalar>(&self, op: impl Fn(D, D) -> D + 'static) -> Box<dyn Fn(Opt<D>, Opt<D>) -> Opt<D>> {
        match self {
            BinaryArgMode::Mode00 => Box::new(move |a, _b| match a {
                Opt::Init(a) => Opt::Init(op(a, a)),
                Opt::Uninit => Opt::Uninit,
            }),
            BinaryArgMode::Mode01 => Box::new(move |a, b| match (a, b) {
                (Opt::Init(a), Opt::Init(b)) => Opt::Init(op(a, b)),
                _ => Opt::Uninit,
            }),
            BinaryArgMode::Mode10 => Box::new(move |a, b| match (a, b) {
                (Opt::Init(a), Opt::Init(b)) => Opt::Init(op(b, a)),
                _ => Opt::Uninit,
            }),
            BinaryArgMode::Mode11 => Box::new(move |_a, b| match b {
                Opt::Init(b) => Opt::Init(op(b, b)),
                Opt::Uninit => Opt::Uninit,
            }),
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
    /// Applies arg mode to a ternary operation (Opt version).
    pub fn apply_opt<D: VeScalar>(
        &self,
        op: impl Fn(D, D, D) -> D + 'static,
    ) -> Box<dyn Fn(Opt<D>, Opt<D>, Opt<D>) -> Opt<D>> {
        match self {
            TernaryArgMode::Mode012 => Box::new(move |m, o0, o1| match (m, o0, o1) {
                (Opt::Init(m), Opt::Init(o0), Opt::Init(o1)) => Opt::Init(op(m, o0, o1)),
                _ => Opt::Uninit,
            }),
            TernaryArgMode::Mode002 => Box::new(move |m, _o0, o1| match (m, o1) {
                (Opt::Init(m), Opt::Init(o1)) => Opt::Init(op(m, m, o1)),
                _ => Opt::Uninit,
            }),
            TernaryArgMode::Mode102 => Box::new(move |m, o0, o1| match (m, o0, o1) {
                (Opt::Init(m), Opt::Init(o0), Opt::Init(o1)) => Opt::Init(op(o0, m, o1)),
                _ => Opt::Uninit,
            }),
            TernaryArgMode::Mode112 => Box::new(move |_m, o0, o1| match (o0, o1) {
                (Opt::Init(o0), Opt::Init(o1)) => Opt::Init(op(o0, o0, o1)),
                _ => Opt::Uninit,
            }),
            TernaryArgMode::Mode020 => Box::new(move |m, _o0, o1| match (m, o1) {
                (Opt::Init(m), Opt::Init(o1)) => Opt::Init(op(m, o1, m)),
                _ => Opt::Uninit,
            }),
            TernaryArgMode::Mode021 => Box::new(move |m, o0, o1| match (m, o0, o1) {
                (Opt::Init(m), Opt::Init(o0), Opt::Init(o1)) => Opt::Init(op(m, o1, o0)),
                _ => Opt::Uninit,
            }),
            TernaryArgMode::Mode120 => Box::new(move |m, o0, o1| match (m, o0, o1) {
                (Opt::Init(m), Opt::Init(o0), Opt::Init(o1)) => Opt::Init(op(o0, o1, m)),
                _ => Opt::Uninit,
            }),
        }
    }
}
