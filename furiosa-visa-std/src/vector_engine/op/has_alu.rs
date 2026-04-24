//! HasAlu trait for ops that provide ALU information.

use super::{
    ClipBinaryOpF32, ClipBinaryOpI32, ClipOpF, ClipOpI, FpBinaryOp, FpDivBinaryOp, FpDivOp, FpTernaryOp, FpUnaryOp,
    FxpBinaryOp, FxpOp, LogicBinaryOpF32, LogicBinaryOpI32, LogicOpF, LogicOpI,
};
use crate::vector_engine::alu::RngdAlu;

/// Trait for ops that provide ALU information.
pub trait HasAlu {
    /// Returns the ALU this operation uses.
    fn alu(&self) -> RngdAlu;
}

// ============================================================================
// HasAlu implementations
// ============================================================================

impl HasAlu for LogicBinaryOpI32 {
    fn alu(&self) -> RngdAlu {
        LogicBinaryOpI32::alu(self)
    }
}

impl HasAlu for LogicBinaryOpF32 {
    fn alu(&self) -> RngdAlu {
        LogicBinaryOpF32::alu(self)
    }
}

impl HasAlu for FxpBinaryOp {
    fn alu(&self) -> RngdAlu {
        FxpBinaryOp::alu(self)
    }
}

impl HasAlu for FpUnaryOp {
    fn alu(&self) -> RngdAlu {
        FpUnaryOp::alu(self)
    }
}

impl HasAlu for FpBinaryOp {
    fn alu(&self) -> RngdAlu {
        FpBinaryOp::alu(self)
    }
}

impl HasAlu for FpTernaryOp {
    fn alu(&self) -> RngdAlu {
        FpTernaryOp::alu(self)
    }
}

impl HasAlu for FpDivBinaryOp {
    fn alu(&self) -> RngdAlu {
        FpDivBinaryOp::alu(self)
    }
}

impl HasAlu for FpDivOp {
    fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl HasAlu for ClipBinaryOpI32 {
    fn alu(&self) -> RngdAlu {
        ClipBinaryOpI32::alu(self)
    }
}

impl HasAlu for ClipBinaryOpF32 {
    fn alu(&self) -> RngdAlu {
        ClipBinaryOpF32::alu(self)
    }
}

// HasAlu implementations for "with mode" types
impl HasAlu for FxpOp {
    fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl HasAlu for LogicOpI {
    fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl HasAlu for LogicOpF {
    fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl HasAlu for ClipOpI {
    fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}

impl HasAlu for ClipOpF {
    fn alu(&self) -> RngdAlu {
        self.op.alu()
    }
}
