//! Vector Engine — branched/staged scalar pipeline on the post-Contraction
//! stream.

pub mod alu;
pub mod branch;
pub mod layer;
pub mod op;
pub mod operand;
pub mod scalar;
pub mod stage;
pub mod tensor;

/// Maximum number of concurrent branches the Vector Engine supports.
pub const MAX_TAGS: usize = 5;
