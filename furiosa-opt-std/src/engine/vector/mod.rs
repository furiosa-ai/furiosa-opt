//! Vector Engine: branched/staged scalar pipeline on the post-Contraction stream.
//!
//! A kernel reaches this through the prelude. The names below are what it actually writes: the tag
//! mode that produces execution ids, the guard vocabulary those ids are tested against, and the
//! operand builder that says which slot applies where.

pub mod alu;
pub mod branch;
pub mod layer;
pub mod op;
pub mod operand;
pub mod scalar;
pub mod stage;
pub mod stash_slot;
pub mod tensor;

pub use branch::{BitReq, TagGuard, TagMode};
pub use operand::{Branched, Stash};
