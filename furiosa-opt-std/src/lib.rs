//! Virtual ISA standard library.

#![expect(incomplete_features)]
#![feature(adt_const_params)]
#![feature(inherent_associated_types)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]
#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![allow(clippy::disallowed_types)]

mod array_vec;
mod cast;
mod context;
pub mod diag;
mod engine;
mod float;
mod scalar;
mod tensor;
mod tensor_state;

pub mod runtime;

#[doc(hidden)]
pub use tensor::{BufRawTensor, MathRawTensor, PhantomRawTensor};
pub use tokio::sync::OnceCell;

pub use tensor::BufferConvertError;

pub use tensor::pseudo;

// FIXME: `Cast` is exported at crate root (not just prelude) to help rust-analyzer
// resolve the `Cast` trait bound in tensor's `cast()` methods.
pub use cast::Cast;

/// Prelude module that re-exports commonly used items.
pub mod prelude {
    pub use super::cast::{ContractionCast, FetchCast};
    pub use super::engine::*;
    pub use super::tensor::memory::*;
    pub use super::tensor::tu::*;
    pub use super::{array_vec::*, context::*, runtime::*, scalar::*, tensor::*};
    // Stage types are accessed via stage::Init, stage::Tag, etc.
    pub use super::engine::vector::{alu::*, branch::*, layer::*, op::*, operand::*, scalar::*, stage, tensor::*};
    pub use super::tensor_state::*;
    pub use furiosa_mapping::*;
    pub use furiosa_opt_macro::device;
}
