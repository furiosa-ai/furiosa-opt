//! Virtual ISA standard library.

#![expect(incomplete_features)]
#![feature(adt_const_params)]
#![feature(inherent_associated_types)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]
#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![register_tool(tcp)]

mod array_vec;
mod cast;
mod context;
pub mod diag;
mod float;
mod memory_tensor;
mod raw_tensor;
mod scalar;
mod stream_tensor;
mod tensor;
mod tensor_state;
mod tensor_view;
mod vector_engine;

pub mod runtime;

pub use tokio::sync::OnceCell;

// FIXME: `Cast` is exported at crate root (not just prelude) to help rust-analyzer
// resolve the `Cast` trait bound in tensor's `cast()` methods.
pub use cast::Cast;

/// Prelude module that re-exports commonly used items.
pub mod prelude {
    pub use super::cast::{ContractionCast, FetchCast, StreamCast};
    pub use super::{
        array_vec::*, context::*, memory_tensor::*, raw_tensor::*, runtime::*, scalar::*, stream_tensor::*, tensor::*,
    };
    // Stage types are accessed via stage::Init, stage::Branch, etc.
    pub use super::tensor_state::*;
    pub use super::vector_engine::{alu::*, branch::*, layer::*, op::*, operand::*, scalar::*, stage, tensor::*};
    pub use furiosa_mapping::*;
    pub use furiosa_opt_macro::{device, i, m};
}
