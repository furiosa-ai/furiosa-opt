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
mod constraints;
mod context;
mod engine;
mod float;
mod scalar;
mod storage;
mod tensor;

pub mod backend;
pub mod runtime;

pub use tokio::sync::OnceCell;

pub use context::Device;

pub use tensor::pseudo;

/// Prelude module that re-exports commonly used items.
pub mod prelude {
    pub use super::cast::{
        Cast, CastEngineCast, CommitCast, ContractionAccumulator, ContractionCast, ContractionWeight, FetchCast,
        FetchZeroPointSub, TableLookup,
    };
    pub use super::engine::vector::stash_slot::*;
    pub use super::engine::vector::{alu::*, branch::*, layer::*, op::*, operand::*, scalar::*, stage, tensor::*};
    pub use super::engine::*;
    pub use super::tensor::memory::*;
    pub use super::tensor::tu::*;
    pub use super::{array_vec::*, backend::*, context::*, runtime::*, scalar::*, storage::*, tensor::*};
    pub use furiosa_mapping::*;
    pub use furiosa_opt_macro::{DeviceSend, device};
}
