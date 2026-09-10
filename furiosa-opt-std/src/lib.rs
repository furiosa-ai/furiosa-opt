//! Virtual ISA standard library.

#![expect(incomplete_features)]
#![feature(adt_const_params)]
#![feature(float_erf)]
#![feature(inherent_associated_types)]
#![feature(impl_trait_in_assoc_type)]
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
mod control;
mod engine;
mod float;
mod scalar;
mod storage;
mod tensor;

pub mod backend;
pub mod runtime;

/// Internal APIs used by generated code.
#[doc(hidden)]
pub mod __private {
    pub use crate::backend::npu::__private::DeviceOutput;
    pub use crate::control::__loop_hint_unroll;
}

pub use runtime::{Device, Topology};

/// Why a device, a transfer or a launch failed: the chips could not be opened, the
/// runtime refused or failed the work, or a tensor was not where the operation needs it.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The chips could not be opened: the process's chip policy or the chips refused.
    #[error("opening the chips: {0}")]
    Chips(String),
    /// The runtime refused or failed a load, a launch or a transfer.
    #[error(transparent)]
    Device(#[from] furiosa_opt_rt::Error),
    /// An HBM tensor that never went through `to_hbm` was transferred or launched.
    #[error("an HBM tensor the runtime never placed cannot be transferred or launched")]
    Unplaced,
    /// The binary carries no compiled image for this device function and generic arguments.
    #[error("no compiled device function for `{path}` with generic arguments {key:?}")]
    Uncompiled {
        /// The function's registry path.
        path: &'static str,
        /// Its generic arguments' values.
        key: Vec<usize>,
    },
    /// An earlier panic left the runtime's state unusable.
    #[error("the runtime is poisoned by an earlier panic")]
    Poisoned,
}

impl From<furiosa_opt_rt::FunctionError> for Error {
    fn from(error: furiosa_opt_rt::FunctionError) -> Self {
        Self::Device(error.into())
    }
}

pub use tensor::pseudo;

/// Prelude module that re-exports commonly used items.
pub mod prelude {
    pub use super::cast::{
        Cast, CastEngineCast, CommitCast, ContractionAccumulator, ContractionCast, ContractionWeight, FetchCast,
        FetchZeroPointSub, TableLookupCast,
    };
    pub use super::context::{Dma, DmaContext, Tu, TuContext};
    pub use super::engine::vector::stash_slot::*;
    pub use super::engine::vector::{alu::*, branch::*, layer::*, op::*, operand::*, scalar::*, stage, tensor::*};
    pub use super::engine::*;
    pub use super::runtime::{Device, Topology};
    pub use super::tensor::memory::*;
    pub use super::tensor::tu::*;
    pub use super::{Error, array_vec::*, backend::*, runtime::*, scalar::*, storage::*, tensor::*};
    pub use furiosa_mapping::*;
    pub use furiosa_opt_macro::{DeviceSend, device, unroll};
}
