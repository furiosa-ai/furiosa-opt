//! Vector Engine stage markers and state.
//!
//! This module provides stage markers and state tracking for the Vector Engine pipeline.
//! The actual tensor types are in the `tensor` module.
//!
//! # Stage Order
//! ```text
//! Branch → Logic → Fxp → FxpToFp → Narrow → Fp → IntraSliceReduce → FpDiv → Widen → FpToFxp → Clip → Output
//! ```
//!
//! # Modules
//! - `markers`: Stage marker types and transition traits
//! - `state`: VE state tracking

pub mod markers;
pub(crate) mod state;

pub use markers::*;
