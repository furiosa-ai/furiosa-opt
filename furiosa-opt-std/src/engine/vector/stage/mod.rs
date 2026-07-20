//! Vector Engine stage markers and state. The tensor types themselves live in the `tensor`
//! module; see [`markers`] for the stage order and transition traits.

pub mod markers;
pub(crate) mod state;

pub use markers::*;
