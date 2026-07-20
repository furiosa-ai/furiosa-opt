//! Type-level occupancy of the VE stash slot, and the single source of truth for its lifecycle.
//!
//! RNGD's stash (operand register) is write-once and read-once per VE pass. [`StashSlot`] encodes
//! that at compile time as a three-state machine, with exactly these transitions and no others:
//!
//! ```text
//! Fresh --vector_stash--> Occupied<D, Mapping> --Stash read--> Spent
//! ```
//!
//! - [`Fresh`]: never written. `vector_stash` is defined only here, so the write happens at most
//!   once.
//! - [`Occupied`]: holds a `Tensor<D, Mapping>`. The [`Stash`](crate::prelude::Stash) operand reads
//!   it, and that read is defined only on `Occupied`, so the read happens at most once.
//! - [`Spent`]: written and already read. Empty like `Fresh`, but a read lands here rather than
//!   back in `Fresh`, so neither a second write (no `vector_stash` on `Spent`) nor a second read
//!   (no read transition from `Spent`) has an impl to match.
//!
//! Docs elsewhere (`vector_stash`, `StashTransition`, `VeState::consume_stash`) reference this
//! module rather than restating the machine.

use std::fmt::Debug;

use crate::engine::vector::scalar::VeScalar;
use crate::tensor::Tensor;
use furiosa_mapping::M;

/// Occupancy of the stash slot at compile time. `D` ties an occupied slot's scalar to the
/// pipeline's, so a stash read is checked against the current `D`. See the [module docs] for the
/// full lifecycle.
///
/// [module docs]: crate::engine::vector::stash_slot
pub trait StashSlot<D: VeScalar>: Debug {
    /// Clones the slot's tensor (transposed to `TargetMapping`); `None` for the empty states
    /// ([`Fresh`] / [`Spent`]). Only [`Occupied`] overrides the default.
    fn clone_tensor_as<TargetMapping: M>(&self) -> Option<Tensor<D, TargetMapping>> {
        None
    }
}

/// The stash slot is empty and write-armed: never written this pass, so `vector_stash` is
/// available here (and only here). Named `Fresh` rather than a second "empty" word so a call site
/// reads which empty state still accepts a write ([`Fresh`]) versus which does not ([`Spent`]). See
/// the [module docs].
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct Fresh;
impl<D: VeScalar> StashSlot<D> for Fresh {}

/// The stash slot was written and already read this pass. See the [module docs] for why a read
/// lands here rather than in [`Fresh`].
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct Spent;
impl<D: VeScalar> StashSlot<D> for Spent {}

/// The stash slot holds a [`Tensor<D, Mapping>`]. See the [module docs].
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct Occupied<D: VeScalar, Mapping: M> {
    data: Tensor<D, Mapping>,
}

impl<D: VeScalar, Mapping: M> Occupied<D, Mapping> {
    /// Wraps a tensor into an occupied slot.
    pub(crate) fn new(tensor: Tensor<D, Mapping>) -> Self {
        Self { data: tensor }
    }
}

impl<D: VeScalar, Mapping: M> StashSlot<D> for Occupied<D, Mapping> {
    fn clone_tensor_as<TargetMapping: M>(&self) -> Option<Tensor<D, TargetMapping>> {
        // `transpose` already allocates a fresh tensor, so no separate clone is needed.
        Some(self.data.transpose::<TargetMapping>(true))
    }
}
