//! Type-level occupancy of the VE stash slot, and the single source of truth for its lifecycle.
//!
//! RNGD's stash (operand register) is write-once and read-once per VE pass. [`StashState`] encodes
//! that at compile time as a three-state machine, with exactly these transitions and no others:
//!
//! ```text
//! Fresh --vector_stash--> Occupied<D, Mapping, W> --Stash read--> Spent
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
use crate::engine::vector::stage::markers::Way;
use crate::tensor::Tensor;
use furiosa_mapping::M;

/// Occupancy of the stash slot at compile time: implemented by the three states below and by nothing
/// else. This is the bound the pipeline carries, and it names no scalar: the scalar, the mapping and
/// the way live on [`Occupied`] alone, so a pipeline with nothing stashed carries none of them. See
/// the [module docs] for the full lifecycle.
///
/// [module docs]: crate::engine::vector::stash_slot
pub trait StashState: Debug + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
}

/// The stash slot is empty and write-armed: never written this pass, so `vector_stash` is
/// available here (and only here). Named `Fresh` rather than a second "empty" word so a call site
/// reads which empty state still accepts a write ([`Fresh`]) versus which does not ([`Spent`]). See
/// the [module docs].
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct Fresh;
impl StashState for Fresh {}
impl sealed::Sealed for Fresh {}

/// The stash slot was written and already read this pass. See the [module docs] for why a read
/// lands here rather than in [`Fresh`].
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct Spent;
impl StashState for Spent {}
impl sealed::Sealed for Spent {}

/// The stash slot holds a [`Tensor<D, Mapping>`] written at way `W`. See the [module docs].
///
/// `W` is here because the operand register is addressed per way: a stash written 4-way and read
/// 8-way (or the other way round) hands the reader the wrong elements, which the LIR conversion
/// admits it "cannot sync". Carrying the way makes such a read a missing impl on
/// [`StashTransition`](crate::engine::vector::operand::StashTransition) rather than a wrong number.
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct Occupied<D: VeScalar, Mapping: M, const W: Way> {
    data: Tensor<D, Mapping>,
}

impl<D: VeScalar, Mapping: M, const W: Way> Occupied<D, Mapping, W> {
    /// Wraps a tensor into an occupied slot.
    pub(crate) fn new(tensor: Tensor<D, Mapping>) -> Self {
        Self { data: tensor }
    }
}

impl<D: VeScalar, Mapping: M, const W: Way> StashState for Occupied<D, Mapping, W> {}
impl<D: VeScalar, Mapping: M, const W: Way> sealed::Sealed for Occupied<D, Mapping, W> {}

impl<D: VeScalar, Mapping: M, const W: Way> Occupied<D, Mapping, W> {
    /// The stashed tensor in the reader's mapping. Only this state has one, so the read needs no
    /// `Option` and no scalar of its own: `D` here is the scalar the stash was written with.
    pub(crate) fn tensor_as<TargetMapping: M>(&self) -> Tensor<D, TargetMapping> {
        // `transpose` already allocates a fresh tensor, so no separate clone is needed.
        self.data.transpose::<TargetMapping>(true)
    }
}
