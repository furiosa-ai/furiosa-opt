use std::collections::HashSet;
use std::marker::PhantomData;

use furiosa_mapping::M;

use crate::{
    engine::vector::{
        alu::RngdAlu,
        scalar::VeScalar,
        stash_slot::{Fresh, Occupied, Spent, StashSlot},
    },
    tensor::Tensor,
};

/// VE state that tracks stash and ALU usage.
///
/// The type parameter `D` ties the stash's scalar type to the pipeline's current scalar type,
/// ensuring at compile time that stash reads match the pipeline's `D`. The `Stash` parameter is one
/// of the stash-slot states ([`Fresh`] / [`Occupied`] / [`Spent`]); see the stash-slot
/// [module docs] for the state machine.
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct VeState<D: VeScalar, Stash: StashSlot<D>> {
    /// Stash-slot state: [`Fresh`], [`Occupied`] (holding the tensor), or [`Spent`].
    pub(crate) stash: Stash,
    /// Set of ALUs that have been used.
    pub(crate) used_alus: HashSet<RngdAlu>,
    /// Marker for the scalar type `D`.
    _marker: PhantomData<D>,
}

impl<D: VeScalar, Stash: StashSlot<D>> VeState<D, Stash> {
    /// Checks if ALU is available and marks it as used.
    /// Panics if ALU is already in use.
    pub(crate) fn use_alu(&mut self, alu: RngdAlu) {
        assert!(!self.used_alus.contains(&alu), "{alu:?} is already in use");
        self.used_alus.insert(alu);
    }

    /// Clones the stash data, transposing to target mapping and relabeling its scalar to the
    /// reading pipeline's `D2`. The stashed `D` and the requested `D2` must share a KIND (a
    /// cross-KIND stash read is rejected upstream by the typestate); both are i32 or both f32, so
    /// the relabel is a per-element identity via [`VeScalar::reinterpret`] (an exact-type `Any`
    /// downcast).
    ///
    /// The per-element downcast is `unwrap_unchecked` for perf (no panic landing-pad on the hot
    /// map); its soundness rests on the two-`VeScalar` invariant — see the SAFETY note below and the
    /// `# Panics` assert.
    ///
    /// TODO: fold the KIND equality into the typestate so this stays statically total; that also
    /// retires the `unsafe`.
    ///
    /// # Panics
    /// Panics if the requested `D2` KIND does not match the stashed `D` KIND.
    pub(crate) fn force_clone_stash_as<D2: VeScalar, TargetMapping: M>(&self) -> Option<Tensor<D2, TargetMapping>> {
        assert!(
            D::KIND == D2::KIND,
            "stash type mismatch: stashed as {:?}, requested as {:?}",
            D::KIND,
            D2::KIND
        );

        self.stash
            .clone_tensor_as::<TargetMapping>()
            // SAFETY: the `assert!(D::KIND == D2::KIND)` above holds, and `VeScalar` is implemented only for
            // `i32` and `f32`, whose KINDs are distinct — so equal KIND means the SAME concrete type, and
            // `reinterpret::<D2>()` (an exact-type `Any` downcast) is always `Some` here. If a third
            // `VeScalar` sharing a KIND is ever added, this precondition breaks (the assert would pass but the
            // downcast return `None`); fold the KIND equality into the typestate (see the TODO) before then.
            .map(|t| t.map(|x| unsafe { x.reinterpret::<D2>().unwrap_unchecked() }))
    }
}

impl<D: VeScalar, StashMapping: M> VeState<D, Occupied<D, StashMapping>> {
    /// Empties the stash, moving the state to [`Spent`] and preserving ALU usage. `D` here is the
    /// stashed scalar type, which may differ from the reading pipeline's scalar (cross-type stash).
    /// The read lands in `Spent` rather than `Fresh` so the write cannot be re-armed; see the
    /// stash-slot [module docs] for the state machine. The stashed tensor's data was already read by
    /// the op that calls this.
    ///
    /// [module docs]: crate::engine::vector::stash_slot
    pub(crate) fn consume_stash(self) -> VeState<D, Spent> {
        VeState {
            stash: Spent,
            used_alus: self.used_alus,
            _marker: PhantomData,
        }
    }
}

impl<D: VeScalar> VeState<D, Fresh> {
    /// Creates a new empty VeState.
    pub(crate) fn new() -> Self {
        Self {
            stash: Fresh,
            used_alus: HashSet::new(),
            _marker: PhantomData,
        }
    }

    /// Stores the tensor data in the stash, returning a new VeState with the stash's mapping.
    /// Consumes self to change the `StashSlot` parameter.
    pub(crate) fn stash<NewMapping: M>(self, data: &Tensor<D, NewMapping>) -> VeState<D, Occupied<D, NewMapping>> {
        VeState {
            stash: Occupied::new(data.clone()),
            used_alus: self.used_alus,
            _marker: PhantomData,
        }
    }

    /// Consumes another VeState and merges its ALU usage into this one.
    /// Used when combining two groups in binary operations.
    /// Since both groups share the same ALU state (global resource),
    /// this simply performs a union of used ALUs.
    pub(crate) fn merge(&mut self, other: VeState<D, Fresh>) {
        self.used_alus.extend(other.used_alus);
    }

    /// Converts VeState to different scalar type parameters, preserving ALU tracking.
    pub(crate) fn retype<NewD: VeScalar>(self) -> VeState<NewD, Fresh> {
        VeState {
            stash: Fresh,
            used_alus: self.used_alus,
            _marker: PhantomData,
        }
    }
}
