use std::collections::HashSet;

use furiosa_mapping::M;

use crate::{
    engine::vector::{
        alu::RngdAlu,
        scalar::VeScalar,
        stage::markers::Way,
        stash_slot::{Fresh, Occupied, Spent, StashState},
    },
    tensor::Tensor,
};

/// VE state that tracks stash and ALU usage. `Stash` is one of the stash-slot states ([`Fresh`] /
/// [`Occupied`] / [`Spent`]); the stashed scalar, mapping and way live on [`Occupied`], so a state
/// with nothing stashed carries none of them. See the stash-slot [module docs] for the state machine.
///
/// [module docs]: crate::engine::vector::stash_slot
#[derive(Debug)]
pub struct VeState<Stash: StashState> {
    /// Stash-slot state: [`Fresh`], [`Occupied`] (holding the tensor), or [`Spent`].
    pub(crate) stash: Stash,
    /// Set of ALUs that have been used.
    pub(crate) used_alus: HashSet<RngdAlu>,
}

impl<Stash: StashState> VeState<Stash> {
    /// Checks if ALU is available and marks it as used.
    /// Panics if ALU is already in use.
    pub(crate) fn use_alu(&mut self, alu: RngdAlu) {
        assert!(!self.used_alus.contains(&alu), "{alu:?} is already in use");
        self.used_alus.insert(alu);
    }
}

impl<D: VeScalar, StashMapping: M, const W: Way> VeState<Occupied<D, StashMapping, W>> {
    /// Empties the stash, preserving ALU usage. The read lands in [`Spent`] rather than [`Fresh`] so
    /// the write cannot be re-armed; the data itself was already read by the op that calls this.
    /// The stashed tensor in the reader's mapping; the slot's own scalar is what the reader gets.
    pub(crate) fn stash_tensor<TargetMapping: M>(&self) -> Tensor<D, TargetMapping> {
        self.stash.tensor_as()
    }

    pub(crate) fn consume_stash(self) -> VeState<Spent> {
        VeState {
            stash: Spent,
            used_alus: self.used_alus,
        }
    }
}

impl VeState<Fresh> {
    /// Creates a new empty VeState.
    pub(crate) fn new() -> Self {
        Self {
            stash: Fresh,
            used_alus: HashSet::new(),
        }
    }

    /// Stores the stream's tensor in the stash. The slot records that tensor's scalar and mapping and
    /// the pipeline's way, which is what a later read has to match. Not a conversion: a reinterpret
    /// belongs before the write, not at it.
    pub(crate) fn write_stash<StreamD: VeScalar, StreamMapping: M, const W: Way>(
        self,
        data: &Tensor<StreamD, StreamMapping>,
    ) -> VeState<Occupied<StreamD, StreamMapping, W>> {
        VeState {
            stash: Occupied::new(data.clone()),
            used_alus: self.used_alus,
        }
    }

    /// Consumes another VeState and merges its ALU usage into this one.
    /// Used when combining two groups in binary operations.
    /// Since both groups share the same ALU state (global resource),
    /// this simply performs a union of used ALUs.
    pub(crate) fn merge(&mut self, other: VeState<Fresh>) {
        self.used_alus.extend(other.used_alus);
    }
}
