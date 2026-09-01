//! Raw declarations for the prebuilt lowering implementation.

use abi_stable::std_types::{ROption, RResult, RVec};
use furiosa_mapping::Mapping;

use crate::{
    CommitError, DivideError, DivideTerm, FactorLeaf, FetchBaseError, FetchError, PadError, RelaxedDivision,
    StreamSequencerConfig, SwitchConfig, SwitchError, TransposeConfig, TransposeError,
};

#[expect(improper_ctypes, reason = "all crossing types are #[repr(C)] + StableAbi")]
unsafe extern "C-unwind" {
    pub(super) fn config_transpose(
        time: &Mapping,
        packet: &Mapping,
        out_time: &Mapping,
        out_packet: &Mapping,
        element_bits: usize,
    ) -> RResult<TransposeConfig, TransposeError>;

    pub(super) fn config_fetch(
        in_time: &Mapping,
        in_packet: &Mapping,
        out_time: &Mapping,
        out_packet: &Mapping,
        lifted: ROption<&Mapping>,
    ) -> RResult<StreamSequencerConfig, FetchError>;

    pub(super) fn config_fetch_lift_bases(grid: &Mapping, dm: &Mapping) -> RResult<RVec<usize>, FetchBaseError>;

    pub(super) fn config_fetch_base_aligned(
        bases: abi_stable::std_types::RSlice<'_, usize>,
        element_bits: usize,
    ) -> RResult<(), FetchBaseError>;

    pub(super) fn config_commit(
        in_time: &Mapping,
        in_packet: &Mapping,
        element: &Mapping,
        element_bits: usize,
    ) -> RResult<StreamSequencerConfig, CommitError>;

    pub(super) fn config_divide_exact(dividend: &Mapping, divisor: &Mapping) -> RResult<RVec<DivideTerm>, DivideError>;

    pub(super) fn config_divide_relaxed(dividend: &Mapping, divisor: &Mapping) -> RelaxedDivision;

    pub(super) fn mapping_factor_leaves(mapping: &Mapping) -> RVec<FactorLeaf>;

    pub(super) fn config_switch(
        config: &SwitchConfig,
        in_slice: &Mapping,
        in_time: &Mapping,
        out_slice: &Mapping,
        out_time: &Mapping,
    ) -> RResult<SwitchConfig, SwitchError>;

    pub(super) fn switch_custom_snoop_bitmap(
        slice: &Mapping,
        time: &Mapping,
        out_slice: &Mapping,
        out_time: &Mapping,
        ring_size: usize,
    ) -> RResult<RVec<RVec<usize>>, SwitchError>;

    pub(super) fn config_pad(element: &Mapping, expected: &Mapping) -> RResult<(), PadError>;
}
