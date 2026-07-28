//! Furiosa-opt lowering engines.
//!
//! Two kinds of function live here. The lowering *algorithms* (`config_transpose`, `config_fetch`, …)
//! are thin safe wrappers over the prebuilt, hidden `furiosa-opt-lower-impl` static library: the
//! resolved results cross the `extern "C-unwind"` boundary as [`furiosa_opt_lower_types`] StableAbi
//! values, and each wrapper converts back to a plain `Result`. The engine *verifications* (the
//! `config_*` in [`verify`]) are published in full: they state the DSL contract in pure
//! `furiosa_mapping` terms, with no hidden algorithm. Both the frontend (`furiosa-opt-std`) and the IR
//! backend (`tu-ops` / `npu-visa-translate`) call all of these.

use abi_stable::std_types::{RResult, RString};
use furiosa_mapping::{Mapping, PaddingKind};

use abi_stable::std_types::RVec;
pub use furiosa_opt_lower_types::{
    COMMIT_VALID_PACKET_SIZES, CommitError, DivideTerm, FactorLeaf, FetchError, MAX_SEQUENCER_ENTRIES, RelaxedDivision,
    StreamSequencerConfig, SwitchAxis, SwitchConfig, SwitchError, SwitchFrame, TransposeConfig,
};

mod verify;
pub use verify::{
    BITS_PER_BYTE, CastError, CollectError, CommitTrimError, ContractLaneError, ContractPacketError, ContractTimeError,
    FLIT_BYTES, LaneMode, StreamAdapterError, TEMPORAL_ACCUMULATOR_COLS, ToTrfError, VectorError, config_cast,
    config_collect, config_commit_trim, config_contract_lane, config_contract_packet, config_contract_time,
    config_reduce_label, config_stream_adapter, config_to_trf, config_vector_narrow_split, config_vector_narrow_trim,
    config_vector_widen_concat, config_vector_widen_pad,
};

/// Raw `extern "C-unwind"` decls for the prebuilt impl's exports.
mod sys {
    use super::*;

    #[expect(improper_ctypes, reason = "all crossing types are #[repr(C)] + StableAbi")]
    unsafe extern "C-unwind" {
        pub(super) fn config_transpose(
            time: &Mapping,
            packet: &Mapping,
            out_time: &Mapping,
            out_packet: &Mapping,
            element_bits: usize,
        ) -> RResult<TransposeConfig, RString>;

        pub(super) fn config_fetch(
            in_time: &Mapping,
            in_packet: &Mapping,
            out_time: &Mapping,
            out_packet: &Mapping,
        ) -> RResult<StreamSequencerConfig, FetchError>;

        pub(super) fn config_commit(
            in_time: &Mapping,
            in_packet: &Mapping,
            element: &Mapping,
            element_bits: usize,
        ) -> RResult<StreamSequencerConfig, CommitError>;

        pub(super) fn config_divide_exact(dividend: &Mapping, divisor: &Mapping) -> RResult<RVec<DivideTerm>, RString>;

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

        pub(super) fn config_tile(
            index: &Mapping,
            element: &Mapping,
            expected: &Mapping,
            len: usize,
            hole_fill: PaddingKind,
        ) -> RResult<(), RString>;
    }
}

/// Resolve transpose-engine hardware parameters, or the rendered error.
pub fn config_transpose(
    time: &Mapping,
    packet: &Mapping,
    out_time: &Mapping,
    out_packet: &Mapping,
    element_bits: usize,
) -> Result<TransposeConfig, String> {
    unsafe { sys::config_transpose(time, packet, out_time, out_packet, element_bits) }
        .into_result()
        .map_err(String::from)
}

/// Synthesize the fetch read descriptors, or the [`FetchError`].
pub fn config_fetch(
    in_time: &Mapping,
    in_packet: &Mapping,
    out_time: &Mapping,
    out_packet: &Mapping,
) -> Result<StreamSequencerConfig, FetchError> {
    unsafe { sys::config_fetch(in_time, in_packet, out_time, out_packet) }.into_result()
}

/// Synthesize the commit write descriptors, or the [`CommitError`].
pub fn config_commit(
    in_time: &Mapping,
    in_packet: &Mapping,
    element: &Mapping,
    element_bits: usize,
) -> Result<StreamSequencerConfig, CommitError> {
    unsafe { sys::config_commit(in_time, in_packet, element, element_bits) }.into_result()
}

/// Carve `divisor` out of `dividend` exactly; the matched axes, or a rendered error.
pub fn config_divide_exact(dividend: &Mapping, divisor: &Mapping) -> Result<Vec<DivideTerm>, String> {
    unsafe { sys::config_divide_exact(dividend, divisor) }
        .into_result()
        .map(Vec::from)
        .map_err(String::from)
}

/// Relaxed carve: matched axes, both residues (`Mapping`), and the contiguous tail.
pub fn config_divide_relaxed(dividend: &Mapping, divisor: &Mapping) -> RelaxedDivision {
    unsafe { sys::config_divide_relaxed(dividend, divisor) }
}

/// `Mapping::factor_leaves()`: the factor decomposition, innermost-first — each leaf a named axis or an
/// untagged composite/padding run.
pub trait FactorLeavesExt {
    /// The factor leaves of this mapping, innermost-first.
    fn factor_leaves(&self) -> Vec<FactorLeaf>;
}

impl FactorLeavesExt for Mapping {
    fn factor_leaves(&self) -> Vec<FactorLeaf> {
        unsafe { sys::mapping_factor_leaves(self) }.into()
    }
}

/// Validate a switch topology against the slice/time shapes, or the [`SwitchError`].
pub fn config_switch(
    config: &SwitchConfig,
    in_slice: &Mapping,
    in_time: &Mapping,
    out_slice: &Mapping,
    out_time: &Mapping,
) -> Result<SwitchConfig, SwitchError> {
    unsafe { sys::config_switch(config, in_slice, in_time, out_slice, out_time) }.into_result()
}

/// Verify a `CustomBroadcast` and return its snoop bitmap (`bitmap[out_slice]` = the input
/// slice lanes that output slice reads), or the [`SwitchError`]. The verify and the bitmap
/// share one Read sequencer, so the bitmap is only ever produced for a sound config.
pub fn switch_custom_snoop_bitmap(
    slice: &Mapping,
    time: &Mapping,
    out_slice: &Mapping,
    out_time: &Mapping,
    ring_size: usize,
) -> Result<Vec<Vec<usize>>, SwitchError> {
    unsafe { sys::switch_custom_snoop_bitmap(slice, time, out_slice, out_time, ring_size) }
        .into_result()
        .map(|bitmap| bitmap.into_iter().map(Vec::from).collect())
}

/// Validate a `tile` view (`index` divides `element` into `expected`), or the rendered error.
///
/// `hole_fill` is the padding kind the out-of-tile cells must carry — Top for a
/// read `view().tile()`, Bottom (down padding) for a `view_mut().tile()` write
/// destination.
pub fn config_tile(
    index: &Mapping,
    element: &Mapping,
    expected: &Mapping,
    len: usize,
    hole_fill: PaddingKind,
) -> Result<(), String> {
    unsafe { sys::config_tile(index, element, expected, len, hole_fill) }
        .into_result()
        .map_err(String::from)
}
