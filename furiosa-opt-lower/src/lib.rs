//! Furiosa-opt lowering engines.
//!
//! Thin safe wrappers over the prebuilt `furiosa-opt-lower-impl` static library (linked by this
//! crate's build script). The resolved results cross the `extern "C-unwind"` boundary as
//! [`furiosa_opt_lower_types`] StableAbi values; each wrapper converts back to a plain `Result`. The
//! frontend (`furiosa-opt-std`) and the IR backend (`tu-ops`) both call these.

use abi_stable::std_types::{RResult, RString};
use furiosa_mapping::Mapping;

use abi_stable::std_types::RVec;
pub use furiosa_opt_lower_types::{
    COMMIT_VALID_PACKET_SIZES, CommitError, DivideTerm, FETCH_ALIGN_BYTES, FactorLeaf, FetchError,
    MAX_SEQUENCER_ENTRIES, RelaxedDivision, StreamSequencerConfig, SwitchConfig, TransposeConfig,
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
        ) -> RResult<SwitchConfig, RString>;

        pub(super) fn config_tile(
            index: &Mapping,
            element: &Mapping,
            expected: &Mapping,
            len: usize,
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

/// Validate a switch topology against the slice/time shapes, or the rendered error.
pub fn config_switch(
    config: &SwitchConfig,
    in_slice: &Mapping,
    in_time: &Mapping,
    out_slice: &Mapping,
    out_time: &Mapping,
) -> Result<SwitchConfig, String> {
    unsafe { sys::config_switch(config, in_slice, in_time, out_slice, out_time) }
        .into_result()
        .map_err(String::from)
}

/// Validate a `tile` view (`index` divides `element` into `expected`), or the rendered error.
pub fn config_tile(index: &Mapping, element: &Mapping, expected: &Mapping, len: usize) -> Result<(), String> {
    unsafe { sys::config_tile(index, element, expected, len) }
        .into_result()
        .map_err(String::from)
}
