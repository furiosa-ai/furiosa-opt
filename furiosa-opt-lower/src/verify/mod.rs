//! Engine-constraint verification: the user-visible DSL contract for each Tensor-Unit engine.
//!
//! Unlike the lowering algorithms in the crate root (thin FFI wrappers over the hidden
//! `furiosa-opt-lower-impl`), these are plain, published functions. They state *what* mapping a stage
//! must produce, in pure `furiosa_mapping` terms. Both the frontend (`furiosa-opt-std`, whose
//! type-level `verify_*` delegate here) and the IR translator (`npu-visa-translate`) call them, so a
//! kernel is checked identically whether it runs under an interpreting backend or is compiled.
//!
//! Each check returns a typed, per-engine error enum carrying the offending mappings; both callers
//! render it (`panic!` in the frontend, a spanned diagnostic in the translator).
//!
//! When a check needs a hidden algorithm (e.g. exact division), it calls the FFI wrapper in the crate
//! root rather than reimplementing it, keeping the algorithm private while the constraint stays visible.
//!
//! The submodule layout follows `furiosa-opt-std/src/engine`: each verification lives under the matching
//! engine subtree. The vector engine's five checks are consolidated into one `vector/tensor.rs` rather
//! than split as they are in the frontend.

use std::collections::BTreeMap;

use furiosa_mapping::{Mapping, MappingExt};

use crate::DivideTerm;

pub mod engine;

pub use engine::{
    CastError, CollectError, CommitCastError, CommitTrimError, ContractLaneError, ContractPacketError,
    ContractTimeError, LaneMode, StreamAdapterError, ToTrfError, ToVrfError, VectorError, config_cast, config_collect,
    config_commit_cast, config_commit_trim, config_contract_lane, config_contract_packet, config_contract_time,
    config_reduce_label, config_stream_adapter, config_to_trf, config_to_vrf, config_vector_narrow_split,
    config_vector_narrow_trim, config_vector_widen_concat, config_vector_widen_pad,
};

/// Bits in a byte.
pub const BITS_PER_BYTE: usize = 8;
/// Size of a single flit in bytes; the switching network moves data in flit-sized units.
pub const FLIT_BYTES: usize = 32;
/// Vector register file capacity in bytes, per slice. One `to_vrf` operand must fit this.
pub const VRF_BYTES: usize = 8 * 1024;
/// Columns of the temporal accumulator (the packet reducer's output-width bound).
pub const TEMPORAL_ACCUMULATOR_COLS: usize = 32;
/// Elements of the lane-folder / packet-reducer output packet (one flit of `i32`/`f32`).
pub(crate) const CONTRACT_LANE_OUT_PACKET_ELEMENTS: usize = 8;

/// Byte size of `length` elements of an `element_bits`-wide type. Assumes the total bit count is
/// byte-aligned; sub-byte types must supply a length that fills whole bytes.
pub(crate) fn size_in_bytes(element_bits: usize, length: usize) -> usize {
    debug_assert!(
        (length * element_bits).is_multiple_of(BITS_PER_BYTE),
        "size_in_bytes: {length} x {element_bits} bits is not byte-aligned"
    );
    length * element_bits / BITS_PER_BYTE
}

/// Element count that fills `bytes` bytes of an `element_bits`-wide type.
pub(crate) fn length_from_bytes(element_bits: usize, bytes: usize) -> usize {
    bytes * BITS_PER_BYTE / element_bits
}

/// `a` rounded up to a multiple of `b`.
pub(crate) fn align_up(a: usize, b: usize) -> usize {
    a.div_ceil(b) * b
}

/// The hardware supports 1, 2, 4, or 8 lanes.
pub(crate) fn is_valid_lane_size(size: usize) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

/// Padded extent of each axis at its cumulative stride. On a normalized `Mapping` each axis carries
/// its padded extent as `size()`, and the running stride is the product of the inner axes' extents;
/// the contraction verifies key into this by a division term's `dividend_stride`.
pub(crate) fn padding_per_stride(m: &Mapping) -> BTreeMap<usize, usize> {
    let mut extents = Vec::new();
    collect_axis_extents(&m.normalize(), &mut extents);
    let mut map = BTreeMap::new();
    let mut stride = 1;
    for extent in extents {
        map.insert(stride, extent);
        stride *= extent;
    }
    map
}

/// Padded extent of a division term's axis, or `None` when the term is not an axis boundary.
///
/// Only an axis carries a padded extent. Dividing can split a padded axis into intra-axis sub-terms
/// when an axis between its digits is reduced, and such a sub-term owns no extent to report.
pub(crate) fn padded_extent_at(padding_per_stride: &BTreeMap<usize, usize>, term: &DivideTerm) -> Option<usize> {
    padding_per_stride.get(&term.dividend_stride).copied()
}

/// Pushes each axis's padded extent (`size()`), innermost (right) first, down the normalized `Pair`
/// spine. Relies on `normalize` fully decomposing composite leaves into the `Pair` spine so each
/// non-`Pair` node is a single axis whose stride key matches a division term's `dividend_stride`.
fn collect_axis_extents(m: &Mapping, out: &mut Vec<usize>) {
    match m {
        Mapping::Pair { left, right } => {
            collect_axis_extents(right, out);
            collect_axis_extents(left, out);
        }
        other => out.push(other.size()),
    }
}
