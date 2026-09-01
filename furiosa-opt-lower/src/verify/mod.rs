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
use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Mapping, MappingExt};

use crate::{DivideError, DivideInput, DivideTerm};

pub mod engine;

pub use engine::{
    CastError, CastInput, CastKind, CollectError, CollectInput, CommitCastError, CommitCastInput, CommitCastKind,
    CommitTrimError, CommitTrimInput, ContractLaneError, ContractLaneInput, ContractPacketError, ContractPacketInput,
    ContractTimeError, ContractTimeInput, FetchDimensionsInput, FetchLiftDimension, FetchLiftError, FetchLiftInput,
    LaneMode, ReduceLabelInput, StreamAdapterError, StreamAdapterInput, ToTrfError, ToTrfInput, ToVrfError, ToVrfInput,
    UnreadableCause, VectorError, VectorIntraSliceUnzipInput, VectorNarrowSplitInput, VectorNarrowTrimInput,
    VectorWidenConcatInput, VectorWidenPadInput, VrfOperandError, VrfOperandInput, config_cast, config_collect,
    config_commit_cast, config_commit_trim, config_contract_lane, config_contract_packet, config_contract_time,
    config_fetch_dimensions, config_fetch_lift, config_reduce_label, config_stream_adapter, config_to_trf,
    config_to_vrf, config_vector_intra_slice_unzip, config_vector_narrow_split, config_vector_narrow_trim,
    config_vector_widen_concat, config_vector_widen_pad, config_vrf_operand,
};

/// Why an element count cannot be represented as an exact byte count.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ElementSizeError {
    /// Element width is zero.
    #[error("element width must be nonzero")]
    ZeroWidth,
    /// Multiplying the element count by its width overflowed.
    #[error("{elements} elements x {element_bits} bits overflows usize")]
    BitCountOverflow { elements: usize, element_bits: usize },
    /// Multiplying element extents overflowed.
    #[error("{left} x {right} elements overflows usize")]
    ElementCountOverflow { left: usize, right: usize },
    /// The element sequence ends inside a byte.
    #[error("{elements} elements x {element_bits} bits is not byte-aligned")]
    NotByteAligned { elements: usize, element_bits: usize },
    /// Multiplying a byte count by eight overflowed.
    #[error("a byte-count calculation starting from {bytes} bytes overflows usize")]
    ByteCountOverflow { bytes: usize },
    /// The byte sequence ends inside an element.
    #[error("{bytes} bytes does not contain a whole number of {element_bits}-bit elements")]
    NotElementAligned { bytes: usize, element_bits: usize },
}

/// Packet endpoint checked against a one-flit size constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketSide {
    Input,
    Output,
}

impl Display for PacketSide {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "input",
            Self::Output => "output",
        })
    }
}

/// A packet whose byte size is not exactly one flit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{side} packet must be exactly {FLIT_BYTES} bytes (one flit): {elements} elements = {bytes} bytes")]
pub struct OneFlitPacketError {
    pub side: PacketSide,
    pub elements: usize,
    pub bytes: usize,
}

/// Bits in a byte.
pub const BITS_PER_BYTE: usize = 8;
/// Size of a single flit in bytes; the switching network moves data in flit-sized units.
pub const FLIT_BYTES: usize = 32;
/// Vector register file capacity in bytes, per slice. One `to_vrf` operand must fit this.
pub const VRF_BYTES: usize = 8 * 1024;
/// Vector register-file cache capacity in bytes. One interleaved unzip group must fit this.
pub const VRF_CACHE_BYTES: usize = 1024;
/// Vector-engine element width in bits: the engine computes on 32-bit lanes only.
pub(crate) const VE_ELEMENT_BITS: usize = 32;
/// Columns of the temporal accumulator (the packet reducer's output-width bound).
pub const TEMPORAL_ACCUMULATOR_COLS: usize = 32;
/// Elements of the lane-folder / packet-reducer output packet (one flit of `i32`/`f32`).
pub(crate) const CONTRACT_LANE_OUT_PACKET_ELEMENTS: usize = 8;
/// Elements in a one-flit vector packet (8 x 32-bit lanes), which is `Way8`'s ALU access width.
pub(crate) const ONE_FLIT_ELEMENTS: usize = 8;
/// Elements in a half-flit vector packet (the front-4 lanes, the partial-reduction width, and
/// `Way4`'s ALU access width).
pub(crate) const HALF_FLIT_ELEMENTS: usize = 4;

/// Returns the exact byte size of an element sequence.
pub(crate) fn size_in_bytes(element_bits: usize, elements: usize) -> Result<usize, ElementSizeError> {
    if element_bits == 0 {
        return Err(ElementSizeError::ZeroWidth);
    }
    let bits = elements
        .checked_mul(element_bits)
        .ok_or(ElementSizeError::BitCountOverflow { elements, element_bits })?;
    if !bits.is_multiple_of(BITS_PER_BYTE) {
        return Err(ElementSizeError::NotByteAligned { elements, element_bits });
    }
    Ok(bits / BITS_PER_BYTE)
}

pub(crate) fn require_one_flit(side: PacketSide, elements: usize, bytes: usize) -> Result<(), OneFlitPacketError> {
    if bytes != FLIT_BYTES {
        return Err(OneFlitPacketError { side, elements, bytes });
    }
    Ok(())
}

/// Returns the exact element count contained in a byte sequence.
pub(crate) fn length_from_bytes(element_bits: usize, bytes: usize) -> Result<usize, ElementSizeError> {
    if element_bits == 0 {
        return Err(ElementSizeError::ZeroWidth);
    }
    let bits = bytes
        .checked_mul(BITS_PER_BYTE)
        .ok_or(ElementSizeError::ByteCountOverflow { bytes })?;
    if !bits.is_multiple_of(element_bits) {
        return Err(ElementSizeError::NotElementAligned { bytes, element_bits });
    }
    Ok(bits / element_bits)
}

/// `a` rounded up to a multiple of `b`.
pub(crate) fn align_up(a: usize, b: usize) -> usize {
    a.div_ceil(b) * b
}

/// The hardware supports 1, 2, 4, or 8 lanes.
pub(crate) fn is_valid_lane_size(size: usize) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

/// Padded extent of each axis at its cumulative stride; the contraction verifies key into this by a
/// division term's `dividend_stride`.
pub(crate) fn padding_per_stride(m: &Mapping) -> BTreeMap<usize, usize> {
    let mut map = BTreeMap::new();
    let mut stride = 1;
    for axis in axis_leaves(m) {
        map.insert(stride, axis.size());
        stride *= axis.size();
    }
    map
}

/// The axes of a mapping, innermost (right) first, each carrying its padded extent as `size()`.
pub(crate) fn axis_leaves(m: &Mapping) -> Vec<Mapping> {
    let mut axes = Vec::new();
    collect_axis_leaves(&m.normalize(), &mut axes);
    axes
}

/// Padded extent of a division term's axis, or `None` when the term is not an axis boundary.
///
/// Only an axis carries a padded extent. Dividing can split a padded axis into intra-axis sub-terms
/// when an axis between its digits is reduced, and such a sub-term owns no extent to report.
pub(crate) fn padded_extent_at(padding_per_stride: &BTreeMap<usize, usize>, term: &DivideTerm) -> Option<usize> {
    padding_per_stride.get(&term.dividend_stride).copied()
}

/// Pushes each axis, innermost (right) first, down the normalized `Pair` spine. Relies on `normalize`
/// fully decomposing composite leaves, so each non-`Pair` node is a single axis.
fn collect_axis_leaves(m: &Mapping, out: &mut Vec<Mapping>) {
    match m {
        Mapping::Pair { left, right } => {
            collect_axis_leaves(right, out);
            collect_axis_leaves(left, out);
        }
        other => out.push(other.clone()),
    }
}

/// Live positions inner to the outermost reduced axis (`InnerTime::SIZE`), the accumulator slots a
/// reducer holds at once, or an error when `post_reduce` does not divide `pre_reduce`.
///
/// The division matches the retained axes; the reduced ones are the gaps between them. Padded
/// extents count, since a padded position still takes its slot.
pub(crate) fn inner_reduce_extent(pre_reduce: &Mapping, post_reduce: &Mapping) -> Result<usize, DivideError> {
    let division_terms = crate::config_divide_exact(DivideInput {
        dividend: pre_reduce.clone(),
        divisor: post_reduce.clone(),
    })?;
    let padded_extents = padding_per_stride(pre_reduce);
    // A sub-term is dropped before the walk, not skipped inside it: the walk compares adjacent
    // boundaries, and skipping in place would still compare across the sub-term.
    let boundaries: Vec<(&DivideTerm, usize)> = division_terms
        .iter()
        .filter_map(|term| padded_extent_at(&padded_extents, term).map(|extent| (term, extent)))
        .collect();
    let dividend_end = |&(term, extent): &(&DivideTerm, usize)| term.dividend_stride * extent;
    Ok(if boundaries.is_empty() {
        // All axes reduced.
        1
    } else if dividend_end(&boundaries[0]) < pre_reduce.size() {
        // The outermost axis was reduced, so everything below the top is inner to the reduce.
        post_reduce.size()
    } else {
        // The outermost retained factor reaches the top; walk outer-to-inner to the first gap between
        // adjacent retained terms (the reduce boundary), else nothing is inner to the reduce.
        boundaries
            .windows(2)
            .find(|w| dividend_end(&w[1]) != w[0].0.dividend_stride)
            .map_or(1, |w| w[0].0.divisor_stride)
    })
}
