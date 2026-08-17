//! Lane folder (`contract_lane`): fold `Lane` into the output stream.
//!
//! [`LaneMode`] selects the destination: `Interleaved` relocates `Lane` into `OutPacket`, `Sequential`
//! relocates it into `OutTime`. This mirrors the frontend-only `furiosa_opt_std` `LaneMode`; callers map
//! their mode onto this one so the check stays free of the frontend type.

use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};

use crate::DivideTerm;
use crate::verify::{
    CONTRACT_LANE_OUT_PACKET_ELEMENTS, TEMPORAL_ACCUMULATOR_COLS, align_up, padded_extent_at, padding_per_stride,
};

/// MAC accumulator element capacity; the reduce buffer (axes inner to the reduce) must fit within it.
const ACCUMULATOR_CAPACITY_ELEMENTS: usize = 1024;

/// Where the Lane Folder relocates the `Lane` axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneMode {
    /// `Lane` moves into `OutPacket`.
    Interleaved,
    /// `Lane` moves into `OutTime`.
    Sequential,
}

impl Display for LaneMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Interleaved => "Interleaved",
            Self::Sequential => "Sequential",
        })
    }
}

/// Why a lane fold is not realizable on the Lane Folder.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractLaneError {
    /// The input packet exceeds the accumulator width.
    #[error("contract_lane: Packet::SIZE must be at most {TEMPORAL_ACCUMULATOR_COLS}, got {0}")]
    PacketTooWide(usize),
    /// The output packet is not one flit.
    #[error("contract_lane: OutPacket::SIZE must be {CONTRACT_LANE_OUT_PACKET_ELEMENTS}, got {0}")]
    OutPacketSize(usize),
    /// The output packet does not match the expected fold.
    #[error("contract_lane ({mode}): OutPacket mismatch. Expected: {expected}, got: {got}")]
    OutPacketMismatch {
        /// The fold mode.
        mode: LaneMode,
        /// The expected output packet.
        expected: Mapping,
        /// The declared output packet.
        got: Mapping,
    },
    /// The inner portion of `OutTime` does not equal the folded axes (`[Packet]` Interleaved,
    /// `[Lane, packet_outer]` Sequential).
    #[error("contract_lane ({mode}): OutTime mismatch. Expected {expected}, got {got}")]
    OutTimeMismatch {
        /// The fold mode.
        mode: LaneMode,
        /// The expected inner portion.
        expected: Mapping,
        /// The actual inner portion of `OutTime`.
        got: Mapping,
    },
    /// The outer portion of `OutTime` does not equal `Time`.
    #[error(
        "contract_lane ({mode}): OutTime mismatch. Outer portion of OutTime must equal Time: expected {expected}, got {got}"
    )]
    OuterTimeMismatch {
        /// The fold mode.
        mode: LaneMode,
        /// The (post-reduce) time.
        expected: Mapping,
        /// The outer portion of `OutTime`.
        got: Mapping,
    },
    /// The pre- and post-reduce times are inconsistent (the pre-reduce does not divide by the post).
    #[error("contract_lane ({mode}): inconsistent pre/post reduce Time: {pre_reduce_time}, {time}")]
    InconsistentReduceTime {
        /// The fold mode.
        mode: LaneMode,
        /// The pre-reduce time.
        pre_reduce_time: Mapping,
        /// The post-reduce time.
        time: Mapping,
    },
    /// The `[Lane, Packet]` chunks for the inner-reduce positions overflow the accumulator.
    #[error(
        "contract_lane ({mode}): the [Lane, Packet] accumulator buffer overflows: \
         padded Lane {padded_lane} * InnerTime {inner_time} * padded Packet {padded_packet} = {} \
         exceeds the {limit}-cell accumulator",
        padded_lane * inner_time * padded_packet
    )]
    BufferExceeded {
        /// The fold mode.
        mode: LaneMode,
        /// `Lane` cells per chunk: padded to the 8-wide output bus (Interleaved) or `Lane::SIZE` as-is
        /// (Sequential).
        padded_lane: usize,
        /// Number of buffer slots: the axes inner to the outermost reduce (`InnerTime::SIZE`).
        inner_time: usize,
        /// `Packet` cells per chunk: `Packet::SIZE` as-is (Interleaved) or padded to the 32-column
        /// accumulator (Sequential).
        padded_packet: usize,
        /// The accumulator cell capacity.
        limit: usize,
    },
}

/// `contract_lane` performs no reduction, so the outer portion of `OutTime` must equal `Time`, the
/// output packet must be one flit, and the axes inner to the outermost reduce must fit the accumulator.
pub fn config_contract_lane(
    lane: &Mapping,
    time: &Mapping,
    packet: &Mapping,
    out_time: &Mapping,
    out_packet: &Mapping,
    pre_reduce_time: &Mapping,
    mode: LaneMode,
) -> Result<(), ContractLaneError> {
    let interleaved = matches!(mode, LaneMode::Interleaved);

    if packet.size() > TEMPORAL_ACCUMULATOR_COLS {
        return Err(ContractLaneError::PacketTooWide(packet.size()));
    }
    if out_packet.size() != CONTRACT_LANE_OUT_PACKET_ELEMENTS {
        return Err(ContractLaneError::OutPacketSize(out_packet.size()));
    }

    let lane_size = lane.size();

    let outer_time = if interleaved {
        // `OutTime = [Time, Packet]`, `OutPacket = [Lane # 8]`.
        let expected_out_packet = lane
            .clone()
            .padding(CONTRACT_LANE_OUT_PACKET_ELEMENTS, PaddingKind::Top)
            .normalize();
        if out_packet.normalize() != expected_out_packet {
            return Err(ContractLaneError::OutPacketMismatch {
                mode,
                expected: expected_out_packet,
                got: out_packet.normalize(),
            });
        }

        split_inner_time(out_time, packet, mode)?
    } else {
        // `OutTime = [Time, Lane, packet_outer]`, `OutPacket = [packet_inner # 8]`.
        let padded = packet.clone().padding(
            align_up(packet.size(), CONTRACT_LANE_OUT_PACKET_ELEMENTS),
            PaddingKind::Top,
        );
        let (packet_outer, packet_inner) = padded.split_at(CONTRACT_LANE_OUT_PACKET_ELEMENTS);

        if packet_inner.normalize() != out_packet.normalize() {
            return Err(ContractLaneError::OutPacketMismatch {
                mode,
                expected: packet_inner,
                got: out_packet.normalize(),
            });
        }

        split_inner_time(out_time, &lane.clone().pair(packet_outer), mode)?
    };

    // The post-split outer portion of `OutTime` must equal `Time` exactly.
    if outer_time.normalize() != time.normalize() {
        return Err(ContractLaneError::OuterTimeMismatch {
            mode,
            expected: time.clone(),
            got: outer_time,
        });
    }

    // Recover the axes inner to the outermost reduce (dividing `pre_reduce_time` by post-reduce `time`).
    let division_terms =
        crate::config_divide_exact(pre_reduce_time, time).map_err(|_| ContractLaneError::InconsistentReduceTime {
            mode,
            pre_reduce_time: pre_reduce_time.clone(),
            time: time.clone(),
        })?;
    let time_padding_per_stride = padding_per_stride(pre_reduce_time);
    // A sub-term is dropped before the walk, not skipped inside it: the walk compares adjacent
    // boundaries, and skipping in place would still compare across the sub-term.
    let boundaries: Vec<(&DivideTerm, usize)> = division_terms
        .iter()
        .filter_map(|term| padded_extent_at(&time_padding_per_stride, term).map(|extent| (term, extent)))
        .collect();
    let dividend_end = |&(term, extent): &(&DivideTerm, usize)| term.dividend_stride * extent;
    let inner_time = if boundaries.is_empty() {
        // All axes reduced.
        1
    } else if dividend_end(&boundaries[0]) < pre_reduce_time.size() {
        // The outermost axis was reduced, so everything below the top is inner to the reduce.
        time.size()
    } else {
        // The outermost retained factor reaches the top; walk outer-to-inner to the first gap between
        // adjacent retained terms (the reduce boundary), else nothing is inner to the reduce.
        boundaries
            .windows(2)
            .find(|w| dividend_end(&w[1]) != w[0].0.dividend_stride)
            .map_or(1, |w| w[0].0.divisor_stride)
    };

    // Each `InnerTime` slot holds one `[Lane, Packet]` chunk; the `LaneMode` pads exactly one of the two
    // axes to a fixed width (Interleaved pads `Lane` to the 8-wide output bus, Sequential pads `Packet`
    // to the 32-column accumulator). The chunks for every inner-reduce position must fit the accumulator.
    let (padded_lane, padded_packet) = if interleaved {
        // Chunk = `[Lane # 8, Packet]`.
        (align_up(lane_size, CONTRACT_LANE_OUT_PACKET_ELEMENTS), packet.size())
    } else {
        // Chunk = `[Lane, Packet # 32]`.
        (lane_size, align_up(packet.size(), TEMPORAL_ACCUMULATOR_COLS))
    };
    if padded_lane * inner_time * padded_packet > ACCUMULATOR_CAPACITY_ELEMENTS {
        return Err(ContractLaneError::BufferExceeded {
            mode,
            padded_lane,
            inner_time,
            padded_packet,
            limit: ACCUMULATOR_CAPACITY_ELEMENTS,
        });
    }
    Ok(())
}

/// Splits `[outer, inner]` off `OutTime` and checks the inner portion equals the folded axes (`Packet`
/// Interleaved, `[Lane, packet_outer]` Sequential). The inner size must divide `OutTime`; a size that
/// does not divide it is a mismatch.
fn split_inner_time(out_time: &Mapping, inner: &Mapping, mode: LaneMode) -> Result<Mapping, ContractLaneError> {
    let mismatch = |got: Mapping| ContractLaneError::OutTimeMismatch {
        mode,
        expected: inner.clone(),
        got,
    };
    if !out_time.size().is_multiple_of(inner.size()) {
        return Err(mismatch(out_time.clone()));
    }
    let (outer_time, inner_time) = out_time.split_at(inner.size());
    if inner_time.normalize() != inner.normalize() {
        return Err(mismatch(inner_time));
    }
    Ok(outer_time)
}
