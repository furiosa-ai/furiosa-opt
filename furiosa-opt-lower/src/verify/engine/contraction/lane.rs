//! Lane folder (`contract_lane`): fold `Lane` into the output stream.
//!
//! [`LaneMode`] selects the destination: `Interleaved` relocates `Lane` into `OutPacket`, `Sequential`
//! relocates it into `OutTime`. This mirrors the frontend-only `furiosa_opt_std` `LaneMode`; callers map
//! their mode onto this one so the check stays free of the frontend type.

use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Mapping, MappingExt};

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
    /// `OutTime` could not be decomposed into `[Time, Packet (truncated)]` (Interleaved).
    #[error(
        "contract_lane ({mode}): OutTime mismatch. Could not decompose OutTime {out_time} into \
         [Time, Packet (truncated)] where Time is {time} and Packet is a truncation of {packet}"
    )]
    OutTimeUndecomposable {
        /// The fold mode.
        mode: LaneMode,
        /// The declared output time.
        out_time: Mapping,
        /// The (post-reduce) time.
        time: Mapping,
        /// The padding-stripped input packet.
        packet: Mapping,
    },
    /// The inner portion of `OutTime` does not equal `[Lane, packet_outer]` (Sequential).
    #[error("contract_lane ({mode}): OutTime mismatch. Expected {expected}, got {got}")]
    OutTimeMismatch {
        /// The fold mode.
        mode: LaneMode,
        /// The expected inner portion `[Lane, packet_outer]`.
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
    /// The axes inner to the reduce exceed the accumulator buffer.
    #[error("contract_lane ({mode}): axes inner to reduce must be <= {limit} in size, got {buffer}")]
    BufferExceeded {
        /// The fold mode.
        mode: LaneMode,
        /// The buffer limit.
        limit: usize,
        /// The actual buffer size.
        buffer: usize,
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
    let packet = packet.clone().remove_padding();

    let (outer_time, packet_outer_size) = if interleaved {
        // `OutPacket = [Lane # 8]`.
        let expected_out_packet = lane
            .clone()
            .replace_padding(CONTRACT_LANE_OUT_PACKET_ELEMENTS)
            .normalize();
        let out_packet_n = out_packet.normalize();
        if out_packet_n != expected_out_packet {
            return Err(ContractLaneError::OutPacketMismatch {
                mode,
                expected: expected_out_packet,
                got: out_packet_n,
            });
        }

        // `OutTime = [Time, Packet (may be sliced)]`; search for the `Packet / Time` boundary.
        let packet_norm = packet.normalize();
        let outer_time = (1..=out_time.size().min(packet.size()).min(TEMPORAL_ACCUMULATOR_COLS))
            .filter(|&split| {
                out_time.size().is_multiple_of(split)
                    && (split > 1 || packet.size() == 1)
                    && out_time.size() / split <= time.size()
            })
            .find_map(|split| {
                let (outer_time, sliced_packet) = out_time.split_at(split);
                (sliced_packet.normalize() == packet_norm).then_some(outer_time)
            });
        let Some(outer_time) = outer_time else {
            return Err(ContractLaneError::OutTimeUndecomposable {
                mode,
                out_time: out_time.clone(),
                time: time.clone(),
                packet: packet.clone(),
            });
        };
        (outer_time, 1)
    } else {
        // `OutTime = [Time, Lane, packet_outer]`, `OutPacket = [packet_inner # 8]`.
        let padded = packet
            .clone()
            .replace_padding(align_up(packet.size(), CONTRACT_LANE_OUT_PACKET_ELEMENTS));
        let (packet_outer, packet_inner) = padded.split_at(CONTRACT_LANE_OUT_PACKET_ELEMENTS);
        let packet_outer_size = packet_outer.size();

        if packet_inner.normalize() != out_packet.normalize() {
            return Err(ContractLaneError::OutPacketMismatch {
                mode,
                expected: packet_inner,
                got: out_packet.normalize(),
            });
        }

        let lane_packet = lane.clone().pair(packet_outer);
        let (outer_time, inner_time) = out_time.split_at(lane_packet.size());
        if inner_time.normalize() != lane_packet.normalize() {
            return Err(ContractLaneError::OutTimeMismatch {
                mode,
                expected: lane_packet,
                got: inner_time,
            });
        }
        (outer_time, packet_outer_size)
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
    let padding_end = |d: &DivideTerm| d.dividend_stride * padded_extent_at(&time_padding_per_stride, d);
    let inner_time = if division_terms.is_empty() {
        // All axes reduced.
        1
    } else if padding_end(&division_terms[0]) < pre_reduce_time.size() {
        // The outermost axis was reduced, so everything below the top is inner to the reduce.
        time.size()
    } else {
        // The outermost retained factor reaches the top; walk outer-to-inner to the first gap between
        // adjacent retained terms (the reduce boundary), else nothing is inner to the reduce.
        division_terms
            .windows(2)
            .find(|w| padding_end(&w[1]) != w[0].dividend_stride)
            .map_or(1, |w| w[0].divisor_stride)
    };

    let (buffer, limit) = if interleaved {
        (
            inner_time * packet.size(),
            ACCUMULATOR_CAPACITY_ELEMENTS / CONTRACT_LANE_OUT_PACKET_ELEMENTS,
        )
    } else {
        (
            inner_time * lane_size * packet_outer_size,
            ACCUMULATOR_CAPACITY_ELEMENTS / TEMPORAL_ACCUMULATOR_COLS,
        )
    };
    if buffer > limit {
        return Err(ContractLaneError::BufferExceeded { mode, limit, buffer });
    }
    Ok(())
}
