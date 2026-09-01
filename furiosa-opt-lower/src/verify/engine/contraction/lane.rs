//! Lane folder (`contract_lane`): fold `Lane` into the output stream.
//!
//! [`LaneMode`] selects the destination: `Interleaved` relocates `Lane` into `OutPacket`, `Sequential`
//! relocates it into `OutTime`. This mirrors the frontend-only `furiosa_opt_std` `LaneMode`; callers map
//! their mode onto this one so the check stays free of the frontend type.

use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};

use crate::DivideError;
use crate::verify::{
    CONTRACT_LANE_OUT_PACKET_ELEMENTS, TEMPORAL_ACCUMULATOR_COLS, align_up, inner_reduce_extent, is_valid_lane_size,
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

/// Why a lane fold is not realizable on the Lane Folder.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractLaneError {
    /// The lane count is unsupported.
    #[error("contract_lane: Lane::SIZE must be 1, 2, 4, or 8, got {0}")]
    LaneSize(usize),
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
    #[error("contract_lane ({mode}): post-reduce Time {time} must divide pre-reduce Time {pre_reduce_time}: {source}")]
    InconsistentReduceTime {
        /// The fold mode.
        mode: LaneMode,
        /// The pre-reduce time.
        pre_reduce_time: Mapping,
        time: Mapping,
        #[source]
        source: DivideError,
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

/// Inputs used to configure a lane contraction.
pub struct ContractLaneInput {
    pub in_lane: Mapping,
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
    pub pre_reduce_time: Mapping,
    pub mode: LaneMode,
}

impl Display for LaneMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Interleaved => "Interleaved",
            Self::Sequential => "Sequential",
        })
    }
}

/// `contract_lane` performs no reduction, so the outer portion of `OutTime` must equal `Time`, the
/// output packet must be one flit, and the axes inner to the outermost reduce must fit the accumulator.
pub fn config_contract_lane(input: ContractLaneInput) -> Result<(), ContractLaneError> {
    let ContractLaneInput {
        in_lane,
        in_time,
        in_packet,
        out_time,
        out_packet,
        pre_reduce_time,
        mode,
    } = input;
    if in_packet.size() > TEMPORAL_ACCUMULATOR_COLS {
        return Err(ContractLaneError::PacketTooWide(in_packet.size()));
    }
    if out_packet.size() != CONTRACT_LANE_OUT_PACKET_ELEMENTS {
        return Err(ContractLaneError::OutPacketSize(out_packet.size()));
    }

    let lane_size = in_lane.size();
    if !is_valid_lane_size(lane_size) {
        return Err(ContractLaneError::LaneSize(lane_size));
    }

    let (outer_time, padded_lane, padded_packet) = match mode {
        LaneMode::Interleaved => {
            let expected_out_packet = in_lane
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

            (
                split_inner_time(&out_time, &in_packet, mode)?,
                align_up(lane_size, CONTRACT_LANE_OUT_PACKET_ELEMENTS),
                in_packet.size(),
            )
        }
        LaneMode::Sequential => {
            let padded = in_packet.clone().padding(
                align_up(in_packet.size(), CONTRACT_LANE_OUT_PACKET_ELEMENTS),
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

            (
                split_inner_time(&out_time, &in_lane.clone().pair(packet_outer), mode)?,
                lane_size,
                align_up(in_packet.size(), TEMPORAL_ACCUMULATOR_COLS),
            )
        }
    };

    // The post-split outer portion of `OutTime` must equal `Time` exactly.
    if outer_time.normalize() != in_time.normalize() {
        return Err(ContractLaneError::OuterTimeMismatch {
            mode,
            expected: in_time.clone(),
            got: outer_time,
        });
    }

    // The axes inner to the outermost reduce, one buffer slot each.
    let inner_time = inner_reduce_extent(&pre_reduce_time, &in_time).map_err(|source| {
        ContractLaneError::InconsistentReduceTime {
            mode,
            pre_reduce_time: pre_reduce_time.clone(),
            time: in_time.clone(),
            source,
        }
    })?;

    // Every inner-reduce position retains one hardware-padded `[Lane, Packet]` chunk.
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

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![Lane = 3, Packet = 8];

    #[test]
    fn rejects_an_unsupported_lane_size_before_padding() {
        assert_eq!(
            config_contract_lane(ContractLaneInput {
                in_lane: <m![Lane]>::to_value(),
                in_time: Mapping::identity(),
                in_packet: <m![Packet]>::to_value(),
                out_time: Mapping::identity(),
                out_packet: <m![Packet]>::to_value(),
                pre_reduce_time: Mapping::identity(),
                mode: LaneMode::Interleaved,
            }),
            Err(ContractLaneError::LaneSize(3))
        );
    }
}
