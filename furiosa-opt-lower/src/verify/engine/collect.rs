//! Collect engine: normalize the packet to exactly one flit, and store to the TRF (`to_trf`) or
//! the VRF (`to_vrf`).

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};

use crate::verify::{FLIT_BYTES, VRF_BYTES, is_valid_lane_size, length_from_bytes, size_in_bytes};

/// Why a collect is not realizable on the Collect engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CollectError {
    /// The output packet is not exactly one flit.
    #[error("Collect output packet must be exactly {FLIT_BYTES} bytes (one flit).")]
    OutputNotOneFlit,
    /// The output packet is not the inner flit of the padded input.
    #[error("Collect packet mismatch. Expected: {expected}, got: {got}")]
    PacketMismatch {
        /// The inner flit of the padded input packet.
        expected: Mapping,
        /// The declared output packet.
        got: Mapping,
    },
    /// The output time is not the input time folded with the outer flit portion.
    #[error("Collect time mismatch. Expected: {expected}, got: {got}")]
    TimeMismatch {
        /// The input time folded with the outer flit portion.
        expected: Mapping,
        /// The declared output time.
        got: Mapping,
    },
}

/// Why a `to_trf` is not realizable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToTrfError {
    /// `Lane` is not 1, 2, 4, or 8.
    #[error("Lane::SIZE must be 1, 2, 4, or 8, got {0}")]
    LaneSize(usize),
    /// The TRF data does not fit the register file.
    #[error(
        "TRF data ({total_bytes} bytes = {lanes} lanes x {per_lane_bytes} bytes) exceeds register file capacity ({capacity} bytes)"
    )]
    ExceedsCapacity {
        /// Total byte size across all lanes.
        total_bytes: usize,
        /// Lane count.
        lanes: usize,
        /// Byte size of one lane's element.
        per_lane_bytes: usize,
        /// Register file capacity in bytes.
        capacity: usize,
    },
    /// `Lane::SIZE` does not divide `Time::SIZE`.
    #[error("Lane::SIZE ({lane}) does not divide Time::SIZE ({time})")]
    LaneDoesNotDivideTime {
        /// Lane size.
        lane: usize,
        /// Time size.
        time: usize,
    },
    /// The outer factors of `Time` do not equal `Lane`.
    #[error("`to_trf` lane mismatch: time_outer != Lane: {time_outer} != {lane}")]
    LaneMismatch {
        /// Outer factors of `Time`.
        time_outer: Mapping,
        /// The declared `Lane`.
        lane: Mapping,
    },
    /// The inner factors of `Time` concatenated with `Packet` do not equal `Element`.
    #[error("`to_trf` element mismatch: [time_inner, Packet] != Element: {expected} != {got}")]
    ElementMismatch {
        /// `[time_inner, Packet]`.
        expected: Mapping,
        /// The declared `Element`.
        got: Mapping,
    },
}

/// Why a `to_vrf` is not realizable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToVrfError {
    /// The VRF data does not fit the register file.
    #[error("VRF data ({bytes} bytes) exceeds register file capacity ({capacity} bytes per slice)")]
    ExceedsCapacity {
        /// Byte size of one slice's `Element`.
        bytes: usize,
        /// Register file capacity in bytes.
        capacity: usize,
    },
}

/// Pads the input packet to a flit-aligned boundary, then splits at the flit: the inner flit must be
/// `packet2`, and the outer portion folded onto `time` must be `time2`.
pub fn config_collect(
    time: &Mapping,
    packet: &Mapping,
    time2: &Mapping,
    packet2: &Mapping,
    element_bits: usize,
) -> Result<(), CollectError> {
    let in_packet_bytes = size_in_bytes(element_bits, packet.size());
    let aligned_bytes = in_packet_bytes.div_ceil(FLIT_BYTES) * FLIT_BYTES;
    let flit_elements = length_from_bytes(element_bits, FLIT_BYTES);

    let out_packet_bytes = size_in_bytes(element_bits, packet2.size());
    if out_packet_bytes != FLIT_BYTES {
        return Err(CollectError::OutputNotOneFlit);
    }

    let padded = packet
        .clone()
        .padding(length_from_bytes(element_bits, aligned_bytes), PaddingKind::Top);
    let (in_outer, in_flit) = padded.split_at(flit_elements);

    let expected_packet = in_flit.normalize();
    let out_packet = packet2.normalize();
    if expected_packet != out_packet {
        return Err(CollectError::PacketMismatch {
            expected: expected_packet,
            got: out_packet,
        });
    }

    let expected_time = time.clone().pair(in_outer).normalize();
    let out_time = time2.normalize();
    if expected_time != out_time {
        return Err(CollectError::TimeMismatch {
            expected: expected_time,
            got: out_time,
        });
    }
    Ok(())
}

/// `to_trf`: reshape the collected `[Time, Packet]` into the TRF `[Lane, Element]`.
///
/// `Lane` must be 1/2/4/8 and fit `capacity` bytes; the outer factors of `Time` must equal `Lane`,
/// and the remaining inner factors concatenated with `Packet` must equal `Element`.
pub fn config_to_trf(
    lane: &Mapping,
    time: &Mapping,
    packet: &Mapping,
    element: &Mapping,
    capacity: usize,
    element_bits: usize,
) -> Result<(), ToTrfError> {
    let lane_size = lane.size();
    if !is_valid_lane_size(lane_size) {
        return Err(ToTrfError::LaneSize(lane_size));
    }

    let total_trf_bytes = size_in_bytes(element_bits, lane_size * element.size());
    if total_trf_bytes > capacity {
        return Err(ToTrfError::ExceedsCapacity {
            total_bytes: total_trf_bytes,
            lanes: lane_size,
            per_lane_bytes: size_in_bytes(element_bits, element.size()),
            capacity,
        });
    }

    let time_size = time.size();
    if !time_size.is_multiple_of(lane_size) {
        return Err(ToTrfError::LaneDoesNotDivideTime {
            lane: lane_size,
            time: time_size,
        });
    }
    let (time_outer, time_inner) = time.split_at(time_size / lane_size);
    let time_outer = time_outer.normalize();
    let lane_n = lane.normalize();
    if time_outer != lane_n {
        return Err(ToTrfError::LaneMismatch {
            time_outer,
            lane: lane_n,
        });
    }

    let expected_element = time_inner.pair(packet.clone()).normalize();
    let element_n = element.normalize();
    if expected_element != element_n {
        return Err(ToTrfError::ElementMismatch {
            expected: expected_element,
            got: element_n,
        });
    }
    Ok(())
}

/// `to_vrf`: store the collected stream as the VRF `[Element]`.
///
/// One slice's `Element` must fit the vector register file. The VRF is not partitioned by an address
/// the way the TRF is, so the capacity is the whole file ([`VRF_BYTES`]).
pub fn config_to_vrf(element: &Mapping, element_bits: usize) -> Result<(), ToVrfError> {
    let bytes = size_in_bytes(element_bits, element.size());
    if bytes > VRF_BYTES {
        return Err(ToVrfError::ExceedsCapacity {
            bytes,
            capacity: VRF_BYTES,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Glob import: the `m!` macro expands to DSL type-level structs that must all be in scope.
    use furiosa_mapping::*;

    use super::*;

    axes![B = 2048, C = 4096];

    /// One slice's operand filling the file exactly is the largest legal `to_vrf`.
    #[test]
    fn to_vrf_at_capacity() {
        assert_eq!(config_to_vrf(&<m![B]>::to_value(), 32), Ok(()));
    }

    /// The same element count in a wider type no longer fits, so the bound is on bytes and not on
    /// the element count.
    #[test]
    fn to_vrf_over_capacity_by_element_width() {
        assert_eq!(
            config_to_vrf(&<m![B]>::to_value(), 64),
            Err(ToVrfError::ExceedsCapacity {
                bytes: 16_384,
                capacity: VRF_BYTES,
            })
        );
    }

    /// A larger axis overruns the file even in a type that fits at half the count.
    #[test]
    fn to_vrf_over_capacity_by_axis_size() {
        assert_eq!(
            config_to_vrf(&<m![C]>::to_value(), 32),
            Err(ToVrfError::ExceedsCapacity {
                bytes: 16_384,
                capacity: VRF_BYTES,
            })
        );
    }
}
