//! Collect engine: normalize the packet to exactly one flit, and store to the TRF (`to_trf`) or
//! the VRF (`to_vrf`).

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};

use crate::verify::{
    ElementSizeError, FLIT_BYTES, OneFlitPacketError, PacketSide, VRF_BYTES, is_valid_lane_size, length_from_bytes,
    require_one_flit, size_in_bytes,
};

/// Why a collect is not realizable on the Collect engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CollectError {
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
    /// The output packet is not exactly one flit.
    #[error("Collect {0}")]
    PacketSize(#[from] OneFlitPacketError),
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
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
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
    #[error("Lane::SIZE ({stream_lane_size}) does not divide Time::SIZE ({stream_time_size})")]
    LaneDoesNotDivideTime {
        stream_lane_size: usize,
        stream_time_size: usize,
    },
    /// The outer factors of `Time` do not equal `Lane`.
    #[error("`to_trf` lane mismatch: time_outer != Lane: {stream_time_outer} != {stream_lane}")]
    LaneMismatch {
        stream_time_outer: Mapping,
        stream_lane: Mapping,
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
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
    /// The VRF data does not fit the register file.
    #[error("VRF data ({bytes} bytes) exceeds register file capacity ({capacity} bytes per slice)")]
    ExceedsCapacity {
        /// Byte size of one slice's `Element`.
        bytes: usize,
        /// Register file capacity in bytes.
        capacity: usize,
    },
    /// `Time` concatenated with `Packet` does not equal `Element`.
    #[error(
        "`to_vrf` element mismatch: the register holds the stream it stores, `[Time, Packet]` = \
         {expected}, but the declared `Element` is {got}. Name the stream the `collect` left, \
         padding included."
    )]
    ElementMismatch {
        /// `[Time, Packet]`.
        expected: Mapping,
        /// The declared `Element`.
        got: Mapping,
    },
}

/// Inputs used to configure the Collect engine.
pub struct CollectInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
    pub element_bits: usize,
}

/// Inputs used to configure a TRF store.
pub struct ToTrfInput {
    pub stream_lane: Mapping,
    pub stream_time: Mapping,
    pub stream_packet: Mapping,
    pub trf_element: Mapping,
    /// TRF capacity in bytes.
    pub capacity: usize,
    pub element_bits: usize,
}

/// Inputs used to configure a VRF store.
pub struct ToVrfInput {
    pub stream_time: Mapping,
    pub stream_packet: Mapping,
    pub vrf_element: Mapping,
    pub element_bits: usize,
}

/// Pads the input packet to a flit-aligned boundary, then splits at the flit: the inner flit must be
/// `out_packet`, and the outer portion folded onto `time` must be `out_time`.
pub fn config_collect(input: CollectInput) -> Result<(), CollectError> {
    let CollectInput {
        in_time,
        in_packet,
        out_time,
        out_packet,
        element_bits,
    } = input;
    let in_packet_bytes = size_in_bytes(element_bits, in_packet.size())?;
    let aligned_bytes = in_packet_bytes.div_ceil(FLIT_BYTES) * FLIT_BYTES;
    let flit_elements = length_from_bytes(element_bits, FLIT_BYTES)?;

    let out_packet_bytes = size_in_bytes(element_bits, out_packet.size())?;
    require_one_flit(PacketSide::Output, out_packet.size(), out_packet_bytes)?;

    let padded = in_packet.padding(length_from_bytes(element_bits, aligned_bytes)?, PaddingKind::Top);
    let (in_outer, in_flit) = padded.split_at(flit_elements);

    let expected_packet = in_flit.normalize();
    let out_packet = out_packet.normalize();
    if expected_packet != out_packet {
        return Err(CollectError::PacketMismatch {
            expected: expected_packet,
            got: out_packet,
        });
    }

    let expected_time = in_time.pair(in_outer).normalize();
    let out_time = out_time.normalize();
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
pub fn config_to_trf(input: ToTrfInput) -> Result<(), ToTrfError> {
    let ToTrfInput {
        stream_lane,
        stream_time,
        stream_packet,
        trf_element,
        capacity,
        element_bits,
    } = input;
    let lane_size = stream_lane.size();
    if !is_valid_lane_size(lane_size) {
        return Err(ToTrfError::LaneSize(lane_size));
    }

    let element_count = lane_size
        .checked_mul(trf_element.size())
        .ok_or(ElementSizeError::ElementCountOverflow {
            left: lane_size,
            right: trf_element.size(),
        })?;
    let total_trf_bytes = size_in_bytes(element_bits, element_count)?;
    if total_trf_bytes > capacity {
        return Err(ToTrfError::ExceedsCapacity {
            total_bytes: total_trf_bytes,
            lanes: lane_size,
            per_lane_bytes: size_in_bytes(element_bits, trf_element.size())?,
            capacity,
        });
    }

    let time_size = stream_time.size();
    if !time_size.is_multiple_of(lane_size) {
        return Err(ToTrfError::LaneDoesNotDivideTime {
            stream_lane_size: lane_size,
            stream_time_size: time_size,
        });
    }
    let (time_outer, time_inner) = stream_time.split_at(time_size / lane_size);
    let time_outer = time_outer.normalize();
    let lane_n = stream_lane.normalize();
    if time_outer != lane_n {
        return Err(ToTrfError::LaneMismatch {
            stream_time_outer: time_outer,
            stream_lane: lane_n,
        });
    }

    let expected_element = time_inner.pair(stream_packet).normalize();
    let element_n = trf_element.normalize();
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
///
/// `Element` is that stream and nothing else: the store writes one slice's `[Time, Packet]`, so a
/// declared `Element` naming other cells describes a register the store never wrote. The comparison
/// is on normalized mappings, which is what lets a kernel regroup the seam it does not care about
/// (`[m![B / 8], m![B % 8]]` declared as `m![B]`) while a shape that reaches other cells is refused.
/// [`config_to_trf`] states the same rule for the TRF, where `Lane` takes `Time`'s outer half first.
pub fn config_to_vrf(input: ToVrfInput) -> Result<(), ToVrfError> {
    let ToVrfInput {
        stream_time,
        stream_packet,
        vrf_element,
        element_bits,
    } = input;
    let bytes = size_in_bytes(element_bits, vrf_element.size())?;
    if bytes > VRF_BYTES {
        return Err(ToVrfError::ExceedsCapacity {
            bytes,
            capacity: VRF_BYTES,
        });
    }

    let expected = stream_time.pair(stream_packet).normalize();
    let got = vrf_element.normalize();
    if expected != got {
        return Err(ToVrfError::ElementMismatch { expected, got });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Glob import: the `m!` macro expands to DSL type-level structs that must all be in scope.
    use furiosa_mapping::*;

    use super::*;

    axes![B = 2048, C = 4096, E = 2, Odd = 1];

    #[test]
    fn collect_rejects_a_partial_byte() {
        assert_eq!(
            config_collect(CollectInput {
                in_time: Mapping::identity(),
                in_packet: <m![Odd]>::to_value(),
                out_time: Mapping::identity(),
                out_packet: <m![Odd]>::to_value(),
                element_bits: 4,
            }),
            Err(CollectError::ElementSize(ElementSizeError::NotByteAligned {
                elements: 1,
                element_bits: 4,
            }))
        );
    }

    /// One slice's operand filling the file exactly is the largest legal `to_vrf`.
    #[test]
    fn to_vrf_fills_file() {
        assert_eq!(
            config_to_vrf(ToVrfInput {
                stream_time: <m![B / 8]>::to_value(),
                stream_packet: <m![B % 8]>::to_value(),
                vrf_element: <m![B]>::to_value(),
                element_bits: 32,
            }),
            Ok(())
        );
    }

    /// The same element count in a wider type no longer fits, so the bound is on bytes and not on
    /// the element count.
    #[test]
    fn to_vrf_over_capacity_by_element_width() {
        assert_eq!(
            config_to_vrf(ToVrfInput {
                stream_time: <m![B / 8]>::to_value(),
                stream_packet: <m![B % 8]>::to_value(),
                vrf_element: <m![B]>::to_value(),
                element_bits: 64,
            }),
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
            config_to_vrf(ToVrfInput {
                stream_time: <m![C / 8]>::to_value(),
                stream_packet: <m![C % 8]>::to_value(),
                vrf_element: <m![C]>::to_value(),
                element_bits: 32,
            }),
            Err(ToVrfError::ExceedsCapacity {
                bytes: 16_384,
                capacity: VRF_BYTES,
            })
        );
    }

    /// The `Time` / `Packet` seam is the stream's, not the register's: an `Element` spelling it out
    /// names the same cells as one that folds it away, and both are the stream.
    #[test]
    fn to_vrf_element_regroups_the_stream() {
        assert_eq!(
            config_to_vrf(ToVrfInput {
                stream_time: <m![B / 8]>::to_value(),
                stream_packet: <m![B % 8]>::to_value(),
                vrf_element: <m![B / 8, B % 8]>::to_value(),
                element_bits: 32,
            }),
            Ok(())
        );
    }

    /// An `Element` that drops the stream's padding reaches fewer cells than the store wrote, so it
    /// is a different register and is refused here rather than at the VE op that reads it.
    #[test]
    fn to_vrf_element_dropping_stream_padding_rejects() {
        let error = config_to_vrf(ToVrfInput {
            stream_time: <m![1]>::to_value(),
            stream_packet: <m![E # 8]>::to_value(),
            vrf_element: <m![E]>::to_value(),
            element_bits: 32,
        })
        .unwrap_err();
        assert!(matches!(error, ToVrfError::ElementMismatch { .. }), "{error}");
    }
}
