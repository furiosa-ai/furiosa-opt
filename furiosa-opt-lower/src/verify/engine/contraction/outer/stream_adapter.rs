//! Stream adapter of `contract_outer`: pack the innermost `Time` cells into `OutPacket` and broadcast.

use furiosa_mapping::{Mapping, MappingExt};

use crate::verify::{ElementSizeError, FLIT_BYTES, is_valid_lane_size, length_from_bytes, size_in_bytes};

/// Why a stream adapter is not realizable on `contract_outer`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamAdapterError {
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
    /// `Lane` is not 1, 2, 4, or 8.
    #[error("Lane::SIZE must be 1, 2, 4, or 8, got {0}")]
    LaneSize(usize),
    /// `OutPacket` is not one or two flits.
    #[error("OutPacket must be 1 or 2 flits ({FLIT_BYTES} bytes each), got {0} bytes")]
    OutPacketFlits(usize),
    /// The requested packet cannot be packed from `Time`.
    #[error("`contract_outer`: cannot pack {pack_size} cells from Time {time}")]
    TimeNotPackable { pack_size: usize, time: Mapping },
    /// The inner flit of `OutPacket` does not equal the input `Packet`.
    #[error("`contract_outer`: inner flit of OutPacket must equal the input Packet")]
    FlitMismatch,
    /// The packed cells of `OutPacket` do not equal the innermost `Time` cells.
    #[error("`contract_outer`: OutPacket's packed cells must equal the innermost {0} cells of Time")]
    PackedCellsMismatch(usize),
    /// `Time` does not divide `OutTime`.
    #[error("`contract_outer`: Time does not divide OutTime")]
    TimeIndivisible,
    /// `OutTime` is not the outer `Time` with broadcast axes innermost.
    #[error("`contract_outer`: OutTime must be the outer Time with broadcast (tiling) axes innermost")]
    OutTimeMismatch,
}

/// Inputs used to configure the contraction stream adapter.
pub struct StreamAdapterInput {
    pub in_lane: Mapping,
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
    pub element_bits: usize,
}

/// `OutPacket` = [packed innermost `Time` cells, input `Packet` flit]; `OutTime` = outer `Time` with
/// broadcast (tiling) axes innermost. `Lane` must be 1/2/4/8 and `OutPacket` one or two flits.
pub fn config_stream_adapter(input: StreamAdapterInput) -> Result<(), StreamAdapterError> {
    let StreamAdapterInput {
        in_lane,
        in_time,
        in_packet,
        out_time,
        out_packet,
        element_bits,
    } = input;
    if !is_valid_lane_size(in_lane.size()) {
        return Err(StreamAdapterError::LaneSize(in_lane.size()));
    }

    let out_packet_bytes = size_in_bytes(element_bits, out_packet.size())?;
    if !out_packet_bytes.is_multiple_of(FLIT_BYTES) || ![1, 2].contains(&(out_packet_bytes / FLIT_BYTES)) {
        return Err(StreamAdapterError::OutPacketFlits(out_packet_bytes));
    }

    // Packing pulls the innermost `pack_size` cells of `Time` into the packet.
    let pack_size = out_packet_bytes / FLIT_BYTES;
    if !in_time.size().is_multiple_of(pack_size) {
        return Err(StreamAdapterError::TimeNotPackable {
            pack_size,
            time: in_time,
        });
    }
    let (time_outer, time_packed) = in_time.split_at(pack_size);

    // `OutPacket = [packed cells of Time, inner flit]`; the inner flit is the input `Packet`.
    let flit_elements = length_from_bytes(element_bits, FLIT_BYTES)?;
    let (out_packet_packed, out_packet_flit) = out_packet.split_at(flit_elements);
    if out_packet_flit.normalize() != in_packet.normalize() {
        return Err(StreamAdapterError::FlitMismatch);
    }
    if out_packet_packed.normalize() != time_packed.normalize() {
        return Err(StreamAdapterError::PackedCellsMismatch(pack_size));
    }

    // `OutTime = [outer Time, broadcast]`; stripping the tiling axes off the bottom leaves outer `Time`.
    if !out_time.size().is_multiple_of(time_outer.size()) {
        return Err(StreamAdapterError::TimeIndivisible);
    }
    let tiling_size = out_time.size() / time_outer.size();
    let (out_time_outer, _broadcast) = out_time.split_at(tiling_size);
    if out_time_outer.normalize() != time_outer.normalize() {
        return Err(StreamAdapterError::OutTimeMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![Lane = 1, Time = 3, Packet = 8, OutPacket = 16];

    #[test]
    fn rejects_time_that_cannot_be_split_into_packet_groups() {
        assert_eq!(
            config_stream_adapter(StreamAdapterInput {
                in_lane: <m![Lane]>::to_value(),
                in_time: <m![Time]>::to_value(),
                in_packet: <m![Packet]>::to_value(),
                out_time: <m![Time]>::to_value(),
                out_packet: <m![OutPacket]>::to_value(),
                element_bits: 32,
            }),
            Err(StreamAdapterError::TimeNotPackable {
                pack_size: 2,
                time: <m![Time]>::to_value(),
            })
        );
    }
}
