//! Stream adapter of `contract_outer`: pack the innermost `Time` cells into `OutPacket` and broadcast.

use furiosa_mapping::{Mapping, MappingExt};

use crate::verify::{FLIT_BYTES, is_valid_lane_size, length_from_bytes, size_in_bytes};

/// Why a stream adapter is not realizable on `contract_outer`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamAdapterError {
    /// `Lane` is not 1, 2, 4, or 8.
    #[error("Lane::SIZE must be 1, 2, 4, or 8, got {0}")]
    LaneSize(usize),
    /// `OutPacket` is not one or two flits.
    #[error("OutPacket must be 1 or 2 flits ({FLIT_BYTES} bytes each), got {0} bytes")]
    OutPacketFlits(usize),
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

/// `OutPacket` = [packed innermost `Time` cells, input `Packet` flit]; `OutTime` = outer `Time` with
/// broadcast (tiling) axes innermost. `Lane` must be 1/2/4/8 and `OutPacket` one or two flits.
pub fn config_stream_adapter(
    lane: &Mapping,
    time: &Mapping,
    packet: &Mapping,
    out_time: &Mapping,
    out_packet: &Mapping,
    element_bits: usize,
) -> Result<(), StreamAdapterError> {
    if !is_valid_lane_size(lane.size()) {
        return Err(StreamAdapterError::LaneSize(lane.size()));
    }

    let out_packet_bytes = size_in_bytes(element_bits, out_packet.size());
    if !out_packet_bytes.is_multiple_of(FLIT_BYTES) || ![1, 2].contains(&(out_packet_bytes / FLIT_BYTES)) {
        return Err(StreamAdapterError::OutPacketFlits(out_packet_bytes));
    }

    // Packing pulls the innermost `pack_size` cells of `Time` into the packet.
    let pack_size = out_packet_bytes / FLIT_BYTES;
    let (time_outer, time_packed) = time.split_at(pack_size);

    // `OutPacket = [packed cells of Time, inner flit]`; the inner flit is the input `Packet`.
    let flit_elements = length_from_bytes(element_bits, FLIT_BYTES);
    let (out_packet_packed, out_packet_flit) = out_packet.split_at(flit_elements);
    if out_packet_flit.normalize() != packet.normalize() {
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
