//! Packet reducer (`contract_packet`): contract a one/two-flit input to a power-of-two output packet.

use furiosa_mapping::{Mapping, MappingExt};

use crate::verify::{TEMPORAL_ACCUMULATOR_COLS, size_in_bytes};

/// Why a packet contraction is not realizable on the Packet Reducer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractPacketError {
    /// The input packet is not one or two flits.
    #[error("Packet must be 32 or 64 bytes, got {0} bytes")]
    InvalidPacketSize(usize),
    /// The output packet exceeds the accumulator width.
    #[error("OutPacket::SIZE must be at most {TEMPORAL_ACCUMULATOR_COLS}, got {0}")]
    OutPacketTooWide(usize),
    /// The output packet is not a power of two.
    #[error("OutPacket::SIZE must be a power of two, got {0}")]
    OutPacketNotPow2(usize),
    /// The output packet is not a prefix contraction of the input.
    #[error("OutPacket {out_packet} is not a valid contraction of Packet {packet}")]
    NotAContraction {
        /// The normalized output packet.
        out_packet: Mapping,
        /// The input packet.
        packet: Mapping,
    },
}

/// The input packet is one or two flits, the output packet is a power-of-two within the accumulator
/// width, and the output is a prefix-contraction of the input.
pub fn config_contract_packet(
    packet: &Mapping,
    out_packet: &Mapping,
    element_bits: usize,
) -> Result<(), ContractPacketError> {
    let packet_size = size_in_bytes(element_bits, packet.size());
    if ![32, 64].contains(&packet_size) {
        return Err(ContractPacketError::InvalidPacketSize(packet_size));
    }

    let out_packet_elems = out_packet.size();
    if out_packet_elems > TEMPORAL_ACCUMULATOR_COLS {
        return Err(ContractPacketError::OutPacketTooWide(out_packet_elems));
    }
    if !out_packet_elems.is_power_of_two() {
        return Err(ContractPacketError::OutPacketNotPow2(out_packet_elems));
    }

    // The Packet Reducer removes `ReducePacket`'s outermost padding, keeping only the live columns, so
    // `OutPacket` matches whether or not that padding is declared. `remove_padding` peels the outer spine,
    // so normalize first to surface it.
    let out_packet_norm = out_packet.normalize().remove_padding();
    let is_valid_contraction = (0..=packet.size().trailing_zeros()).any(|depth| {
        let split = 1usize << depth;
        packet.split_at(split).0.normalize().remove_padding() == out_packet_norm
    });
    if !is_valid_contraction {
        return Err(ContractPacketError::NotAContraction {
            out_packet: out_packet_norm,
            packet: packet.clone(),
        });
    }
    Ok(())
}
