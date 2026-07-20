//! Commit trim: the output packet must be a valid commit width and a trimming of the input packet.

use furiosa_mapping::{Mapping, MappingExt};
use furiosa_opt_lower_types::COMMIT_VALID_PACKET_SIZES;

use crate::verify::size_in_bytes;

/// Why a commit trim is not realizable on the Commit Adapter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommitTrimError {
    /// The output packet is not one of the hardware commit widths.
    #[error("commit_trim output packet must be one of {COMMIT_VALID_PACKET_SIZES:?} bytes, got {0}")]
    InvalidWidth(usize),
    /// The output packet is not a resize (trimming) of the input.
    #[error("commit_trim packet mismatch. Expected {packet} or a trimming of it, got {out_packet}")]
    PacketMismatch {
        /// The input packet.
        packet: Mapping,
        /// The declared output packet.
        out_packet: Mapping,
    },
}

/// The output packet must be one of the hardware commit widths and a resize (trimming) of the input.
pub fn config_commit_trim(packet: &Mapping, out_packet: &Mapping, element_bits: usize) -> Result<(), CommitTrimError> {
    let out_packet_bytes = size_in_bytes(element_bits, out_packet.size());
    if !COMMIT_VALID_PACKET_SIZES.contains(&out_packet_bytes) {
        return Err(CommitTrimError::InvalidWidth(out_packet_bytes));
    }
    if !out_packet.is_resize_of(packet) {
        return Err(CommitTrimError::PacketMismatch {
            packet: packet.clone(),
            out_packet: out_packet.clone(),
        });
    }
    Ok(())
}
