//! Commit trim and commit cast: the widths one Commit Adapter write may use.

use furiosa_mapping::{Mapping, MappingExt};
use furiosa_opt_lower_types::{COMMIT_BASE_SIZE, COMMIT_VALID_PACKET_SIZES};

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

/// The post-trim packet is not a width a converting commit can write.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "commit_cast input packet is {commit_in_size} B, not a multiple of {unit} B; re-trim it to one \
     of {legal:?} B"
)]
pub struct CommitCastError {
    /// Post-trim packet width, in pre-cast bytes.
    pub commit_in_size: usize,
    /// Bytes one write covers: the commit unit's granularity times the conversion ratio.
    pub unit: usize,
    /// Hardware commit widths this conversion admits.
    pub legal: Vec<usize>,
}

/// The packet a commit cast receives, measured in its pre-cast element type, must be a width the
/// converting commit unit can write: it writes in [`COMMIT_BASE_SIZE`] units and produces
/// `in_bits / out_bits` times fewer bytes than it reads, so `f32 -> bf16` drops the odd multiples of
/// the base size. `commit_trim` produces that packet before the cast runs, so only here are both
/// widths known.
pub fn config_commit_cast(packet: &Mapping, in_bits: usize, out_bits: usize) -> Result<(), CommitCastError> {
    let unit = COMMIT_BASE_SIZE * (in_bits / out_bits);
    let commit_in_size = size_in_bytes(in_bits, packet.size());
    if commit_in_size.is_multiple_of(unit) {
        return Ok(());
    }
    Err(CommitCastError {
        commit_in_size,
        unit,
        legal: COMMIT_VALID_PACKET_SIZES
            .into_iter()
            .filter(|n| n.is_multiple_of(unit))
            .collect(),
    })
}
