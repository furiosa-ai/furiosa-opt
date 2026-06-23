//! The Fetch engine's alignment constant and typed failure reasons.

use abi_stable::StableAbi;
use furiosa_mapping_types::{Mapping, SequencerError};

use crate::MAX_SEQUENCER_ENTRIES;

/// Output packet must be `FETCH_ALIGN_BYTES`-byte aligned.
pub const FETCH_ALIGN_BYTES: usize = 8;

/// Why a fetch is not realizable on the Fetch engine — one variant per `config_fetch` / frontend
/// `verify_fetch` check.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FetchError {
    /// Cluster size is not 1 or 2.
    #[error("Fetch: Cluster size must be 1 or 2, got {0}")]
    ClusterSize(usize),
    /// Slice size is not one of 64/128/192/256.
    #[error("Fetch: Slice size must be one of 64/128/192/256, got {0}")]
    SliceSize(usize),
    /// Output packet is not `FETCH_ALIGN_BYTES`-byte aligned.
    #[error("Fetch: output packet must be {FETCH_ALIGN_BYTES}-byte aligned, got {bytes} bytes")]
    PacketAlignment {
        /// The packet size, in bytes.
        bytes: usize,
    },
    /// The matcher could not place an output axis against the DM memories.
    #[error("Fetch: cannot read an output axis from DM ({0:?})")]
    Unreadable(SequencerError),
    /// A live input axis was left unread (the carved-down DM remainders).
    #[error("Fetch: a live input axis is left unread (Time {time}, Packet {packet})")]
    Unread {
        /// The carved-down `Time` memory remainder.
        time: Mapping,
        /// The carved-down `Packet` memory remainder.
        packet: Mapping,
    },
    /// The packet and time descriptors need more entries than the shared table holds.
    #[error("Fetch: needs {needed} sequencer entries, but the table holds {MAX_SEQUENCER_ENTRIES}")]
    TooManyEntries {
        /// The number of entries the two descriptors need.
        needed: usize,
    },
}
