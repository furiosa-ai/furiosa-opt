//! The Commit engine's legal packet sizes and typed failure reasons.

use abi_stable::StableAbi;
use abi_stable::std_types::RVec;
use furiosa_mapping_types::{Mapping, SequencerError};

use crate::MAX_SEQUENCER_ENTRIES;

/// Bytes a commit packet may carry — a single flit (32 B) trimmed to one of these.
pub const COMMIT_VALID_PACKET_SIZES: [usize; 4] = [8, 16, 24, 32];

/// Why a commit is not realizable on the Commit engine — one variant per `config_commit` check.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommitError {
    /// The input packet, in bytes, is not a legal commit flit size.
    #[error("Commit: input packet must be one of {COMMIT_VALID_PACKET_SIZES:?} bytes, got {bytes}")]
    IllegalPacketBytes {
        /// The packet size, in bytes.
        bytes: usize,
    },
    /// The matcher could not place an input axis into the DM `element` layout.
    #[error("Commit: cannot write an input axis into DM ({0:?})")]
    Unwritable(SequencerError),
    /// A DM cell is left unwritten (the carved-down memory remainders).
    #[error("Commit: a DM cell is left unwritten ({0:?})")]
    Unwritten(RVec<Mapping>),
    /// The packet and time descriptors need more entries than the shared table holds.
    #[error("Commit: needs {needed} sequencer entries, but the table holds {MAX_SEQUENCER_ENTRIES}")]
    TooManyEntries {
        /// The number of entries the two descriptors need.
        needed: usize,
    },
}
