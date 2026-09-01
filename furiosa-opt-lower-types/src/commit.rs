//! The Commit engine's legal packet sizes and typed failure reasons.

use abi_stable::StableAbi;
use abi_stable::std_types::{RBox, RVec};
use furiosa_mapping_types::{Mapping, SequencerError};

use crate::MAX_SEQUENCER_ENTRIES;

/// Bytes a commit packet may carry — a single flit (32 B) trimmed to one of these.
pub const COMMIT_VALID_PACKET_SIZES: [usize; 4] = [8, 16, 24, 32];

/// Granularity of one commit write, in bytes (the SRAM access width). A commit that also converts
/// writes half as many bytes as it reads, so its input width must be a multiple of this times the
/// conversion ratio.
pub const COMMIT_BASE_SIZE: usize = 8;

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
    /// The trimmed packet is not the DM `element`'s innermost run: the engine drains one packet into the
    /// element's inner cells, so the element split at the packet size must give the packet back. The
    /// mappings are boxed to keep the whole error under the large-`Err` budget.
    #[error(
        "Commit: the trimmed packet {} must be the innermost run of the DM element {}, but that \
         element's innermost {} cells are {}. If the element's layout is the intended one, the stream \
         has to reach the commit in that order: run it through the transpose engine so the packet \
         becomes the element's innermost run",
        &**packet, &**dm_element, packet.size(), &**dm_element_inner
    )]
    PacketNotInnermost {
        packet: RBox<Mapping>,
        dm_element: RBox<Mapping>,
        dm_element_inner: RBox<Mapping>,
    },
    /// The matcher could not place a time axis into the DM `element` layout (the packet placed fine).
    /// The mappings are boxed to keep the whole error under the large-`Err` budget.
    #[error("Commit: cannot write the time axes {} into the DM element {} ({reason:?})", &**time, &**dm_element)]
    Unwritable {
        /// Why the matcher gave up.
        reason: SequencerError,
        /// The stream's time axes.
        time: RBox<Mapping>,
        dm_element: RBox<Mapping>,
    },
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
