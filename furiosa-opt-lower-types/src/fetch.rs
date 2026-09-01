//! The Fetch engine's alignment constant and typed failure reasons.

use abi_stable::StableAbi;
use furiosa_mapping_types::{Mapping, SequencerError};

use crate::MAX_SEQUENCER_ENTRIES;

/// Supported fetch-base alignment for efficient SRAM access.
pub const FETCH_BASE_BYTES: usize = 8;
/// Supported Fetch cluster sizes.
pub const FETCH_VALID_CLUSTER_SIZES: [usize; 2] = [1, 2];
/// Supported Fetch slice sizes.
pub const FETCH_VALID_SLICE_SIZES: [usize; 3] = [64, 128, 256];

/// Why fetch bases cannot be materialized for the placement grid.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FetchBaseError {
    #[error("cannot locate fetch bases over the placement grid: {reason:?}")]
    Unlocatable { reason: SequencerError },
    #[error(
        "unaligned fetch base at chip/cluster/slice position {position}: {bits} bits; expected a \
         multiple of {FETCH_BASE_BYTES} bytes"
    )]
    UnalignedBase {
        /// The first unaligned chip x cluster x slice position.
        position: usize,
        /// The position's read offset in bits.
        bits: usize,
    },
    #[error(
        "the fetch base of chip/cluster/slice position {position} overflows: {base} elements at \
         {element_bits} bits each"
    )]
    BaseOffsetOverflow {
        position: usize,
        base: usize,
        element_bits: usize,
    },
}

/// Why a fetch is not realizable on the Fetch engine.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FetchError {
    /// Cluster size is unsupported.
    #[error("Fetch: Cluster size must be one of {FETCH_VALID_CLUSTER_SIZES:?}, got {0}")]
    ClusterSize(usize),
    /// Slice size is unsupported.
    #[error("Fetch: Slice size must be one of {FETCH_VALID_SLICE_SIZES:?}, got {0}")]
    SliceSize(usize),
    /// The matcher could not place an output axis against the DM memories.
    #[error("Fetch: cannot read an output axis from DM ({0:?})")]
    Unreadable(SequencerError),
    /// A live input axis was left unread (the carved-down DM remainders).
    #[error("Fetch: a live input axis is left unread (Time {time}, Packet {packet})")]
    Unread { time: Mapping, packet: Mapping },
    /// The packet's innermost axis is not contiguous in DM.
    #[error(
        "Fetch: the packet's innermost axis must be contiguous in DM, but {innermost} has memory \
         stride {memory_stride}; use the DM's innermost axis as the packet, or transpose after the \
         fetch"
    )]
    NonContiguousPacket {
        innermost: Mapping,
        /// Its stride in DM elements.
        memory_stride: usize,
    },
    /// The packet and time descriptors need more entries than the shared table holds.
    #[error("Fetch: needs {needed} sequencer entries, but the table holds {MAX_SEQUENCER_ENTRIES}")]
    TooManyEntries {
        /// The number of entries the two descriptors need.
        needed: usize,
    },
}
