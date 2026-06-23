//! The sequencer descriptor pair the Fetch and Commit engines both resolve.

use abi_stable::StableAbi;
use furiosa_mapping_types::SequencerConfig;

/// The sequencer config for a whole stream: the `(time, packet)` descriptors an engine resolves, the
/// fetch read or the commit write, each a [`SequencerConfig`] keyed by stream stride. Common to both.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct StreamSequencerConfig {
    /// The time descriptor (the outer loop levels).
    pub time: SequencerConfig,
    /// The packet descriptor (the contiguous inner DM run).
    pub packet: SequencerConfig,
}

/// The Fetch and Commit engines share one sequencer with an 8-entry table.
pub const MAX_SEQUENCER_ENTRIES: usize = 8;
