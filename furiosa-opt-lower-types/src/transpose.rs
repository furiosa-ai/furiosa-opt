//! The transpose-engine hardware parameters and typed failure reasons.

use abi_stable::StableAbi;
use abi_stable::std_types::RBox;
use furiosa_mapping_types::{Mapping, SequencerError};

/// Resolved transpose-engine hardware parameters (all counts in elements).
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransposeConfig {
    /// Actual elements per output row (= OutPacket without the flit padding).
    pub in_rows: usize,
    /// Cols read per transpose tile (`packets_per_col * valid_size_per_packet`).
    pub in_cols: usize,
    /// Rows produced per output tile (≤ `in_cols`).
    pub out_rows: usize,
    /// Input packets consumed per col of source data.
    pub packets_per_col: usize,
    /// Live elements in each input packet.
    pub valid_size_per_packet: usize,
}

/// Why a transpose is not realizable on the Transpose engine.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TransposeError {
    /// The element width is not supported by the engine.
    #[error("Transpose: element_bits ({element_bits}) must be one of [4, 8, 16, 32]")]
    UnsupportedElementBits {
        /// The requested element width in bits.
        element_bits: usize,
    },
    /// The input packet is not one full flit.
    #[error(
        "Transpose input packet must be 32 bytes, got {elements} elements (expected {expected} \
         elements)"
    )]
    InputPacketSize {
        /// The actual number of elements.
        elements: usize,
        /// The required number of elements.
        expected: usize,
    },
    /// The output packet is not one full flit.
    #[error(
        "Transpose output packet must be 32 bytes, got {elements} elements (expected {expected} \
         elements)"
    )]
    OutputPacketSize {
        /// The actual number of elements.
        elements: usize,
        /// The required number of elements.
        expected: usize,
    },
    /// The unused input packet lanes are not pure padding.
    #[error(
        "Transpose: input packet beyond valid_size_per_packet ({valid_size_per_packet}) must be \
         `1 # n` pure padding, got: {padding}"
    )]
    InputPacketPadding {
        /// The number of valid elements per input packet.
        valid_size_per_packet: usize,
        /// The invalid outer packet lanes.
        padding: Mapping,
    },
    /// The unused output packet lanes are not pure padding.
    #[error(
        "Transpose: output packet beyond max_in_rows ({max_in_rows}) must be `1 # n` pure padding, \
         got: {padding}"
    )]
    OutputPacketPadding {
        /// The maximum number of input rows.
        max_in_rows: usize,
        /// The invalid outer packet lanes.
        padding: Mapping,
    },
    /// The supplied config uses the wrong valid packet width.
    #[error(
        "TransposeConfig valid_size_per_packet ({actual}) must equal the element-type packet width \
         ({expected})"
    )]
    ConfigPacketWidth {
        /// The configured valid packet width.
        actual: usize,
        /// The element-type packet width.
        expected: usize,
    },
    /// The supplied config's column count does not match its factors.
    #[error(
        "TransposeConfig in_cols ({in_cols}) must equal packets_per_col ({packets_per_col}) * \
         valid_size_per_packet ({valid_size_per_packet})"
    )]
    ConfigColumns {
        /// The configured input column count.
        in_cols: usize,
        /// The configured packet count per column.
        packets_per_col: usize,
        /// The configured valid packet width.
        valid_size_per_packet: usize,
    },
    /// The supplied config produces more rows than its input columns.
    #[error("TransposeConfig out_rows ({out_rows}) must be <= in_cols ({in_cols})")]
    ConfigRows {
        /// The configured output row count.
        out_rows: usize,
        /// The configured input column count.
        in_cols: usize,
    },
    /// The packet lanes beyond the configured valid width are not pure padding.
    #[error(
        "Transpose: Packet beyond valid_size_per_packet ({valid_size_per_packet}) must be `1 # n` \
         pure padding, got: {padding}"
    )]
    ConfigPacketPadding {
        /// The configured valid packet width.
        valid_size_per_packet: usize,
        /// The invalid outer packet lanes.
        padding: Mapping,
    },
    /// The configured packet count does not split Time.
    #[error("TransposeConfig packets_per_col ({packets_per_col}) does not split Time ({time})")]
    PacketsPerColDoesNotSplitTime {
        /// The configured packet count per column.
        packets_per_col: usize,
        /// The input Time mapping.
        time: Mapping,
    },
    /// The configured input row count does not split the remaining Time mapping.
    #[error("TransposeConfig in_rows ({in_rows}) does not split Time rest ({time})")]
    InRowsDoesNotSplitTime {
        /// The configured input row count.
        in_rows: usize,
        /// The remaining Time mapping.
        time: Mapping,
    },
    /// No input row placement satisfies both requested output mappings.
    #[error(
        "Transpose: cannot place in_rows in Time ({time}) consistently with OutTime ({out_time}) | \
         OutPacket ({out_packet})"
    )]
    CannotPlaceInRows {
        /// The input Time mapping.
        time: RBox<Mapping>,
        /// The requested output Time mapping.
        out_time: RBox<Mapping>,
        /// The requested output Packet mapping.
        out_packet: RBox<Mapping>,
    },
    /// The output row evidence cannot be matched in Time.
    #[error("Transpose row evidence ({row_evidence}) is not present in Time ({time}): {reason:?}")]
    RowEvidenceNotPresent {
        /// The live output row evidence.
        row_evidence: RBox<Mapping>,
        /// The input Time mapping.
        time: RBox<Mapping>,
        /// The sequencer failure.
        reason: SequencerError,
    },
    /// The output row evidence has a non-integral stride in Time.
    #[error("Transpose row evidence ({row_evidence}) is not aligned in Time ({time})")]
    RowEvidenceNotAligned {
        /// The live output row evidence.
        row_evidence: RBox<Mapping>,
        /// The input Time mapping.
        time: RBox<Mapping>,
    },
    /// Different row terms imply different packet counts per column.
    #[error(
        "Transpose row evidence ({row_evidence}) implies inconsistent packets_per_col sizes in \
         Time ({time})"
    )]
    InconsistentPacketsPerCol {
        /// The live output row evidence.
        row_evidence: RBox<Mapping>,
        /// The input Time mapping.
        time: RBox<Mapping>,
    },
    /// The output row evidence contains no live row terms.
    #[error("Transpose row evidence ({row_evidence}) has no live row terms")]
    NoLiveRowTerms {
        /// The output row evidence.
        row_evidence: Mapping,
    },
}
