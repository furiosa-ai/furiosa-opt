//! The resolved transpose-engine hardware parameters.

use abi_stable::StableAbi;

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
