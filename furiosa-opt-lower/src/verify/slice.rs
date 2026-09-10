//! Slice and redistribution verification.

mod config;
mod pattern;

pub use config::{SliceError, SlicePlan, SliceRequest, config_outermost_dm_slice, config_slice};
pub use pattern::{
    ClusterPlacement, SlicePatternError, SramRedistributeEntry, validate_chip_shuffle, validate_slice_indices,
    validate_sram_redistribution,
};
