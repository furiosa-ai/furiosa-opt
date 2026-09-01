// `register_tool(furiosa_opt)` admits the `#[furiosa_opt::primitive = "…"]` markers the
// `#[primitive(SwitchConfig)]` macro emits, so the visa MIR plugin can translate the switch op.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]
//! Shared types for the furiosa-opt lowering engines.
//!
//! These `StableAbi` values cross the boundary between the private lowering implementation and its
//! public wrapper.

mod commit;
mod divide;
mod fetch;
mod sequencer;
mod switch;
mod tile;
mod transpose;

pub use commit::{COMMIT_BASE_SIZE, COMMIT_VALID_PACKET_SIZES, CommitError};
pub use divide::{DivideError, DivideTerm, FactorLeaf, RelaxedDivision};
pub use fetch::{FETCH_BASE_BYTES, FETCH_VALID_CLUSTER_SIZES, FETCH_VALID_SLICE_SIZES, FetchBaseError, FetchError};
pub use sequencer::{MAX_SEQUENCER_ENTRIES, StreamSequencerConfig};
pub use switch::{SwitchAxis, SwitchConfig, SwitchError, SwitchFrame};
pub use tile::{PadError, TileError};
pub use transpose::{TransposeConfig, TransposeError};
