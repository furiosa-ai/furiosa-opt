// `register_tool(furiosa_opt)` admits the `#[furiosa_opt::primitive = "…"]` markers the
// `#[primitive(SwitchConfig)]` macro emits, so the visa MIR plugin can translate the switch op.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]
//! Shared types for the furiosa-opt lowering engines.
//!
//! The sequencer descriptor pair, the per-engine errors, the transpose config, and the hardware
//! constants both the private impl (`npu-opt-lower-impl`) and the public `furiosa-opt-lower` wrapper
//! name. The errors and configs are `#[repr(C)]` + `StableAbi`, so the full diagnostic crosses the
//! verifier `extern "C-unwind"` boundary instead of collapsing to a string. Defined per-concern in the
//! submodules below and re-exported flat.

mod commit;
mod divide;
mod fetch;
mod sequencer;
mod switch;
mod tile;
mod transpose;

pub use commit::{COMMIT_BASE_SIZE, COMMIT_VALID_PACKET_SIZES, CommitError};
pub use divide::{DivideTerm, FactorLeaf, RelaxedDivision};
pub use fetch::FetchError;
pub use sequencer::{MAX_SEQUENCER_ENTRIES, StreamSequencerConfig};
pub use switch::{SwitchAxis, SwitchConfig, SwitchError, SwitchFrame};
pub use tile::{PadError, TileError};
pub use transpose::{TransposeConfig, TransposeError};
