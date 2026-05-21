//! TCP mapping expressions.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]

// Re-export these so that users only need to depend on this crate, not the individual crates.
pub use furiosa_mapping_macro::*;
pub use furiosa_mapping_types::*;

mod mapping;
pub use mapping::*;

mod ext;
pub use ext::*;
