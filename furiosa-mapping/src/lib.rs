//! TCP mapping expressions.

#![feature(register_tool)]
#![register_tool(tcp)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]

pub use furiosa_mapping_types::*;

mod mapping;
pub use mapping::*;

pub mod parser;

mod ext;
pub use ext::*;
