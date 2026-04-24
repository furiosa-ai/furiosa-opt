//! Crate aliases for mdbook test
//!
//! HACK: This crate contains crate aliases to avoid mdbook failures due to multiple rlibs found in the target
//! directory.  https://github.com/rust-lang/mdBook/issues/394

use ::rand as original_rand;

pub mod rand {
    pub use crate::original_rand::*;
}
