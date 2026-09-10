//! Control-flow shaping primitives.

use furiosa_opt_macro::primitive;

/// Carries `#[unroll]` from macro expansion to the device translator.
/// The marker is a call because rustc does not preserve loop attributes in MIR.
#[doc(hidden)]
#[primitive(loop_hint::unroll)]
pub fn __loop_hint_unroll() {}
