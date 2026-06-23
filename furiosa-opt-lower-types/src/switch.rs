//! The switch-engine topology the caller picks and `config_switch` validates.

use abi_stable::StableAbi;
use furiosa_opt_macro::primitive;

/// Configuration for the switch (slice/time topology) engine. The caller picks the topology and its
/// dimensions; `config_switch` validates it against the input/output slice/time shapes and echoes it
/// back. Moved here (from `furiosa-opt-std`) so the validation can run in the lowering impl, where the
/// `CustomBroadcast` divide-algebra stays internal; the engine re-exports it. `#[primitive]` tags the
/// variants so the visa MIR plugin can translate the switch op (it reads the markers from metadata).
#[primitive(SwitchConfig)]
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchConfig {
    /// Passes each slice's data through unchanged (`ring_size = 1`); `Slice` and `Time` preserved.
    Forwarding,
    /// Replicates data across slices along slice dimensions 0 and 1.
    Broadcast01 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
        /// Time dimension 0 size.
        time0: usize,
    },
    /// Replicates data across slices along slice dimension 1.
    Broadcast1 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// Swaps slice1 and slice0 in the slice dimension; time unchanged.
    Transpose {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// Swaps and transposes between the slice and time dimensions.
    InterTranspose {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
        /// Time dimension 0 size.
        time0: usize,
    },
    /// Routes data across slices using a custom snoop bitmap, computed by the compiler from the input
    /// shape and topology parameters.
    CustomBroadcast {
        /// Ring group size for the custom routing.
        ring_size: usize,
    },
    /// Swaps slice1/slice0 and replicates along slice dimension 0 (Transpose + Broadcast1 at once).
    TransposedBroadcast1 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
}
