//! The switch-engine topology the caller picks and `config_switch` validates, plus its typed
//! failure reasons.

use core::fmt;

use abi_stable::StableAbi;
use furiosa_mapping_types::{Mapping, SequencerError};
use furiosa_opt_macro::primitive;

/// Configuration for the switch (slice/time topology) engine: the caller picks the topology and its
/// dimensions, and `config_switch` validates it against the input/output slice/time shapes. Lives in
/// `furiosa-opt-lower-types` (the engine re-exports it) so the `CustomBroadcast` divide-algebra stays
/// off the public engine surface. `#[primitive]` tags the variants for the visa MIR plugin.
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
        /// Ring group size for the custom routing. Must be a power of two: it feeds the SFR's
        /// `outer_dim0_size_log = ring_size.trailing_zeros()`, so a non-power-of-two misencodes.
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

/// A named slice sub-axis of the `(slice2, slice1, slice0)` partition a switch config
/// splits its input into. Labels which component failed the power-of-two check.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // The names are self-describing.
pub enum SwitchAxis {
    Slice0,
    Slice1,
    Slice2,
}

impl fmt::Display for SwitchAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SwitchAxis::Slice0 => "slice0",
            SwitchAxis::Slice1 => "slice1",
            SwitchAxis::Slice2 => "slice2",
        };
        f.write_str(s)
    }
}

/// The output frame a mismatched component lives in (the only frames an error reports).
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // The names are self-describing.
pub enum SwitchFrame {
    OutSlice,
    OutTime,
}

impl fmt::Display for SwitchFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SwitchFrame::OutSlice => "OutSlice",
            SwitchFrame::OutTime => "OutTime",
        };
        f.write_str(s)
    }
}

/// Why a switch is not realizable on the Switch engine. Named configs verify constructively,
/// so a positional failure is one [`SwitchError::OutputMismatch`] naming the offending frame;
/// `CustomBroadcast` keeps the general sequencer variants (ring size, broadcast newness, moved
/// order).
///
/// `#[repr(C)]` + `StableAbi` so the full diagnostic crosses the verifier `extern "C-unwind"`
/// boundary into `furiosa-opt-std` (like [`crate::FetchError`]) rather than a string.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SwitchError {
    /// Input and output slice volumes differ.
    #[error("Switch input and output slice sizes must match, got {in_slice} and {out_slice}")]
    SliceSizeMismatch {
        /// InSlice volume.
        in_slice: usize,
        /// OutSlice volume.
        out_slice: usize,
    },
    /// A `(slice1, slice0, time0)` dimension was zero.
    #[error("All dimensions must be greater than 0")]
    ZeroDimension,
    /// A slice sub-dimension is not a power of two (the switch network requires it).
    #[error("Switch {axis} must be a power of 2, got {value}")]
    NotPowerOfTwo {
        /// Which sub-axis.
        axis: SwitchAxis,
        /// Its size.
        value: usize,
    },
    /// InSlice volume is not a multiple of `slice1 * slice0`.
    #[error(
        "InSlice::SIZE must be divisible by (slice1 * slice0), got InSlice::SIZE {in_slice_size} (slice1 {slice1}, slice0 {slice0})"
    )]
    SliceNotDivisible {
        /// InSlice volume.
        in_slice_size: usize,
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// InTime's volume is not a multiple of the split the config's rule asks for, so the components it
    /// names do not exist. One variant for every named config, since the divisor is what differs.
    #[error(
        "Switch InTime::SIZE must be divisible by the config's time split, got InTime::SIZE {in_time_size} (split {by})"
    )]
    TimeNotDivisible {
        /// InTime volume.
        in_time_size: usize,
        /// The product of the time split the rule asks for.
        by: usize,
    },
    /// The actual output frame does not match the one the config reassembles from the
    /// input components (a moved/preserved component differs, or a tile slot is not a
    /// fresh broadcast). One comparison per output frame; `expected` shows tiles as
    /// `broadcast(n)`.
    #[error("Switch {frame} mismatch: expected {expected}, got {got}")]
    OutputMismatch {
        /// The output frame.
        frame: SwitchFrame,
        /// The output the `(input, config)` recipe reassembles (tiles shown as `broadcast(n)`).
        expected: Mapping,
        /// The actual output.
        got: Mapping,
    },
    /// The asserted ring size is not a power of two (the switch network requires it).
    #[error("Switch ring size must be a power of 2, got {ring_size}")]
    RingSizeNotPowerOfTwo {
        /// The asserted ring size.
        ring_size: usize,
    },
    /// The asserted ring size differs from the slot-sweep value.
    #[error("Switch ring size mismatch: the slot sweep computed {computed}, but {asserted} was asserted")]
    RingSizeMismatch {
        /// The slot-sweep-derived ring size.
        computed: usize,
        /// The asserted ring size.
        asserted: usize,
    },
    /// The `(in_slice, in_time) → (out_slice, out_time)` read does not sequence: an
    /// output slot has no input source, or a live input cell is dropped (padding may be
    /// trimmed, live data may not).
    #[error("Switch read is not sequenceable: {0:?}")]
    Unsequenceable(SequencerError),
    /// A broadcast tile lands on an output region that must not be written (a `Bottom` hole) or that must
    /// read zero (a `Zero` hole). Fanning data onto either is wrong whatever axis the tile names; a `Top` hole
    /// is the one a tile MAY extend into, since it is don't-care.
    #[error("Switch broadcast tile lands on {region}, which must not be written; only a `Top` hole is don't-care.")]
    OutputHasUnwritableRegion {
        /// The output slot the tile would fan data onto.
        region: Mapping,
    },
    /// A broadcast tile is an axis the input frame already carries. Reported by its mapping rather than
    /// an `Ident`: the check is per AXIS, so a tile may name a symbol the input uses and still be new (a
    /// different digit of it), and naming one identifier out of the window would point at the wrong thing.
    #[error("Switch broadcast tile {offending_axis} must be an axis the input Slice/Time does not carry.")]
    BroadcastNotNew {
        /// The tile window the input frame already carries.
        offending_axis: Mapping,
    },
    /// An OutSlice slot is sourced from InTime (a time→slice cross), not from InSlice or a
    /// fresh broadcast tile. Reported by its source mapping rather than an `Ident`, since a
    /// padding/broadcast time axis carries no identifier to name.
    #[error(
        "Switch OutSlice slot must source from InSlice or a fresh broadcast tile, not InTime; offending axis: {offending_axis}"
    )]
    OutSliceSourcedFromTime {
        /// The InTime-sourced axis that appeared in OutSlice.
        offending_axis: Mapping,
    },
    /// Two broadcast tiles claim the same axis, so one output cell would be sourced twice. Reported by the
    /// axis rather than an identifier: two DIGITS of one symbol are distinct axes and are allowed.
    #[error("Switch broadcast tiles both claim {offending_axis}; two tiles may not claim one axis.")]
    BroadcastUsedMoreThanOnce {
        /// The axis a second tile claimed.
        offending_axis: Mapping,
    },
    /// Slice→time axes are not at OutTime's innermost positions.
    #[error(
        "Switch axes moving from input slice to output time must be at the output time innermost positions (inner to InTime {time}); offending moved axis: {offending_axis}"
    )]
    MovedAxesNotInnermost {
        /// The InTime layout the moved axes must sit inner to.
        time: Mapping,
        /// The single InSlice-sourced axis that landed outer to an InTime axis.
        offending_axis: Mapping,
    },
    /// Slice→time axes lost their relative InSlice order in OutTime, reported as the
    /// offending pair of InSlice strides (an axis may be padding/broadcast, which has no
    /// `Term`, so the strides, not the terms, name the violation).
    #[error(
        "Switch axes moving from input Slice to output Time must preserve their relative order: {landed_inner} (InSlice stride {landed_inner_stride}) wrongly landed inner to {nested_under} (stride {nested_under_stride}); the larger stride must stay outer"
    )]
    MovedOrderNotPreserved {
        /// The moved axis that wrongly landed inner (the larger InSlice stride).
        landed_inner: Mapping,
        /// InSlice stride of `landed_inner`.
        landed_inner_stride: usize,
        /// The axis it nested under (the smaller InSlice stride).
        nested_under: Mapping,
        /// InSlice stride of `nested_under`.
        nested_under_stride: usize,
    },
    /// A switch carries InTime through to OutTime unchanged (including padding); only the
    /// axes moved from InSlice are added, at OutTime's innermost positions.
    #[error(
        "Switch input Time must be carried to output Time unchanged, got {in_time} but output Time carries {carried}"
    )]
    InTimeNotPreserved {
        /// InTime layout.
        in_time: Mapping,
        /// The InTime portion OutTime actually carries (OutTime with the moved-axes block
        /// stripped), not the full OutTime.
        carried: Mapping,
    },
    /// The sequencer-derived snoop bitmap disagreed with an independent enumeration at
    /// output slice `out_slice_lane`. This is the always-on soundness backstop on the snoop
    /// lanes that reach hardware, so a mismatch is a real attribution bug, surfaced as an
    /// error rather than a panic in the fallible bitmap path.
    #[error("Switch snoop bitmap disagrees with the (OutSlice x OutTime) enumeration at output slice {out_slice_lane}")]
    BitmapAttributionMismatch {
        /// The first output slice whose snooped input-lane set differs between the two derivations.
        out_slice_lane: usize,
    },
}
