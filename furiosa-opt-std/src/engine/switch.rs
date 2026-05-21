//! Switch Engine: slice/time topology rearrangements.
//!
//! Applies a switching-network routing to a `FetchTensor`. The packet passes
//! through unchanged — use [`super::collect`] afterwards to normalize the packet
//! to flit-sized chunks.

use std::collections::HashSet;

use abi_stable::std_types::{RBox, Tuple2};
use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::{CanApplySwitch, exact_div};
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::tu::{Position, TuTensor};

/// After the switch engine.
#[derive(Debug)]
pub struct PositionSwitch;

impl Position for PositionSwitch {}

/// Tensor streamed after the switch engine.
pub type SwitchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionSwitch, D, Chip, Cluster, Slice, Time, Packet, B>;

/// Configuration for the `switch` operation.
#[derive(Debug, Clone)]
pub enum SwitchConfig {
    /// Replicates data across slices along slice dimensions 0 and 1.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | tile | tile\]
    /// Time:  \[time1 | time0\] to \[time1 | slice1 | time0 | slice0\]
    Broadcast01 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
        /// Time dimension 0 size.
        time0: usize,
    },
    /// Replicates data across slices along slice dimension 1.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | tile | slice0\]
    /// Time:  \[time0\] to \[time0 | slice1\]
    Broadcast1 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// Swaps slice1 and slice0 dimensions in the slice dimension. Time is unchanged.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | slice0 | slice1\]
    Transpose {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// Swaps and transposes between the slice and time dimensions.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | time1 | slice0\]
    /// Time:  \[time2 | time1 | time0\] to \[time2 | time0 | slice1\]
    InterTranspose {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
        /// Time dimension 0 size.
        time0: usize,
    },
    /// Routes data across slices using a custom snoop bitmap.
    /// The bitmap is computed by the compiler from the input shape and topology
    /// parameters.
    CustomBroadcast {
        /// Ring group size for the custom routing.
        ring_size: usize,
    },
    /// Swaps slice1 and slice0 dimensions in the slice dimension and replicates
    /// data acorss slices along slice dimension 0.  The behavior is equivalent
    /// to applying `Transpose` and `Broadcast1` at once.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | tile | slice1\]
    /// Time:  \[time0\] to \[time0 | slice0\]
    TransposedBroadcast1 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
}

impl SwitchConfig {
    /// Verifies that the switch configuration is compatible with the provided
    /// input and output slice/time mappings.
    pub fn verify<InSlice: M, InTime: M, OutSlice: M, OutTime: M>(&self) {
        match self {
            SwitchConfig::Broadcast01 { slice1, slice0, time0 } => {
                assert!(
                    [slice1, slice0, time0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );
                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );
                assert_eq!(InTime::SIZE % time0, 0, "InTime::SIZE must be divisible by time0");

                // m![InSlice / (slice1 * slice0)] = m![OutSlice / (slice1 * slice0)]
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );

                // Broadcast axes in OutSlice should be new.
                if *slice1 * *slice0 > 1 {
                    let out_slice_broadcast_symbols = extract_symbols(&Mapping::Modulo {
                        inner: RBox::new(OutSlice::to_value()),
                        modulo: *slice1 * *slice0,
                    });

                    let mut symbols = HashSet::new();
                    symbols.extend(extract_symbols(&InSlice::to_value()));
                    symbols.extend(extract_symbols(&InTime::to_value()));
                    assert_eq!(
                        out_slice_broadcast_symbols.intersection(&symbols).count(),
                        0,
                        "OutSlice broadcast axes must be new axes"
                    );
                }

                let mut expected_out_time = Mapping::Identity
                    // time1 = m![InTime / time0]
                    .pair(Mapping::Stride {
                        inner: RBox::new(InTime::to_value()),
                        stride: *time0,
                    })
                    // slice1 = m![InSlice / slice0 % slice1]
                    .pair(Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(InSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    });
                if *time0 > 1 {
                    // time0 = m![InTime % time0]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(InTime::to_value()),
                        modulo: *time0,
                    })
                }
                if *slice0 > 1 {
                    // slice0 = m![InSlice % slice0]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(InSlice::to_value()),
                        modulo: *slice0,
                    })
                }

                // OutTime must match [time1, slice1, time0, slice0]
                assert_eq!(
                    expected_out_time.factorize(),
                    OutTime::to_value().factorize(),
                    "OutTime does not match expected layout: [time1, slice1, time0, slice0]"
                );
            }

            SwitchConfig::Broadcast1 { slice1, slice0 } => {
                assert!(
                    [slice1, slice0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );
                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );

                // m![InSlice / (slice1 * slice0)] = m![OutSlice / (tile * slice0)]
                // Since tile has the same size as slice1, this is: m![InSlice / (slice1 * slice0)] = m![OutSlice / (slice1 * slice0)]
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );

                // Broadcast axes in OutSlice should be new.
                if *slice1 > 1 {
                    let out_slice_broadcast_symbols = extract_symbols(&Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(OutSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    });

                    let mut symbols = HashSet::new();
                    symbols.extend(extract_symbols(&InSlice::to_value()));
                    symbols.extend(extract_symbols(&InTime::to_value()));
                    assert_eq!(
                        out_slice_broadcast_symbols.intersection(&symbols).count(),
                        0,
                        "OutSlice broadcast axes must be new axes"
                    );
                }

                // slice0 is preserved at the innermost part of OutSlice
                if *slice0 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(InSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        "OutSlice must preserve slice0 from InSlice",
                    );
                }

                // OutTime must match [time0, slice1]
                let mut expected_out_time = Mapping::Identity.pair(InTime::to_value());
                if *slice1 > 1 {
                    // slice1 = m![InSlice / slice0 % slice1]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(InSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    });
                }

                assert_eq!(
                    expected_out_time.factorize(),
                    OutTime::to_value().factorize(),
                    "OutTime does not match expected layout: [time0, slice1]"
                );
            }

            SwitchConfig::Transpose { slice1, slice0 } => {
                assert!(
                    [slice1, slice0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );

                // Time can be padded, but padding must not change the size.
                assert_eq!(
                    InTime::SIZE,
                    OutTime::SIZE,
                    "Input and output time dimensions must have the same size"
                );
                assert_eq!(
                    InTime::to_value().factorize().remove_padding(),
                    OutTime::to_value().factorize().remove_padding(),
                    "Input and output time dimensions must match (excluding padding)"
                );

                // OutSlice must match [slice2, slice0, slice1] (slice1 and slice0 are swapped)
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .pair(Mapping::Modulo {
                        inner: RBox::new(InSlice::to_value()),
                        modulo: *slice0,
                    })
                    .pair(Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(InSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    })
                    .factorize(),
                    OutSlice::to_value().factorize(),
                    "OutSlice does not match expected layout: [slice2, slice0, slice1]"
                );
            }

            SwitchConfig::InterTranspose { slice1, slice0, time0 } => {
                assert!(
                    [slice1, slice0, time0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );

                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );
                assert_eq!(
                    InTime::SIZE % (slice1 * time0),
                    0,
                    "InTime::SIZE must be divisible by (slice1 * time0)"
                );

                let slice2 = InSlice::SIZE / (slice1 * slice0);
                let time2 = InTime::SIZE / (slice1 * time0);

                assert_eq!(slice2 * slice1 * slice0, 256, "All dimensions should multiply to 256");
                assert_eq!(
                    time2 * slice1 * time0,
                    InTime::SIZE,
                    "time2 * slice1 * time0 must equal InTime::SIZE"
                );

                // InSlice  = [slice2, slice1,  slice0]
                // OutSlice = [slice2, time1, slice0]
                // slice2 and slice0 are preserved
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );
                if *slice0 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(InSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        "OutSlice must preserve slice0 from InSlice",
                    );
                }

                // InTime   = [time2,  time1, time0]
                // OutSlice = [slice2, time1, slice0]
                // time1 comes from InTime
                if *slice1 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(OutSlice::to_value()),
                                stride: *slice0,
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(InTime::to_value()),
                                stride: *time0,
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        "OutSlice time1 must come from InTime"
                    );
                }

                // InTime  = [time2, time1, time0]
                // OutTime = [time2, time0,  slice1]
                // time2 is preserved
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(OutTime::to_value()),
                        stride: *slice1 * time0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(InTime::to_value()),
                        stride: *time0 * slice1,
                    }
                    .factorize(),
                    "OutTime must preserve 'time2' from InTime",
                );
                // time0 is moved from InTime to OutTime
                if *time0 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(OutTime::to_value()),
                                stride: *slice1,
                            }),
                            modulo: *time0,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(InTime::to_value()),
                            modulo: *time0,
                        }
                        .factorize(),
                        "OutTime must preserve 'time0' from InTime"
                    );
                }
                // InSlice = [slice2, slice1, slice0]
                // OutTime = [time2,  time0,   slice1]
                // slice1 in OutTime matches the one in InSlice
                if *slice1 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutTime::to_value()),
                            modulo: *slice1,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(InSlice::to_value()),
                                stride: *slice0,
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        "OutTime must preserve 'slice1' from InSlice"
                    );
                }
            }

            // Custom topologies allow:
            // 1. Arbitrary slice permutations in Slice.
            // 2. Slice to Time broadcasts with slicing.

            // Constraints checked:
            // 1. Broadcast axes must use new unique axes (not in input `Slice` or `Time`).
            // 2. Broadcast axes should not be padded.
            // 3. Outer portion of output `Time` must match input `Time`.
            // 4. Axes moving from `Slice` to `Time` appear at the innermost output `Time` positions.
            // 5. Axes moving from `Slice` to `Time` must preserve their relative order from input `Slice`.
            // 6. Ring size must match the outermost non-directcast boundary and be a power of 2.
            SwitchConfig::CustomBroadcast { ring_size } => {
                assert!(
                    ring_size.is_power_of_two(),
                    "Switch ring size must be a power of 2, got {ring_size}"
                );

                let slice = InSlice::to_value().factorize();
                let out_slice = OutSlice::to_value().factorize();
                let time = InTime::to_value().factorize();
                let out_time = OutTime::to_value().factorize();

                // Identify broadcast axes. Broadcast axes use new axes, not
                // present in InSlice.
                let division = out_slice.clone().divide(slice.clone());
                let Residues {
                    dividend_residue,
                    divisor_residue,
                } = division.relaxed_residues();
                assert!(
                    dividend_residue
                        .clone()
                        .divide(slice.clone().mul(time.clone()))
                        .division_terms
                        .is_empty(),
                    "Switch broadcast axes must be new axes (not present in input Slice or Time)."
                );

                // Each broadcast axis must be used exactly once in OutSlice.
                // Count surviving Term factors in the dividend residue;
                // `idents()` returns unique axes. A mismatch means an axis
                // appears more than once.
                let dividend_unmatched_term_count = dividend_residue
                    .factors()
                    .iter()
                    .filter(|f| matches!(f, Factor::Term { .. }))
                    .count();
                assert_eq!(
                    dividend_unmatched_term_count,
                    dividend_residue.idents().len(),
                    "Switch broadcast axes must each be used exactly once in OutSlice"
                );

                // Broadcast axes must not have padding.
                let factors = out_slice.factors();
                for (i, factor) in factors.iter().enumerate() {
                    if let Factor::Term { inner, .. } = factor
                        && !division.division_terms.iter().any(|d| d.term.inner == inner.inner)
                        && matches!(factors.get(i + 1), Some(Factor::Padding { .. }))
                    {
                        panic!("Switch broadcast axis {inner} in output Slice must not be padded.");
                    }
                }

                // OutTime = [InTime (outer) | moved axes (inner)].
                let Tuple2(outer, inner) =
                    out_time.split_at(exact_div(OutTime::SIZE, InTime::SIZE).unwrap_or_else(|| {
                        panic!(
                            "Input Time size ({}) does not divide Output Time size ({})",
                            InTime::SIZE,
                            OutTime::SIZE
                        )
                    }));
                assert_eq!(
                    outer, time,
                    "Switch axes moving from input slice to output time must be at the output time innermost positions. \
                     Expected outer portion of output time to be {time}, got {outer}"
                );

                let broadcast_divisions = divisor_residue
                    .clone()
                    .divide(inner.clone())
                    .exact_checked()
                    .unwrap_or_else(|_| {
                        panic!(
                            "Switch broadcast axes in output time must come from input slice. \
                             Input Slice: {slice}, inner part of output Time: {inner}"
                        )
                    })
                    .division_terms;

                // Axes moving from Slice to Time must preserve their relative
                // order from input Slice.
                for window in broadcast_divisions.windows(2) {
                    assert!(
                        window[0].divisor_stride > window[1].divisor_stride,
                        "Switch axes moving from input Slice to output Time must preserve their relative order from input Slice. \
                         {} (outer in input Slice) appears inner to {} in output Time",
                        window[0].term,
                        window[1].term
                    );
                }

                // Find the span of the outermost non-directcast axis in the
                // input Slice.
                let mut max_non_dc = 0;
                for d in division.division_terms.iter() {
                    if d.dividend_stride != d.divisor_stride {
                        max_non_dc = max_non_dc.max(d.divisor_stride * d.resize);
                    }
                }
                let mut stride = 1usize;
                for factor in divisor_residue.factors().iter() {
                    match factor {
                        Factor::Term { resize, .. } => {
                            max_non_dc = max_non_dc.max(stride * *resize);
                            stride *= *resize;
                        }
                        Factor::Padding { size, .. } => {
                            stride = *size;
                        }
                    }
                }

                // Find the directcast axis with the smallest stride that is at
                // or above `max_non_dc`.
                let mut expected_ring_size = InSlice::SIZE;
                for d in division.division_terms.iter() {
                    if d.dividend_stride == d.divisor_stride && d.divisor_stride >= max_non_dc {
                        expected_ring_size = expected_ring_size.min(d.divisor_stride);
                    }
                }
                assert_eq!(
                    *ring_size, expected_ring_size,
                    "Switch ring size mismatch. Expected {expected_ring_size}, got {ring_size}"
                );
            }

            SwitchConfig::TransposedBroadcast1 { slice1, slice0 } => {
                assert!(
                    [slice1, slice0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );

                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );

                // m![InSlice / (slice1 * slice0)] = m![OutSlice / (tile * slice1)]
                // Since tile has the same size as slice0, this is: m![InSlice / (slice1 * slice0)] = m![OutSlice / (tile * slice1)]
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice0 * *slice1,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );

                // Broadcast axes in OutSlice should be new.
                if *slice0 > 1 {
                    let out_slice_broadcast_symbols = extract_symbols(&Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(OutSlice::to_value()),
                            stride: *slice1,
                        }),
                        modulo: *slice0,
                    });

                    let mut symbols = HashSet::new();
                    symbols.extend(extract_symbols(&InSlice::to_value()));
                    symbols.extend(extract_symbols(&InTime::to_value()));
                    assert_eq!(
                        out_slice_broadcast_symbols.intersection(&symbols).count(),
                        0,
                        "OutSlice broadcast axes must be new axes"
                    );
                }

                // slice1 is preserved at the innermost part of OutSlice
                if *slice1 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutSlice::to_value()),
                            modulo: *slice1,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(InSlice::to_value()),
                                stride: *slice0
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        "OutSlice must preserve slice1 from InSlice",
                    );
                }

                // OutTime must match [time0, slice0]
                let mut expected_out_time = Mapping::Identity.pair(InTime::to_value());
                if *slice0 > 1 {
                    // slice0 = m![InSlice % slice0]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(InSlice::to_value()),
                        modulo: *slice0,
                    });
                }

                assert_eq!(
                    expected_out_time.factorize(),
                    OutTime::to_value().factorize(),
                    "OutTime does not match expected layout: [time0, slice0]"
                );
            }
        }
    }
}

// ANCHOR: switch_impl
impl<'l, const T: Tu, P: CanApplySwitch, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Applies switching network routing only. The packet passes through
    /// unchanged, no padding, no reshaping. Use `collect` afterwards to
    /// normalize the packet to flit-sized chunks.
    #[primitive(TuTensor::switch)]
    pub fn switch<OutSlice: M, OutTime: M>(
        self,
        config: SwitchConfig,
    ) -> SwitchTensor<'l, T, D, Chip, Cluster, OutSlice, OutTime, Packet, B> {
        verify_switch::<Slice, Time, OutSlice, OutTime>(&config);
        SwitchTensor::new(self.ctx, self.inner.transpose(true))
    }
}
// ANCHOR_END: switch_impl

/// Validates switch engine constraints:
/// 1. Switch input and output slice sizes must match.
///
/// Delegates to [`SwitchConfig::verify`] for topology-specific checks.
fn verify_switch<InSlice: M, InTime: M, OutSlice: M, OutTime: M>(config: &SwitchConfig) {
    assert_eq!(
        InSlice::SIZE,
        OutSlice::SIZE,
        "Switch input and output slice sizes must match, got {} and {}",
        InSlice::SIZE,
        OutSlice::SIZE
    );
    config.verify::<InSlice, InTime, OutSlice, OutTime>();
}

/// Gathers all symbols present in a mapping.
fn extract_symbols(mapping: &Mapping) -> HashSet<Ident> {
    let mut symbols = HashSet::new();
    let mapping = mapping.factorize();
    let mut stack = mapping.clone().into_inner();
    while let Some(factor) = stack.pop() {
        if let Factor::Term { inner, .. } = factor {
            match &inner.inner {
                Atom::Symbol { symbol, .. } => {
                    symbols.insert(*symbol);
                }
                Atom::Composite(inner) => {
                    stack.extend(RBox::into_inner(inner.clone()).into_inner());
                }
            }
        }
    }
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;

    mod custom_broadcast {
        use super::*;

        axes![
            A = 16,
            B = 16,
            C = 8,
            D = 2,
            E = 2,
            P = 4,
            Q = 8,
            R = 8,
            S = 256,
            X = 4,
            Y = 2,
            Z = 2,
        ];

        mod permutation {
            use super::*;

            #[test]
            fn identity() {
                verify_switch::<m![S], m![C], m![S], m![C]>(&SwitchConfig::CustomBroadcast { ring_size: 1 });
            }

            #[test]
            fn full_permutation() {
                verify_switch::<m![A, B], m![C], m![B % 4, B / 4, A % 4, A / 4], m![C]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 256 },
                );
            }

            #[test]
            fn partial_permutation() {
                verify_switch::<m![A, B], m![C], m![A, B % 4, B / 4], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 16,
                });
            }

            #[test]
            fn three_axis_inner_swap() {
                verify_switch::<m![R, Q, P], m![C], m![R, P, Q], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 32,
                });
            }

            #[test]
            fn three_axis_outer_swap() {
                verify_switch::<m![R, Q, P], m![C], m![Q, R, P], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 256,
                });
            }

            #[test]
            fn padded_identity() {
                verify_switch::<m![P # 16, Q # 16], m![C], m![P # 16, Q # 16], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 1,
                });
            }

            #[test]
            fn padded_full_swap() {
                verify_switch::<m![R # 16, Q # 16], m![C], m![Q # 16, R # 16], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 256,
                });
            }

            #[test]
            fn padded_full_swap_different_padding() {
                verify_switch::<m![R # 16, Q # 16], m![C], m![Q # 32, R # 8], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 256,
                });
            }

            #[test]
            fn padded_partial_permutation() {
                verify_switch::<m![R # 16, Q # 16], m![C], m![R # 16, Q # 16 % 4, Q # 16 / 4], m![C]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 16 },
                );
            }

            #[test]
            #[should_panic(
                expected = "Switch axes moving from input slice to output time must be at the output time innermost positions."
            )]
            fn permutation_time_change() {
                verify_switch::<m![A, B], m![C], m![B, A], m![R]>(&SwitchConfig::CustomBroadcast { ring_size: 256 });
            }
        }

        mod broadcast {
            use super::*;

            #[test]
            fn broadcast() {
                verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 4,
                });
            }

            #[test]
            fn multi_axis_broadcast() {
                verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, A % 2, B % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            fn broadcast_with_permutation() {
                verify_switch::<m![A, B], m![C], m![A % 4, A / 4, B / 4, X], m![C, B % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 256 },
                );
            }

            #[test]
            fn broadcast_with_inner_permutation() {
                verify_switch::<m![R, Q, P], m![C], m![R, P / 2, Q, Y], m![C, P % 2]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 32,
                });
            }

            #[test]
            fn broadcast_innermost_axis() {
                verify_switch::<m![R, Q, P], m![C], m![R, Q, P / 2, Y], m![C, P % 2]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 2,
                });
            }

            #[test]
            fn non_contiguous_broadcast() {
                // Move R % 2 and P % 2 to Time (skipping Q).
                verify_switch::<m![R, Q, P], m![C], m![R / 2, Y, Q, P / 2, Z], m![C, R % 2, P % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 64 },
                );
            }

            #[test]
            fn full_broadcast() {
                verify_switch::<m![A, B], m![C], m![S], m![C, A, B]>(&SwitchConfig::CustomBroadcast { ring_size: 256 });
            }

            #[test]
            fn padded_outer_time() {
                verify_switch::<m![A, B], m![C # 32], m![A, B / 4, X], m![C # 32, B % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            fn padded_inner_axis_broadcast() {
                verify_switch::<m![P # 8, Q # 32], m![C], m![P # 8, Q # 32 / 4, X], m![C, Q # 32 % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            fn broadcast_with_padded_outer_axis() {
                verify_switch::<m![P # 32, Q], m![C], m![P # 32, Q / 4, X], m![C, Q % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            fn padded_both_axes_broadcast() {
                verify_switch::<m![P # 16, Q # 16], m![C], m![P # 16, Q # 16 / 4, X], m![C, Q # 16 % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            fn padded_time_broadcast() {
                verify_switch::<m![P # 8, Q # 32], m![C # 16], m![P # 8, Q # 32 / 4, X], m![C # 16, Q # 32 % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch broadcast axes must each be used exactly once in OutSlice")]
            fn duplicate_broadcast_symbol() {
                verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Y], m![C, A % 2, B % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch broadcast axes must be new axes (not present in input Slice or Time).")]
            fn broadcast_axis_from_time() {
                verify_switch::<m![A, B], m![C], m![A, B / 8, C], m![C, B % 8]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 8,
                });
            }

            #[test]
            #[should_panic(expected = "Switch broadcast axes must be new axes (not present in input Slice or Time).")]
            fn inter_transpose() {
                verify_switch::<m![A, B], m![C], m![A, C, B / 8], m![B % 8]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 256,
                });
            }

            #[test]
            fn partial_broadcast_replacement() {
                // A % 2 replaced by broadcast
                verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, B % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch broadcast axis X in output Slice must not be padded.")]
            fn padded_broadcast_axis() {
                verify_switch::<m![A, B], m![C], m![A / 2, X # 8, B / 8, Y], m![C, A % 2, B % 8]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            #[should_panic(
                expected = "Switch axes moving from input slice to output time must be at the output time innermost positions."
            )]
            fn moved_axes_not_innermost() {
                verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![A % 2, C, B % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            #[should_panic(
                expected = "Switch axes moving from input Slice to output Time must preserve their relative order from input Slice."
            )]
            fn reversed_order_in_time() {
                // A % 2 and B % 2 are reversed in output Time.
                verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, B % 2, A % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            #[should_panic(
                expected = "Switch axes moving from input slice to output time must be at the output time innermost positions."
            )]
            fn outer_time_padding_mismatch() {
                verify_switch::<m![A, B], m![C # 32], m![A, B / 4, X], m![C # 16, B % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            fn broadcast_replace_in_place() {
                verify_switch::<m![R, P], m![C], m![R, X], m![C]>(&SwitchConfig::CustomBroadcast { ring_size: 4 });
            }

            #[test]
            fn broadcast_with_moved_axis() {
                axes![A = 2, B = 2, C = 2, D = 2, E = 2, X = 2];
                verify_switch::<m![A, B, C, D, E], m![1], m![E, B, X, A, D], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 32,
                })
            }
        }

        mod slicing {
            use super::*;

            #[test]
            fn slicing() {
                verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4 = 3]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 4,
                });
            }

            #[test]
            fn slicing_with_broadcast() {
                verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 4, X], m![C, A % 2, B % 4 = 3]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
            }

            #[test]
            fn single_axis_slicing() {
                verify_switch::<m![S], m![C], m![S / 4, X], m![C, S % 4 = 3]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 4,
                });
            }

            #[test]
            fn padded_broadcast_slicing() {
                verify_switch::<m![P # 8, Q # 32], m![C], m![P # 8, Q # 32 / 4, X], m![C, Q # 32 % 4 = 3]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch broadcast axes in output time must come from input slice.")]
            fn wrong_axis_in_slicing() {
                verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B / 4 = 3]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 4,
                });
            }
        }

        mod ring_size {
            use super::*;

            #[test]
            #[should_panic(expected = "Switch ring size must be a power of 2, got 3")]
            fn non_power_of_two() {
                verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 3,
                });
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 256, got 4")]
            fn wrong_full_permutation() {
                verify_switch::<m![A, B], m![C], m![B % 4, B / 4, A % 4, A / 4], m![C]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 16, got 256")]
            fn wrong_partial_permutation() {
                verify_switch::<m![A, B], m![C], m![A, B % 4, B / 4], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 256,
                });
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 4, got 32")]
            fn wrong_broadcast() {
                verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 32,
                });
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 256, got 2")]
            fn wrong_permutation_above_broadcast() {
                verify_switch::<m![R, Q, P], m![C], m![Q, R, P / 2, Y], m![C, P % 2]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 2,
                });
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 4, got 16")]
            fn wrong_padded_broadcast() {
                verify_switch::<m![A # 16, B # 16], m![C], m![A # 16, B # 16 / 4, X], m![C, B # 16 % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 16 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 16, got 256")]
            fn wrong_padded_permutation() {
                verify_switch::<m![A # 16, B # 16], m![C], m![A # 16, B # 16 % 4, B # 16 / 4], m![C]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 256 },
                );
            }

            #[test]
            #[should_panic(expected = "Switch ring size mismatch. Expected 256, got 4")]
            fn wrong_padded_permutation_above_broadcast() {
                verify_switch::<m![P # 8, Q # 32], m![C], m![Q # 32 / 4, P # 8, X], m![C, Q # 32 % 4]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
            }
        }
    }
}
