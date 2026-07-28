//! Switch Engine: slice/time topology rearrangements.
//!
//! Applies a switching-network routing to a `FetchTensor`. The packet passes
//! through unchanged — use [`super::collect`] afterwards to normalize the packet
//! to flit-sized chunks.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_lower::config_switch;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplySwitch;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// After the switch engine.
#[derive(Debug)]
pub struct PositionSwitch;

impl Position for PositionSwitch {}

/// Tensor streamed after the switch engine.
pub type SwitchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionSwitch, D, Chip, Cluster, Slice, Time, Packet, B>;

/// Configuration and typed failure reasons for the `switch` operation. Defined in
/// `furiosa-opt-lower-types` and validated by the lowering `config_switch`, so the FMapping
/// divide-algebra the `CustomBroadcast` case needs stays off the public engine surface.
pub use furiosa_opt_lower::{SwitchConfig, SwitchError};

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    SwitchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_aligned_by_access_width::<D, Packet>();
    }

    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
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

/// Validates switch engine constraints (including the slice-size match) via the lowering
/// `config_switch`, which runs the topology-specific checks and the `CustomBroadcast` FMapping
/// divide-algebra inside the impl. The resolved config is discarded; only success/failure matters.
fn verify_switch<InSlice: M, InTime: M, OutSlice: M, OutTime: M>(config: &SwitchConfig) {
    config_switch(
        config,
        &InSlice::to_value(),
        &InTime::to_value(),
        &OutSlice::to_value(),
        &OutTime::to_value(),
    )
    .unwrap_or_else(|err| panic!("{err}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `SwitchError` a rejected switch produces. Rejection tests pin the exact variant
    /// (not the rendered message), so a reworded error can't silently change why it rejects.
    fn switch_error<InSlice: M, InTime: M, OutSlice: M, OutTime: M>(config: &SwitchConfig) -> SwitchError {
        config_switch(
            config,
            &InSlice::to_value(),
            &InTime::to_value(),
            &OutSlice::to_value(),
            &OutTime::to_value(),
        )
        .unwrap_err()
    }

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
                    ring_size: 128,
                });
            }

            #[test]
            fn padded_full_swap_different_padding() {
                verify_switch::<m![R # 16, Q # 16], m![C], m![Q # 32, R # 8], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 128,
                });
            }

            #[test]
            fn padded_partial_permutation() {
                verify_switch::<m![R # 16, Q # 16], m![C], m![R # 16, Q # 16 % 4, Q # 16 / 4], m![C]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 16 },
                );
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
            fn partial_broadcast_replacement() {
                // A % 2 replaced by broadcast
                let err = switch_error::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, B % 2]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
                assert!(matches!(err, SwitchError::Unsequenceable(..)), "{err}");
            }

            #[test]
            fn broadcast_replace_in_place() {
                let err =
                    switch_error::<m![R, P], m![C], m![R, X], m![C]>(&SwitchConfig::CustomBroadcast { ring_size: 4 });
                assert!(matches!(err, SwitchError::Unsequenceable(..)), "{err}");
            }

            #[test]
            fn broadcast_with_moved_axis() {
                axes![A = 2, B = 2, C = 2, D = 2, E = 2, X = 2];
                verify_switch::<m![A, B, C, D, E], m![1], m![E, B, X, A, D], m![C]>(&SwitchConfig::CustomBroadcast {
                    ring_size: 32,
                })
            }
        }

        /// Slicing (an OutTime sub-axis that reads only part of its source, e.g. `B % 4 = 3` of 4)
        /// is intentionally NOT supported by the sequencer-based verify: a partial read leaves the
        /// dropped live input lanes unconsumed, so the Read does not sequence. These pin that
        /// rejection; each surfaces as `Unsequenceable`.
        mod slicing_unsupported {
            use super::*;

            #[test]
            fn slice_moved_axis() {
                let err = switch_error::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4 = 3]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
                assert!(matches!(err, SwitchError::Unsequenceable(..)), "{err}");
            }

            #[test]
            fn slice_with_broadcast() {
                let err = switch_error::<m![A, B], m![C], m![A / 2, Y, B / 4, X], m![C, A % 2, B % 4 = 3]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 32 },
                );
                assert!(matches!(err, SwitchError::Unsequenceable(..)), "{err}");
            }

            #[test]
            fn slice_single_axis() {
                let err =
                    switch_error::<m![S], m![C], m![S / 4, X], m![C, S % 4 = 3]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 4,
                    });
                assert!(matches!(err, SwitchError::Unsequenceable(..)), "{err}");
            }

            #[test]
            fn slice_padded_broadcast() {
                let err = switch_error::<m![P # 8, Q # 32], m![C], m![P # 8, Q # 32 / 4, X], m![C, Q # 32 % 4 = 3]>(
                    &SwitchConfig::CustomBroadcast { ring_size: 4 },
                );
                assert!(matches!(err, SwitchError::Unsequenceable(..)), "{err}");
            }
        }
    }
}
