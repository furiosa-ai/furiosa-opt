//! Lane Folder (`contract_lane`): folds `Lane` into the output stream,
//! producing the contraction pipeline's final [`ContractTensor`].

use furiosa_mapping::*;
use furiosa_opt_lower::{ContractLaneInput, LaneMode as LowerLaneMode, config_contract_lane};
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::ContractionAccumulator;
use crate::context::*;
use crate::engine::contraction::{ContractTensor, ContractTimeTensor};
use crate::tensor::Tensor;

/// Contraction mode for the Lane Folder.
#[primitive(LaneMode)]
#[derive(Clone, Copy, Debug)]
pub enum LaneMode {
    /// Interleaved: outputs data element-by-element across all `Lane`s.
    Interleaved,
    /// Sequential: outputs reduced data in each `Lane` sequentially.
    Sequential,
}

impl std::fmt::Display for LaneMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaneMode::Interleaved => write!(f, "Interleaved"),
            LaneMode::Sequential => write!(f, "Sequential"),
        }
    }
}

impl From<LaneMode> for LowerLaneMode {
    fn from(mode: LaneMode) -> Self {
        match mode {
            LaneMode::Interleaved => Self::Interleaved,
            LaneMode::Sequential => Self::Sequential,
        }
    }
}

// ANCHOR: contract_lane_def
impl<'l, const T: Tu, D: ContractionAccumulator, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Folds the `Lane` dimension into the output stream.
    /// `LaneMode::Interleaved` relocates `Lane` into `OutPacket`;
    /// `LaneMode::Sequential` relocates `Lane` into `OutTime`.
    #[primitive(ContractTimeTensor::contract_lane)]
    pub fn contract_lane<OutTime: M, OutPacket: M>(
        self,
        mode: LaneMode,
    ) -> ContractTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_contract_lane(ContractLaneInput {
            in_lane: Lane::to_value(),
            in_time: Time::to_value(),
            in_packet: Packet::to_value(),
            out_time: OutTime::to_value(),
            out_packet: OutPacket::to_value(),
            pre_reduce_time: self.pre_reduce_time,
            mode: mode.into(),
        });
        // Earlier stages only retype the deferred operands. Perform their fused contraction here,
        // then apply the lane-fold relayout.
        // `contraction_prewidened` validates `out` through mapping carve. Symbol-wise comparison is
        // invalid because one contracted symbol may span spatial and reduced slots.
        let contraction = self.inner;
        let out = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }]>::to_value();
        let reduced: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }], B> =
            Tensor::from_inner(B::contraction_prewidened(
                &contraction.lhs,
                &contraction.rhs,
                &contraction.lhs_map,
                &contraction.rhs_map,
                &contraction.pre_reduce,
                &out,
            ));
        ContractTensor::new(self.device, reduced.transpose(false))
    }
}
// ANCHOR_END: contract_lane_def

/// Validates the Lane Folder via [`furiosa_opt_lower::config_contract_lane`] (packet / time /
/// accumulator rules documented there).
pub(crate) fn verify_contract_lane(input: ContractLaneInput) {
    config_contract_lane(input).unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_lower::ContractLaneError;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    mod out_packet_size {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![1]>::to_value(),
                in_time: <m![A]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![A]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![A]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }
        #[test]
        fn invalid() {
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![1]>::to_value(),
                    in_time: <m![A]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![A]>::to_value(),
                    out_packet: <m![D]>::to_value(),
                    pre_reduce_time: <m![A]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::OutPacketSize(32))
            ));
        }
    }

    mod interleaved {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![1]>::to_value(),
                in_time: <m![B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![B]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![B]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }

        #[test]
        fn valid_padding() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![1]>::to_value(),
                in_time: <m![B # 4]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![B # 4]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![B # 4]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![1]>::to_value(),
                in_time: <m![A # 8, B]>::to_value(),
                in_packet: <m![D]>::to_value(),
                out_time: <m![A # 8, B, D]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![A # 8, B]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }

        #[test]
        fn valid_non_outermost() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![C, B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![C, B]>::to_value(),
                out_packet: <m![N]>::to_value(),
                pre_reduce_time: <m![C, B]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }

        #[test]
        fn valid_four_rows() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![M]>::to_value(),
                in_time: <m![C, B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![C, B]>::to_value(),
                out_packet: <m![M # 8]>::to_value(),
                pre_reduce_time: <m![C, B]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }

        #[test]
        fn valid_all_time_reduced() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![1]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![1]>::to_value(),
                out_packet: <m![N]>::to_value(),
                pre_reduce_time: <m![1]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }
        #[test]
        fn invalid_sliced_no_padding() {
            // Packet truncation is only allowed on padded elements.
            // Otherwise, input data would be silently discarded.
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![M]>::to_value(),
                    in_packet: <m![D]>::to_value(),
                    out_time: <m![M, D = 16]>::to_value(),
                    out_packet: <m![N]>::to_value(),
                    pre_reduce_time: <m![M]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::OutTimeMismatch { .. })
            ));
        }

        #[test]
        fn invalid_packet_size_out_time() {
            // OutTime=[M,K]; split_at(Packet=32) yields inner K%32 != Packet D, so the fold mismatches.
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![M]>::to_value(),
                    in_packet: <m![D]>::to_value(),
                    out_time: <m![M, K]>::to_value(),
                    out_packet: <m![N]>::to_value(),
                    pre_reduce_time: <m![M]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::OutTimeMismatch { .. })
            ));
        }

        #[test]
        fn invalid_resize() {
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![1]>::to_value(),
                    in_time: <m![A]>::to_value(),
                    in_packet: <m![D]>::to_value(),
                    out_time: <m![A, D / 4 % 4]>::to_value(),
                    out_packet: <m![1 # 8]>::to_value(),
                    pre_reduce_time: <m![A]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::OutTimeMismatch { .. })
            ));
        }

        #[test]
        fn invalid_out_time() {
            // OutTime decomposes as outer_time=[C], packet=identity. outer_time==time fails.
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![A, B]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![C]>::to_value(),
                    out_packet: <m![N]>::to_value(),
                    pre_reduce_time: <m![A, B]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::OuterTimeMismatch { .. })
            ));
        }

        #[test]
        fn invalid_buffer() {
            // pre=[A,D#64,C], post=[D#64,C] (A reduced). inner_time = D#64*C/1 = 256.
            assert_eq!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![D # 64, C]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![D # 64, C]>::to_value(),
                    out_packet: <m![N]>::to_value(),
                    pre_reduce_time: <m![A, D # 64, C]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::BufferExceeded {
                    mode: LowerLaneMode::Interleaved,
                    padded_lane: 8,
                    inner_time: 256,
                    padded_packet: 1,
                    limit: 1024,
                })
            );
        }

        #[test]
        fn invalid_buffer_multiple_reduce_axes() {
            // pre=[A,B#64,M#8,C], post=[B#64,C]. Reduce A and M#8.
            assert_eq!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![B # 64, C]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![B # 64, C]>::to_value(),
                    out_packet: <m![N]>::to_value(),
                    pre_reduce_time: <m![A, B # 64, M # 8, C]>::to_value(),
                    mode: LowerLaneMode::Interleaved,
                }),
                Err(ContractLaneError::BufferExceeded {
                    mode: LowerLaneMode::Interleaved,
                    padded_lane: 8,
                    inner_time: 256,
                    padded_packet: 1,
                    limit: 1024,
                })
            );
        }

        /// Reducing an axis between two digits of a padded axis must keep the padded extent intact.
        #[test]
        fn valid_inner_time_across_a_split_padded_axis() {
            axes![L1 = 192, Bat = 4096, W = 12];

            verify_contract_lane(ContractLaneInput {
                in_lane: <m![L1 # 256 % 8]>::to_value(),
                in_time: <m![L1 # 256 / 32 % 4, Bat / 32 % 2, Bat / 64 % 2, L1 # 256 / 8 % 4]>::to_value(),
                in_packet: <m![Bat % 32]>::to_value(),
                out_time: <m![L1 # 256 / 32 % 4, Bat / 32 % 2, Bat / 64 % 2, L1 # 256 / 8 % 4, Bat % 32]>::to_value(),
                out_packet: <m![L1 # 256 % 8]>::to_value(),
                pre_reduce_time:
                    <m![L1 # 256 / 32 % 4, Bat / 32 % 2, Bat / 64 % 2, W % 6, W / 6, L1 # 256 / 8 % 4]>::to_value(),
                mode: LowerLaneMode::Interleaved,
            });
        }
    }

    mod sequential {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![B, N]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![B]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }

        #[test]
        fn valid_padded_row() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![B, N # 8]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![B]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }

        #[test]
        fn valid_all_time_reduced() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![1]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![N]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![1]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![A # 8, B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![A # 8, B, N]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![A # 8, B]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }

        #[test]
        fn valid_padded_packet() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![M]>::to_value(),
                in_packet: <m![B]>::to_value(),
                out_time: <m![M, N]>::to_value(),
                out_packet: <m![B # 8]>::to_value(),
                pre_reduce_time: <m![M]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }

        #[test]
        fn valid_full_temporal_reduction() {
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![1]>::to_value(),
                in_packet: <m![D]>::to_value(),
                out_time: <m![N, D / 8]>::to_value(),
                out_packet: <m![D % 8]>::to_value(),
                pre_reduce_time: <m![1]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }
        #[test]
        fn invalid_packet_axis() {
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![M]>::to_value(),
                    in_packet: <m![B]>::to_value(),
                    out_time: <m![M, N]>::to_value(),
                    out_packet: <m![A # 8]>::to_value(),
                    pre_reduce_time: <m![M]>::to_value(),
                    mode: LowerLaneMode::Sequential,
                }),
                Err(ContractLaneError::OutPacketMismatch { .. })
            ));
        }

        #[test]
        fn invalid_out_time() {
            // post=[B] (A reduced), OutTime=[C,N], split_at|Lane*packet_outer|=|N|=8: inner=N, outer=[C].
            // outer_time==time fails ([C] != [B]).
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![B]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![C, N]>::to_value(),
                    out_packet: <m![1 # 8]>::to_value(),
                    pre_reduce_time: <m![B]>::to_value(),
                    mode: LowerLaneMode::Sequential,
                }),
                Err(ContractLaneError::OuterTimeMismatch { .. })
            ));
        }

        #[test]
        fn invalid_out_time_row() {
            // Lane=N, OutTime ends with [Lane, packet_outer]=[N]; OutTime=[B,M] ends in M not N.
            assert!(matches!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![B]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![B, M]>::to_value(),
                    out_packet: <m![1 # 8]>::to_value(),
                    pre_reduce_time: <m![B]>::to_value(),
                    mode: LowerLaneMode::Sequential,
                }),
                Err(ContractLaneError::OutTimeMismatch { .. })
            ));
        }

        #[test]
        fn valid_multi_axis_reduction() {
            // Reduce A and C. post=[B].
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![B]>::to_value(),
                in_packet: <m![1]>::to_value(),
                out_time: <m![B, N]>::to_value(),
                out_packet: <m![1 # 8]>::to_value(),
                pre_reduce_time: <m![A, B, C]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }
        #[test]
        fn invalid_buffer() {
            // pre=[A,P], post=[P] (A=4 reduced). inner_time = P = 8. Sequential pads Packet=1 to 32,
            // so buffer = Lane 8 * 8 * 32 = 2048 > 1024.
            assert_eq!(
                config_contract_lane(ContractLaneInput {
                    in_lane: <m![N]>::to_value(),
                    in_time: <m![P]>::to_value(),
                    in_packet: <m![1]>::to_value(),
                    out_time: <m![P, N]>::to_value(),
                    out_packet: <m![1 # 8]>::to_value(),
                    pre_reduce_time: <m![A, P]>::to_value(),
                    mode: LowerLaneMode::Sequential,
                }),
                Err(ContractLaneError::BufferExceeded {
                    mode: LowerLaneMode::Sequential,
                    padded_lane: 8,
                    inner_time: 8,
                    padded_packet: 32,
                    limit: 1024,
                })
            );
        }

        #[test]
        fn valid_wide_packet_reduced() {
            // pre=[A,N,B], post=[N,B] (A reduced). inner_time = N*B = 16. Lane = 1, Packet pads to 32,
            // so buffer = 1 * 16 * 32 = 512 <= 1024.
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![1]>::to_value(),
                in_time: <m![N, B]>::to_value(),
                in_packet: <m![D]>::to_value(),
                out_time: <m![N, B, D / 8]>::to_value(),
                out_packet: <m![D % 8]>::to_value(),
                pre_reduce_time: <m![A, N, B]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }

        #[test]
        fn valid_wide_packet_at_capacity() {
            // pre=[A,M], post=[M] (A reduced). inner_time = M = 4. Lane = 8, Packet pads to 32,
            // so buffer = 8 * 4 * 32 = 1024 = accumulator capacity.
            verify_contract_lane(ContractLaneInput {
                in_lane: <m![N]>::to_value(),
                in_time: <m![M]>::to_value(),
                in_packet: <m![D]>::to_value(),
                out_time: <m![M, N, D / 8]>::to_value(),
                out_packet: <m![D % 8]>::to_value(),
                pre_reduce_time: <m![A, M]>::to_value(),
                mode: LowerLaneMode::Sequential,
            });
        }
    }
}
