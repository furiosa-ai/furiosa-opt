//! Lane Folder (`contract_lane`): folds `Lane` into the output stream,
//! producing the contraction pipeline's final [`ContractTensor`].

use std::collections::HashMap;

use abi_stable::std_types::Tuple2;
use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::align_up;
use crate::engine::contraction::{
    CONTRACT_LANE_OUT_PACKET_ELEMENTS, ContractTensor, ContractTimeTensor, TEMPORAL_ACCUMULATOR_COLS,
};
use crate::runtime::Backend;
use crate::scalar::*;

/// Contraction mode for the Lane Folder.
#[primitive(LaneMode)]
#[derive(Clone, Debug)]
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

// ANCHOR: contract_lane_def
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
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
        verify_contract_lane::<Lane, Time, Packet, OutTime, OutPacket>(self.pre_reduce_time, mode);
        ContractTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: contract_lane_def

/// Validates the Lane Folder.
///
/// Checks:
/// 1. `Packet::SIZE` must be at most 32 (`i32`/`f32`).
/// 2. `OutPacket::SIZE` must be 8 (`i32`/`f32`).
/// 3. Output factor composition matches retained (non-reduced) axes:
///    - Interleaved: `OutTime = [Time, Packet (may be sliced)]`, `OutPacket = [Lane # 8]`.
///    - Sequential: `OutTime = [Time, Lane, packet_outer]`, `OutPacket = [packet_inner # 8]`.
/// 4. Accumulator fits hardware limit (1024 elements):
///    - Interleaved: `inner_time * ReducedPacket <= 128` (since `align_up(Lane, 8) = 8`).
///    - Sequential: `inner_time * Lane * packet_outer <= 32` (since `align_up(ReducedPacket, 32) = 32`).
pub(crate) fn verify_contract_lane<Lane: M, Time: M, Packet: M, OutTime: M, OutPacket: M>(
    pre_reduce_time: FMapping,
    kind: LaneMode,
) {
    assert!(
        Packet::SIZE <= TEMPORAL_ACCUMULATOR_COLS,
        "contract_lane: Packet::SIZE must be at most {TEMPORAL_ACCUMULATOR_COLS}, got {}",
        Packet::SIZE
    );
    assert_eq!(
        OutPacket::SIZE,
        CONTRACT_LANE_OUT_PACKET_ELEMENTS,
        "contract_lane: OutPacket::SIZE must be {CONTRACT_LANE_OUT_PACKET_ELEMENTS}, got {}",
        OutPacket::SIZE
    );

    let time = Time::to_value().factorize();
    let packet = Packet::to_value().factorize().remove_padding();
    let out_time = OutTime::to_value().factorize();
    let out_packet = OutPacket::to_value().factorize();

    // Determine `outer_time` and `packet_outer_size` based on contraction kind.
    // `packet_outer_size` is 1 for Interleaved (no packet split into OutTime),
    // and the size of the outer packet portion for Sequential.
    let (outer_time, packet_outer_size) = match kind {
        LaneMode::Interleaved => {
            // `OutPacket = [Lane # 8]`
            let expected_out_packet = Lane::to_value().factorize().pad(CONTRACT_LANE_OUT_PACKET_ELEMENTS);
            assert_eq!(
                out_packet, expected_out_packet,
                "contract_lane ({kind}): OutPacket mismatch. Expected: {expected_out_packet}, got: {out_packet}"
            );

            // `OutTime = [Time, Packet (may be sliced)]`
            // Search for the `Packet / Time` boundary.
            let outer_time = (1..=out_time.size().min(packet.size()).min(TEMPORAL_ACCUMULATOR_COLS))
                .filter(|&split| {
                    out_time.size() % split == 0
                        // If `packet` is not the identity mapping, require it
                        // to be present in `OutTime`.
                        && (split > 1 || packet.size() == 1)
                        // `Time::SIZE` <= pre-reduce `Time::SIZE`.
                        && out_time.size() / split <= time.size()
                })
                .find_map(|split| {
                    let Tuple2(outer_time, sliced_packet) = out_time.split_at(split);

                    // Slicing may only remove padding.
                    if sliced_packet != packet {
                        return None;
                    }

                    Some(outer_time)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "contract_lane ({kind}): OutTime mismatch. \
                         Could not decompose OutTime {out_time} into [Time, Packet (truncated)] \
                         where Time is {time} and Packet is a truncation of {packet}"
                    )
                });
            (outer_time, 1)
        }
        LaneMode::Sequential => {
            // `OutTime   = [Time, Lane, packet_outer]`
            // `OutPacket = [packet_inner # CONTRACT_LANE_OUT_PACKET_ELEMENTS]`
            //
            // `packet` is padded to the next multiple of CONTRACT_LANE_OUT_PACKET_ELEMENTS,
            // then split at CONTRACT_LANE_OUT_PACKET_ELEMENTS:
            // - `packet_inner` (CONTRACT_LANE_OUT_PACKET_ELEMENTS elements): becomes `OutPacket`
            // - `packet_outer`                                             : absorbed into `OutTime`
            let padded = packet
                .clone()
                .pad(align_up(packet.size(), CONTRACT_LANE_OUT_PACKET_ELEMENTS));
            let Tuple2(packet_outer, packet_inner) = padded.split_at(CONTRACT_LANE_OUT_PACKET_ELEMENTS);
            let packet_outer_size = packet_outer.size();

            // `OutPacket = [packet_inner # CONTRACT_LANE_OUT_PACKET_ELEMENTS]`.
            assert_eq!(
                packet_inner, out_packet,
                "contract_lane ({kind}): OutPacket mismatch. Expected: {packet_inner}, got: {out_packet}"
            );

            // `OutTime` ends with `[Lane, packet_outer]`
            let lane_packet = Lane::to_value().factorize().mul(packet_outer);
            let Tuple2(outer_time, inner_time) = out_time.split_at(lane_packet.size());
            assert_eq!(
                inner_time, lane_packet,
                "contract_lane ({kind}): OutTime mismatch. Expected {lane_packet}, got {inner_time}"
            );

            (outer_time, packet_outer_size)
        }
    };

    // `contract_lane` performs no reduction, so the post-`split_at` outer portion of
    // `OutTime` must equal `Time` exactly (order and padding both already enforced by
    // `contract_time`).
    assert_eq!(
        outer_time, time,
        "contract_lane ({kind}): OutTime mismatch. Outer portion of OutTime must equal Time: \
         expected {time}, got {outer_time}"
    );

    // Recover the cross-stage `inner_time` (axes inner to the outermost reduce performed
    // by `contract_time`) by dividing `pre_reduce_time` (captured at the `contract_time`
    // call site) by the post-reduce `time`.
    let division = pre_reduce_time
        .clone()
        .divide(time.clone())
        .exact_checked()
        .unwrap_or_else(|_| {
            panic!("contract_lane ({kind}): inconsistent pre/post reduce Time: {pre_reduce_time}, {time}")
        });
    let division_terms = &division.division_terms;

    // Build `cumulative_stride : padding_per_stride` table for each term in `pre_reduce_time`.
    // `m![A # 8, B # 4]` with `axes![A = 4, B = 2]` -> { 1: 8, 8: 4 }
    let mut time_padding_per_stride: HashMap<usize, usize> = HashMap::new();
    let factors = pre_reduce_time.factors();
    let mut stride = 1;
    for (i, factor) in factors.iter().enumerate() {
        match factor {
            Factor::Term { resize, .. } => {
                time_padding_per_stride.insert(
                    stride,
                    if let Some(Factor::Padding { size, .. }) = factors.get(i + 1) {
                        size / stride
                    } else {
                        *resize
                    },
                );
                stride *= resize;
            }
            Factor::Padding { size, .. } => {
                stride = *size;
            }
        }
    }

    // Calculate axis size inner to outermost reduce.
    let padding_end = |d: &DivisionTerm| {
        d.dividend_stride
            * time_padding_per_stride
                .get(&d.dividend_stride)
                .copied()
                .unwrap_or(d.resize)
    };
    let inner_time = if division_terms.is_empty() {
        // Case 1: All axes reduced.
        1
    } else if padding_end(&division_terms[0]) < pre_reduce_time.size() {
        // Case 2: The outermost axis was reduced.
        time.size()
    } else {
        // Case 3: The outermost retained factor reaches the top of `pre_reduce_time`.
        // Walk outer-to-inner looking for the first gap between adjacent division terms.
        division_terms
            .windows(2)
            .find(|w| padding_end(&w[1]) != w[0].dividend_stride)
            .map_or(1, |w| w[0].divisor_stride)
    };

    // Check buffer limits
    match kind {
        LaneMode::Interleaved => {
            let buffer = inner_time * packet.size();
            assert!(
                buffer <= 1024 / CONTRACT_LANE_OUT_PACKET_ELEMENTS,
                "contract_lane ({}): axes inner to reduce must be <= {} in size, got {}",
                kind,
                1024 / CONTRACT_LANE_OUT_PACKET_ELEMENTS,
                buffer
            );
        }
        LaneMode::Sequential => {
            let buffer = inner_time * Lane::SIZE * packet_outer_size;
            assert!(
                buffer <= 1024 / TEMPORAL_ACCUMULATOR_COLS,
                "contract_lane ({}): axes inner to reduce must be <= {} in size, got {}",
                kind,
                1024 / TEMPORAL_ACCUMULATOR_COLS,
                buffer
            );
        }
    }
}

/// Test helper: simulates the `contract_time` → `contract_lane` chain by
/// capturing `Time::to_value().factorize()` as `pre_reduce_time` and forwarding
/// to `verify_contract_lane`.
#[cfg(test)]
fn vcl<Lane: M, Time: M, Packet: M, OutTime: M, OutPacket: M>(kind: LaneMode) {
    verify_contract_lane::<Lane, Time, Packet, OutTime, OutPacket>(Time::to_value().factorize(), kind);
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    mod out_packet_size {
        use super::*;

        #[test]
        fn valid() {
            vcl::<m![1], m![A], m![1], m![A], m![1 # 8]>(LaneMode::Interleaved);
        }
    }

    mod interleaved {
        use super::*;

        // For tests that performed reduction in the old monolithic `verify_accumulate`,
        // the new test passes `Time` = post-`contract_time` time (i.e., the retained
        // axes), and uses `vcl_pre` if buffer-bound is being tested with reduction.

        #[test]
        fn valid() {
            // Old: Time=[A,B], OutTime=[B] (reduced A). New post-reduce time=[B].
            vcl::<m![1], m![B], m![1], m![B], m![1 # 8]>(LaneMode::Interleaved);
        }

        #[test]
        fn valid_padding() {
            // Old: Time=[A#8,B#4], OutTime=[B#4] (reduced A#8). post=[B#4].
            vcl::<m![1], m![B # 4], m![1], m![B # 4], m![1 # 8]>(LaneMode::Interleaved);
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            // No reduction: post=pre.
            vcl::<m![1], m![A # 8, B], m![D], m![A # 8, B, D], m![1 # 8]>(LaneMode::Interleaved);
        }

        #[test]
        fn valid_non_outermost() {
            // Old: Time=[C,A,B], OutTime=[C,B] (reduced A). post=[C,B].
            vcl::<m![N], m![C, B], m![1], m![C, B], m![N]>(LaneMode::Interleaved);
        }

        #[test]
        fn valid_four_rows() {
            vcl::<m![M], m![C, B], m![1], m![C, B], m![M # 8]>(LaneMode::Interleaved);
        }

        #[test]
        fn valid_all_time_reduced() {
            // Old: Time=[A], OutTime=[1] (all reduced). post=[1].
            vcl::<m![N], m![1], m![1], m![1], m![N]>(LaneMode::Interleaved);
        }
    }

    mod sequential {
        use super::*;

        #[test]
        fn valid() {
            // Old: Time=[A,B], OutTime=[B,N]. Reduced A. post=[B].
            vcl::<m![N], m![B], m![1], m![B, N], m![1 # 8]>(LaneMode::Sequential);
        }

        #[test]
        fn valid_padded_row() {
            vcl::<m![N], m![B], m![1], m![B, N # 8], m![1 # 8]>(LaneMode::Sequential);
        }

        #[test]
        fn valid_all_time_reduced() {
            vcl::<m![N], m![1], m![1], m![N], m![1 # 8]>(LaneMode::Sequential);
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            vcl::<m![N], m![A # 8, B], m![1], m![A # 8, B, N], m![1 # 8]>(LaneMode::Sequential);
        }

        #[test]
        fn valid_padded_packet() {
            vcl::<m![N], m![M], m![B], m![M, N], m![B # 8]>(LaneMode::Sequential);
        }

        #[test]
        fn valid_full_temporal_reduction() {
            // Old: Time=[M], OutTime=[N,D/8] - all M reduced, post=[1].
            vcl::<m![N], m![1], m![D], m![N, D / 8], m![D % 8]>(LaneMode::Sequential);
        }

        #[test]
        fn valid_multi_axis_reduction() {
            // Reduce A and C. post=[B].
            vcl::<m![N], m![B], m![1], m![B, N], m![1 # 8]>(LaneMode::Sequential);
        }
    }
}
