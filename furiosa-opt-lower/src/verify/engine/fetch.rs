//! Fetch engine: a base per chip, cluster or slice, which lifts a DM axis onto that dimension.

use std::fmt::{Display, Formatter};
use std::iter::zip;

use furiosa_mapping::{Mapping, MappingExt, into_slots_of_size_two};

use crate::{FETCH_VALID_CLUSTER_SIZES, FETCH_VALID_SLICE_SIZES, FetchError};

/// The placement dimension receiving a lifted fetch axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchLiftDimension {
    Chip,
    Cluster,
    Slice,
}

/// Why a placement change cannot provide axes for a Fetch lift.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FetchLiftError {
    #[error("{dimension} lift changes no placement axis; use a plain fetch")]
    NoAxisLifted { dimension: FetchLiftDimension },
    #[error("{dimension} lift placement size must be a power of two, got {size}: {placement}")]
    UnsupportedPlacementSize {
        dimension: FetchLiftDimension,
        size: usize,
        placement: Mapping,
    },
    #[error(
        "{dimension} lift placement size differs: input {in_size}, output {out_size}: \
         DM {in_placement}, stream {out_placement}"
    )]
    PlacementSizeMismatch {
        dimension: FetchLiftDimension,
        in_size: usize,
        out_size: usize,
        in_placement: Mapping,
        out_placement: Mapping,
    },
    #[error(
        "{dimension} lift placement differs at stride {stride}: input {in_slot}, output \
         {out_slot}; only an input size-2 broadcast may differ"
    )]
    PlacementMismatch {
        dimension: FetchLiftDimension,
        stride: usize,
        in_slot: Mapping,
        out_slot: Mapping,
    },
    #[error(
        "{dimension} lift replaces the input broadcast at stride {stride} with padding {padding}; \
         reshape the DM placement and leave the corresponding input padding unread instead"
    )]
    PaddingLift {
        dimension: FetchLiftDimension,
        stride: usize,
        padding: Mapping,
    },
}

/// Placement mappings used to configure one Fetch lift.
pub struct FetchLiftInput {
    pub dimension: FetchLiftDimension,
    pub in_placement: Mapping,
    pub out_placement: Mapping,
}

/// Dimensions used to configure the Fetch engine.
pub struct FetchDimensionsInput {
    pub cluster_size: usize,
    pub slice_size: usize,
}

impl Display for FetchLiftDimension {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chip => write!(f, "chip"),
            Self::Cluster => write!(f, "cluster"),
            Self::Slice => write!(f, "slice"),
        }
    }
}

/// Checks the Fetch engine's cluster and slice dimensions.
pub fn config_fetch_dimensions(input: FetchDimensionsInput) -> Result<(), FetchError> {
    let FetchDimensionsInput {
        cluster_size,
        slice_size,
    } = input;
    if !FETCH_VALID_CLUSTER_SIZES.contains(&cluster_size) {
        return Err(FetchError::ClusterSize(cluster_size));
    }
    if !FETCH_VALID_SLICE_SIZES.contains(&slice_size) {
        return Err(FetchError::SliceSize(slice_size));
    }
    Ok(())
}

/// Returns the axes replacing size-2 broadcasts in one placement dimension.
pub fn config_fetch_lift(input: FetchLiftInput) -> Result<Mapping, FetchLiftError> {
    let FetchLiftInput {
        dimension,
        in_placement,
        out_placement,
    } = input;
    if in_placement.size() != out_placement.size() {
        return Err(FetchLiftError::PlacementSizeMismatch {
            dimension,
            in_size: in_placement.size(),
            out_size: out_placement.size(),
            in_placement,
            out_placement,
        });
    }
    if in_placement == out_placement {
        return Err(FetchLiftError::NoAxisLifted { dimension });
    }
    if !in_placement.size().is_power_of_two() {
        return Err(FetchLiftError::UnsupportedPlacementSize {
            dimension,
            size: in_placement.size(),
            placement: in_placement,
        });
    }
    let input_slots = into_slots_of_size_two(&in_placement);
    let output_slots = into_slots_of_size_two(&out_placement);
    let mut lifted = Mapping::identity();
    let mut stride = in_placement.size() / 2;
    for (input, output) in zip(input_slots, output_slots) {
        if input == output {
            stride /= 2;
            continue;
        }
        if !matches!(&input, Mapping::Broadcast { size: 2 }) {
            return Err(FetchLiftError::PlacementMismatch {
                dimension,
                stride,
                in_slot: input,
                out_slot: output,
            });
        }
        if output.idents().is_empty() {
            return Err(FetchLiftError::PaddingLift {
                dimension,
                stride,
                padding: output,
            });
        }
        lifted = lifted.pair(output);
        stride /= 2;
    }
    let lifted = lifted.normalize();
    if lifted.idents().is_empty() {
        return Err(FetchLiftError::NoAxisLifted { dimension });
    }
    Ok(lifted)
}

#[cfg(test)]
mod tests {
    // Glob import: the `m!` macro expands to DSL type-level structs that must all be in scope.
    use furiosa_mapping::*;

    use super::*;
    use crate::{
        FetchBaseAlignedInput, FetchBaseError, FetchError, FetchInput, FetchLiftBasesInput, MAX_SEQUENCER_ENTRIES,
        config_fetch, config_fetch_base_aligned, config_fetch_lift_bases,
    };

    axes![Feat = 128, Quad = 4, Half = 2, Other = 2, Third = 3, Oct = 8];
    // Two size-4 broadcasts with a live axis between them, filled by four element axes in an order of
    // their own: `[4, X, 4]` -> `[D, A, X, C, B]` over an element of `[A, B, C, D, V]`.
    axes![X = 16, A = 2, B = 2, C = 2, D = 2, V = 16, Rows = 63, OddRows = 85];

    const TEST_DIMENSION: FetchLiftDimension = FetchLiftDimension::Slice;

    #[test]
    fn unchanged_placement_rejects() {
        let placement = <m![Feat, 2]>::to_value();
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: placement.clone(),
                out_placement: placement,
            }),
            Err(FetchLiftError::NoAxisLifted {
                dimension: TEST_DIMENSION,
            })
        );
    }

    #[test]
    fn equivalent_placement_spelling_rejects() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Quad]>::to_value(),
                out_placement: <m![Quad / 2, Quad % 2]>::to_value(),
            }),
            Err(FetchLiftError::NoAxisLifted {
                dimension: TEST_DIMENSION,
            })
        );
    }

    #[test]
    fn non_binary_placement_rejects() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![3]>::to_value(),
                out_placement: <m![Third]>::to_value(),
            }),
            Err(FetchLiftError::UnsupportedPlacementSize {
                dimension: TEST_DIMENSION,
                size: 3,
                placement: <m![3]>::to_value(),
            })
        );
    }

    /// Fills the inner two-slice broadcast with `Quad`'s outer digit.
    #[test]
    fn filled_broadcast_lifts_its_axis() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Feat, 2]>::to_value(),
                out_placement: <m![Feat, Quad / 2]>::to_value(),
            })
            .unwrap(),
            <m![Quad / 2]>::to_value()
        );
    }

    #[test]
    fn padding_lift_recommends_reshape() {
        let error = config_fetch_lift(FetchLiftInput {
            dimension: TEST_DIMENSION,
            in_placement: <m![Feat, 2]>::to_value(),
            out_placement: <m![Feat, 1 # 2]>::to_value(),
        })
        .unwrap_err();
        assert_eq!(
            error,
            FetchLiftError::PaddingLift {
                dimension: FetchLiftDimension::Slice,
                stride: 1,
                padding: <m![1 # 2]>::to_value(),
            }
        );
        assert_eq!(
            error.to_string(),
            "slice lift replaces the input broadcast at stride 1 with padding 1 # 2; reshape the DM \
             placement and leave the corresponding input padding unread instead"
        );
    }

    #[test]
    fn plain_fetch_may_leave_padding_unread() {
        assert!(
            config_fetch(FetchInput {
                in_time: <m![1 # 2]>::to_value(),
                in_packet: <m![Oct]>::to_value(),
                out_time: <m![1]>::to_value(),
                out_packet: <m![Oct]>::to_value(),
                lifted: None,
            })
            .is_ok()
        );
    }

    #[test]
    fn input_padding_cannot_change() {
        assert!(matches!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Feat, 1 # 2]>::to_value(),
                out_placement: <m![Feat, Half]>::to_value(),
            }),
            Err(FetchLiftError::PlacementMismatch { .. })
        ));
    }

    #[test]
    fn only_changed_broadcast_slots_are_lifted() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Feat, 2, 2]>::to_value(),
                out_placement: <m![Feat, Half, 2]>::to_value(),
            })
            .unwrap(),
            <m![Half]>::to_value()
        );
    }

    #[test]
    fn padded_binary_broadcast_lifts() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![[Rows, 2] # 256]>::to_value(),
                out_placement: <m![[Rows, Half] # 256]>::to_value(),
            })
            .unwrap(),
            <m![Half]>::to_value()
        );
    }

    #[test]
    fn two_broadcasts_lift_together() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![2, Feat, 2]>::to_value(),
                out_placement: <m![Half, Feat, Other]>::to_value(),
            })
            .unwrap(),
            <m![Half, Other]>::to_value()
        );
    }

    #[test]
    fn one_broadcast_holds_two_axes() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Feat, 4]>::to_value(),
                out_placement: <m![Feat, Half, Other]>::to_value(),
            })
            .unwrap(),
            <m![Half, Other]>::to_value()
        );
    }

    #[test]
    fn padded_odd_broadcast_rejects() {
        assert!(matches!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![[OddRows, 3] # 256]>::to_value(),
                out_placement: <m![[OddRows, Third] # 256]>::to_value(),
            }),
            Err(FetchLiftError::PlacementMismatch { .. })
        ));
    }

    /// A broadcast filled by an axis of a different size is not a lift: the stream would place a
    /// different placement size than the DM holds.
    #[test]
    fn wrong_sized_fill_rejects() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Feat, 2]>::to_value(),
                out_placement: <m![Feat, Quad]>::to_value(),
            }),
            Err(FetchLiftError::PlacementSizeMismatch {
                dimension: TEST_DIMENSION,
                in_size: 256,
                out_size: 512,
                in_placement: <m![Feat, 2]>::to_value(),
                out_placement: <m![Feat, Quad]>::to_value(),
            })
        );
    }

    /// A live DM axis moved out of the placement is not a lift either: only a broadcast may be
    #[test]
    fn relabelled_live_axis_rejects() {
        assert!(matches!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![Feat, Half]>::to_value(),
                out_placement: <m![Feat, Quad / 2]>::to_value(),
            }),
            Err(FetchLiftError::PlacementMismatch { .. })
        ));
    }

    /// A lifted outer digit leaves a contiguous inner packet.
    #[test]
    fn lifted_axis_leaves_contiguous_packet() {
        let config = config_fetch(FetchInput {
            in_time: <m![1]>::to_value(),
            in_packet: <m![Quad]>::to_value(),
            out_time: <m![1]>::to_value(),
            out_packet: <m![Quad % 2]>::to_value(),
            lifted: Some(<m![Quad / 2]>::to_value()),
        })
        .unwrap();

        assert_eq!(
            config.packet.0.iter().map(|(_, e)| e.memory_stride).collect::<Vec<_>>(),
            vec![1]
        );
    }

    /// Rejects a base that is aligned for wide elements but not for narrow ones.
    #[test]
    fn unaligned_base_rejects() {
        let bases = [0, 16];
        assert!(
            config_fetch_base_aligned(FetchBaseAlignedInput {
                bases: bases.to_vec(),
                element_bits: 16,
            })
            .is_ok()
        );
        assert_eq!(
            config_fetch_base_aligned(FetchBaseAlignedInput {
                bases: bases.to_vec(),
                element_bits: 2,
            }),
            Err(FetchBaseError::UnalignedBase { position: 1, bits: 32 })
        );
        assert_eq!(
            config_fetch_base_aligned(FetchBaseAlignedInput {
                bases: vec![0, 1],
                element_bits: 4,
            }),
            Err(FetchBaseError::UnalignedBase { position: 1, bits: 4 })
        );
        assert_eq!(
            config_fetch_base_aligned(FetchBaseAlignedInput {
                bases: vec![usize::MAX / 2 + 1],
                element_bits: 2,
            }),
            Err(FetchBaseError::BaseOffsetOverflow {
                position: 0,
                base: usize::MAX / 2 + 1,
                element_bits: 2,
            })
        );
    }

    /// The 8 entries are the in-slice read's budget; the base's own entries are outside it.
    #[test]
    fn entry_budget_counts_the_unlifted_axes() {
        let dm = <m![A, B, C, D, Half, Other, Quad, Oct, X]>::to_value();
        let one_lifted = <m![X, Oct, Quad, Other, Half, D, C, B]>::to_value();
        let two_lifted = <m![X, Oct, Quad, Other, Half, D, C]>::to_value();
        let packet = <m![V]>::to_value();

        assert_eq!(
            config_fetch(FetchInput {
                in_time: dm.clone(),
                in_packet: packet.clone(),
                out_time: one_lifted,
                out_packet: packet.clone(),
                lifted: Some(<m![A]>::to_value()),
            }),
            Err(FetchError::TooManyEntries { needed: 9 })
        );

        let config = config_fetch(FetchInput {
            in_time: dm,
            in_packet: packet.clone(),
            out_time: two_lifted,
            out_packet: packet,
            lifted: Some(<m![A, B]>::to_value()),
        })
        .unwrap();
        assert_eq!(config.time.0.len() + config.packet.0.len(), MAX_SEQUENCER_ENTRIES);
    }

    /// Locates one base per grid position and preserves the lifted stride.
    #[test]
    fn grid_bases_step_by_the_lifted_stride() {
        let grid = <m![1, 2]>::to_value().pair(<m![Feat, Quad / 2]>::to_value());
        let bases = config_fetch_lift_bases(FetchLiftBasesInput {
            grid,
            dm: <m![Quad, Oct]>::to_value(),
        })
        .unwrap();

        assert_eq!(bases.len(), 512);
        assert!(bases.chunks(2).all(|pair| pair == [0, 16]), "{bases:?}");
    }

    #[test]
    fn rings_hold_reordered_axes() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: TEST_DIMENSION,
                in_placement: <m![4, X, 4]>::to_value(),
                out_placement: <m![D, A, X, C, B]>::to_value(),
            })
            .unwrap(),
            <m![D, A, C, B]>::to_value()
        );
    }

    /// The base for that read is located positionally, so the free order costs nothing:
    /// position `d*128 + a*64 + x*4 + c*2 + b` reads at `a*128 + b*64 + c*32 + d*16` in the element
    /// `[A, B, C, D, V]`, whose strides are `A = 128`, `B = 64`, `C = 32`, `D = 16`.
    #[test]
    fn reordered_broadcasts_locate_their_own_base() {
        let grid = <m![D, A, X, C, B]>::to_value();
        let bases = config_fetch_lift_bases(FetchLiftBasesInput {
            grid,
            dm: <m![A, B, C, D, V]>::to_value(),
        })
        .unwrap();

        assert_eq!(bases.len(), 256);
        for (position, &base) in bases.iter().enumerate() {
            let (d, a, x, c, b) = (
                position / 128,
                (position / 64) % 2,
                (position / 4) % 16,
                (position / 2) % 2,
                position % 2,
            );
            assert_eq!(
                base,
                a * 128 + b * 64 + c * 32 + d * 16,
                "position {position} (x = {x})"
            );
        }
    }

    /// A stream axis the DM never held stays a broadcast: `Half` is not a DM axis, so reading it at
    /// stride 0 replicates the cell rather than shadowing another read.
    #[test]
    fn unbacked_axis_still_broadcasts() {
        let config = config_fetch(FetchInput {
            in_time: <m![1]>::to_value(),
            in_packet: <m![Quad]>::to_value(),
            out_time: <m![Half]>::to_value(),
            out_packet: <m![Quad % 2]>::to_value(),
            lifted: Some(<m![Quad / 2]>::to_value()),
        });
        let config = config.unwrap();

        assert_eq!(
            config
                .time
                .0
                .iter()
                .map(|(_, entry)| (entry.mapping.clone(), entry.memory_stride))
                .collect::<Vec<_>>(),
            vec![(<m![Half]>::to_value(), 0)]
        );
        assert_eq!(
            config
                .packet
                .0
                .iter()
                .map(|(_, entry)| (entry.mapping.clone(), entry.memory_stride))
                .collect::<Vec<_>>(),
            vec![(<m![Quad % 2]>::to_value(), 1)]
        );
    }

    #[test]
    fn innermost_packet_broadcast_rejects() {
        assert_eq!(
            config_fetch(FetchInput {
                in_time: <m![1]>::to_value(),
                in_packet: <m![Quad]>::to_value(),
                out_time: <m![1]>::to_value(),
                out_packet: <m![Quad % 2, Half]>::to_value(),
                lifted: Some(<m![Quad / 2]>::to_value()),
            }),
            Err(FetchError::NonContiguousPacket {
                innermost: <m![Half]>::to_value(),
                memory_stride: 0,
            })
        );
    }

    #[test]
    fn uncovered_element_rejects() {
        assert!(matches!(
            config_fetch(FetchInput {
                in_time: <m![1]>::to_value(),
                in_packet: <m![Quad, Half]>::to_value(),
                out_time: <m![1]>::to_value(),
                out_packet: <m![Half]>::to_value(),
                lifted: Some(<m![Quad / 2]>::to_value()),
            }),
            Err(FetchError::Unread { .. })
        ));
    }
}
