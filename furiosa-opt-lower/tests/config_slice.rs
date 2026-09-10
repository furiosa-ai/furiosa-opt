//! Tests for validating the mapping produced by removing a sliced axis.

use furiosa_mapping::*;
use furiosa_opt_lower::{
    ClusterPlacement, SliceError, SlicePatternError, SliceRequest, SramRedistributeEntry, config_outermost_dm_slice,
    config_slice, validate_chip_shuffle, validate_slice_indices, validate_sram_redistribution,
};

axes![A = 256, B = 4096, Chip = 2, Cluster = 2];

#[test]
fn chip_shuffle_accepts_a_permutation() {
    validate_chip_shuffle(&[1, 2, 3, 0], 4).unwrap();
    validate_chip_shuffle(&[1, 0], 2).unwrap();
}

#[test]
fn chip_shuffle_rejects_a_non_permutation() {
    for (sources, expected) in [
        (vec![0, 1], "expected one source per chip"),
        (vec![0, 1, 2, 3, 0], "expected one source per chip"),
        (vec![0, 1, 2, 9], "outside 0..4"),
        (vec![0, 1, 1, 2], "occurs more than once"),
    ] {
        let error = validate_chip_shuffle(&sources, 4).unwrap_err();
        assert!(error.to_string().contains(expected), "{sources:?}: {error}");
    }
}

#[test]
fn slice_indices_reject_a_wrong_count() {
    axes![Axis = 4];

    let error = validate_slice_indices(&[0, 1], 4, &<m![Axis]>::to_value()).unwrap_err();

    assert!(matches!(
        error,
        SlicePatternError::WrongIndexCount { num_targets: 4, .. }
    ));
}

#[test]
fn slice_indices_accept_a_padded_position() {
    axes![Axis = 4];

    validate_slice_indices(&[0, 1, 2, 4], 4, &<m![Axis # 8]>::to_value()).unwrap();
}

#[test]
fn slice_indices_reject_an_out_of_bounds_position() {
    axes![Axis = 4];

    let error = validate_slice_indices(&[0, 1, 2, 8], 4, &<m![Axis # 8]>::to_value()).unwrap_err();

    assert!(matches!(
        error,
        SlicePatternError::InvalidIndex {
            target: 3,
            index: 8,
            ..
        }
    ));
}

#[test]
fn sram_redistribution_indices_are_chip_major() {
    for chip in 0..2 {
        for cluster in 0..3 {
            let index = ClusterPlacement { chip, cluster }.to_target_index(3);
            assert_eq!(index, chip * 3 + cluster);
            assert_eq!(
                ClusterPlacement::from_target_index(index, 3),
                ClusterPlacement { chip, cluster }
            );
        }
    }
}

#[test]
fn removes_an_outer_axis() {
    config_slice(SliceRequest::Layout {
        element: <m![B / 512 % 4, B / 2048, B / 256 % 2, A % 2, A / 16]>::to_value(),
        axes: vec![<m![B / 512 % 4]>::to_value()],
        output: <m![B / 2048, B / 256 % 2, A % 2, A / 16]>::to_value(),
    })
    .unwrap();
}

#[test]
fn removes_an_inner_axis() {
    config_slice(SliceRequest::Layout {
        element: <m![B / 2048, B / 512 % 4, B / 256 % 2, A % 2, A / 16]>::to_value(),
        axes: vec![<m![B / 512 % 4]>::to_value()],
        output: <m![B / 2048, B / 256 % 2, A % 2, A / 16]>::to_value(),
    })
    .unwrap();
}

#[test]
fn rejects_a_result_that_keeps_the_axis() {
    let error = config_slice(SliceRequest::Layout {
        element: <m![B / 512 % 4, B / 2048, B / 256 % 2, A % 2, A / 16]>::to_value(),
        axes: vec![<m![B / 512 % 4]>::to_value()],
        output: <m![1 # 4, B / 2048, B / 256 % 2, A % 2, A / 16]>::to_value(),
    })
    .unwrap_err();

    assert!(matches!(error, SliceError::UnexpectedOutput { .. }));
}

#[test]
fn dm_slice_accepts_an_eight_byte_axis_stride() {
    axes![Outer = 2, Inner = 2];

    config_slice(SliceRequest::AlignedStrides {
        element: <m![Outer, Inner]>::to_value(),
        axes: vec![<m![Outer]>::to_value()],
        output: <m![Inner]>::to_value(),
        element_bits: 32,
    })
    .unwrap();
}

#[test]
fn dm_slice_rejects_an_unexpected_output() {
    axes![Outer = 2, Inner = 2];

    let error = config_slice(SliceRequest::AlignedStrides {
        element: <m![Outer, Inner]>::to_value(),
        axes: vec![<m![Outer]>::to_value()],
        output: <m![Outer]>::to_value(),
        element_bits: 32,
    })
    .unwrap_err();

    assert!(matches!(error, SliceError::UnexpectedOutput { .. }));
}

#[test]
fn dm_slice_rejects_an_unaligned_innermost_axis() {
    axes![Outer = 2, Inner = 2];

    let error = config_slice(SliceRequest::AlignedStrides {
        element: <m![Outer, Inner]>::to_value(),
        axes: vec![<m![Inner]>::to_value()],
        output: <m![Outer]>::to_value(),
        element_bits: 32,
    })
    .unwrap_err();

    assert!(matches!(
        error,
        SliceError::UnalignedDmStride {
            stride: 1,
            stride_bits: 32,
            ..
        }
    ));
    assert!(
        error
            .to_string()
            .contains("a DM output requires the sliced-axis stride")
    );
}

#[test]
fn dm_slice_locates_independent_noncontiguous_axes() {
    axes![OuterSlice = 2, Middle = 3, InnerSlice = 2, Tail = 2];

    let strides = config_slice(SliceRequest::AlignedStrides {
        element: <m![OuterSlice, Middle, InnerSlice, Tail]>::to_value(),
        axes: vec![<m![OuterSlice]>::to_value(), <m![InnerSlice]>::to_value()],
        output: <m![Middle, Tail]>::to_value(),
        element_bits: 32,
    })
    .unwrap()
    .strides;

    assert_eq!(strides, vec![12, 2]);
}

#[test]
fn dm_slice_keeps_a_pair_as_one_contiguous_axis() {
    axes![OuterSlice = 2, Middle = 3, InnerSlice = 2, Tail = 2];

    let strides = config_slice(SliceRequest::AlignedStrides {
        element: <m![OuterSlice, Middle, InnerSlice, Tail]>::to_value(),
        axes: vec![<m![OuterSlice, Middle]>::to_value()],
        output: <m![InnerSlice, Tail]>::to_value(),
        element_bits: 32,
    })
    .unwrap()
    .strides;

    assert_eq!(strides, vec![4]);
}

#[test]
fn slice_rejects_a_pair_whose_members_are_separated() {
    axes![Outer = 2, Middle = 3, Inner = 2];

    let error = config_slice(SliceRequest::Layout {
        element: <m![Outer, Middle, Inner]>::to_value(),
        axes: vec![<m![Outer, Inner]>::to_value()],
        output: <m![Middle]>::to_value(),
    })
    .unwrap_err();

    assert!(matches!(
        error,
        SliceError::AxisNotFound {
            source: FindAxisError::ScatteredInMapping,
            ..
        }
    ));
}

#[test]
fn dm_slice_preserves_requested_axis_order() {
    axes![OuterSlice = 2, Middle = 3, InnerSlice = 2, Tail = 2];

    let strides = config_slice(SliceRequest::AlignedStrides {
        element: <m![OuterSlice, Middle, InnerSlice, Tail]>::to_value(),
        axes: vec![<m![InnerSlice]>::to_value(), <m![OuterSlice]>::to_value()],
        output: <m![Middle, Tail]>::to_value(),
        element_bits: 32,
    })
    .unwrap()
    .strides;

    assert_eq!(strides, vec![2, 12]);
}

#[test]
fn slice_axes_survive_normalization_between_steps() {
    axes![X = 16, Middle = 2];

    config_slice(SliceRequest::Layout {
        element: <m![X / 4, Middle, X % 4]>::to_value(),
        axes: vec![<m![Middle]>::to_value(), <m![X % 4]>::to_value()],
        output: <m![X / 4]>::to_value(),
    })
    .unwrap();
}

#[test]
fn empty_slice_axes_accept_an_equivalent_factorization() {
    axes![X = 512, Tail = 8];

    config_slice(SliceRequest::Layout {
        element: <m![X / 4 % 128, X % 4, Tail]>::to_value(),
        axes: vec![],
        output: <m![X % 512, Tail]>::to_value(),
    })
    .unwrap();
}

#[test]
fn dm_slice_rejects_overlapping_axes() {
    axes![Slice = 2, Tail = 2];
    let axes = vec![<m![Slice]>::to_value(), <m![Slice]>::to_value()];

    let error = config_slice(SliceRequest::AlignedStrides {
        element: <m![Slice, Tail]>::to_value(),
        axes: axes.clone(),
        output: <m![Tail]>::to_value(),
        element_bits: 32,
    })
    .unwrap_err();

    assert!(matches!(error, SliceError::OverlappingAxes { axes: actual, .. } if actual == axes));
}

#[test]
fn dm_slice_checks_alignment_of_every_axis() {
    axes![Outer = 2, Middle = 2, Inner = 2];

    let error = config_slice(SliceRequest::AlignedStrides {
        element: <m![Outer, Middle, Inner]>::to_value(),
        axes: vec![<m![Outer]>::to_value(), <m![Inner]>::to_value()],
        output: <m![Middle]>::to_value(),
        element_bits: 32,
    })
    .unwrap_err();

    assert!(matches!(error, SliceError::UnalignedDmStride { axis, stride: 1, .. } if axis == <m![Inner]>::to_value()));
}

#[test]
fn incomplete_padded_axis_returns_an_error() {
    axes![Slice = 4];

    let error = config_slice(SliceRequest::Layout {
        element: <m![Slice # 6]>::to_value(),
        axes: vec![<m![Slice]>::to_value()],
        output: <m![1]>::to_value(),
    })
    .unwrap_err();

    assert!(matches!(error, SliceError::IncompleteAxis { .. }));
}

#[test]
fn incomplete_later_axis_reports_the_input_mapping() {
    axes![Outer = 2, Slice = 4];
    let element = <m![Outer, Slice # 6]>::to_value();

    let error = config_slice(SliceRequest::AlignedStrides {
        element: element.clone(),
        axes: vec![<m![Outer]>::to_value(), <m![Slice]>::to_value()],
        output: <m![1]>::to_value(),
        element_bits: 32,
    })
    .unwrap_err();

    assert!(matches!(error, SliceError::IncompleteAxis { element: actual, .. } if actual == element));
}

#[test]
fn outermost_dm_slice_returns_its_stride() {
    axes![Slice = 2, Tail = 4];

    let stride = config_outermost_dm_slice(
        <m![Slice]>::to_value(),
        <m![Slice, Tail]>::to_value(),
        <m![Tail]>::to_value(),
        32,
    )
    .unwrap();

    assert_eq!(stride, 4);
}

#[test]
fn outermost_dm_slice_requires_the_outermost_axis() {
    axes![Outer = 2, Slice = 2, Tail = 2];

    let error = config_outermost_dm_slice(
        <m![Slice]>::to_value(),
        <m![Outer, Slice, Tail]>::to_value(),
        <m![Outer, Tail]>::to_value(),
        32,
    )
    .unwrap_err();

    assert!(matches!(error, SliceError::NotOutermost { .. }));
}

#[test]
fn outermost_dm_slice_rejects_a_bottom_hole_in_its_tail() {
    axes![Slice = 2, Tail = 2];

    let error = config_outermost_dm_slice(
        <m![Slice]>::to_value(),
        <m![Slice, 1 #{!} 2, Tail]>::to_value(),
        <m![1 #{!} 2, Tail]>::to_value(),
        32,
    )
    .unwrap_err();

    assert!(matches!(error, SliceError::BottomHoleTail { .. }));
}

#[test]
fn shuffle_slice_accepts_a_padded_axis_position() {
    axes![Axis = 4];

    validate_sram_redistribution(
        &[SramRedistributeEntry {
            source_chip: 0,
            source_cluster: 0,
            slice_indices: vec![6],
        }],
        &<m![1]>::to_value(),
        &<m![1]>::to_value(),
        &[<m![Axis # 8]>::to_value()],
    )
    .unwrap();
}

#[test]
fn shuffle_slice_rejects_an_out_of_bounds_axis_position() {
    axes![Axis = 4];

    let error = validate_sram_redistribution(
        &[SramRedistributeEntry {
            source_chip: 0,
            source_cluster: 0,
            slice_indices: vec![8],
        }],
        &<m![1]>::to_value(),
        &<m![1]>::to_value(),
        &[<m![Axis # 8]>::to_value()],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SlicePatternError::InvalidIndex {
            target: 0,
            index: 8,
            ..
        }
    ));
}

#[test]
fn sram_redistribution_accepts_chip_shuffle_with_cluster_coordinates() {
    axes![Axis = 4];

    validate_sram_redistribution(
        &[
            SramRedistributeEntry {
                source_chip: 1,
                source_cluster: 0,
                slice_indices: vec![1],
            },
            SramRedistributeEntry {
                source_chip: 1,
                source_cluster: 1,
                slice_indices: vec![0],
            },
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 0,
                slice_indices: vec![1],
            },
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 1,
                slice_indices: vec![0],
            },
        ],
        &<m![Chip]>::to_value(),
        &<m![Cluster]>::to_value(),
        &[<m![Axis]>::to_value()],
    )
    .unwrap();
}

#[test]
fn sram_redistribution_rejects_duplicate_source_placement() {
    let error = validate_sram_redistribution(
        &[
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 0,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 0,
                slice_indices: vec![],
            },
        ],
        &<m![1]>::to_value(),
        &<m![Cluster]>::to_value(),
        &[],
    )
    .unwrap_err();

    assert!(matches!(error, SlicePatternError::DuplicatePlacement { .. }));
}

#[test]
fn sram_redistribution_rejects_source_chips_that_vary_within_a_chip() {
    let error = validate_sram_redistribution(
        &[
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 0,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 1,
                source_cluster: 0,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 1,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 1,
                source_cluster: 1,
                slice_indices: vec![],
            },
        ],
        &<m![Chip]>::to_value(),
        &<m![Cluster]>::to_value(),
        &[],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SlicePatternError::NonUniformSourceChip { target_chip: 0, .. }
    ));
}

#[test]
fn sram_redistribution_rejects_source_clusters_that_vary_between_chips() {
    let error = validate_sram_redistribution(
        &[
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 0,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 1,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 1,
                source_cluster: 1,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 1,
                source_cluster: 0,
                slice_indices: vec![],
            },
        ],
        &<m![Chip]>::to_value(),
        &<m![Cluster]>::to_value(),
        &[],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SlicePatternError::NonUniformSourceCluster { target_cluster: 0, .. }
    ));
}

#[test]
fn sram_redistribution_rejects_a_padded_source_for_a_live_target() {
    let error = validate_sram_redistribution(
        &[
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 1,
                slice_indices: vec![],
            },
            SramRedistributeEntry {
                source_chip: 0,
                source_cluster: 0,
                slice_indices: vec![],
            },
        ],
        &<m![1]>::to_value(),
        &<m![1 # 2]>::to_value(),
        &[],
    )
    .unwrap_err();

    assert!(matches!(error, SlicePatternError::PaddedSourcePlacement { .. }));
}
