//! Tests for `config_tile`: what a tile view must state, and what it may not.

use furiosa_mapping::*;
use furiosa_opt_lower::{TileError, TileInput, TileMappingInput, config_tile, tile_mapping};

axes![L1 = 192, L2 = 384, L5 = 256, B = 4096, A = 8, I = 3, Z = 24, One = 1];

#[test]
fn derives_the_tile_mapping_without_an_expected_type() {
    let mapping = tile_mapping(TileMappingInput {
        index: <m![I]>::to_value(),
        element: <m![I, A]>::to_value(),
        len: 1,
        hole_fill: PaddingKind::Top,
    })
    .unwrap();

    assert_eq!(mapping, <m![I = 1 # 3, A]>::to_value().normalize());
}

/// A tile must state every padding the split leaves, its own hole and the base layout's alike.
mod states_the_split_in_full {
    use super::*;

    #[test]
    fn tile_hole() {
        config_tile(TileInput {
            index: <m![I]>::to_value(),
            element: <m![I, A]>::to_value(),
            expected: <m![I = 1 # 3, A]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap();
    }

    #[test]
    fn base_padding() {
        config_tile(TileInput {
            index: <m![L1 # 256]>::to_value(),
            element: <m![L1 # 256, B % 64]>::to_value(),
            expected: <m![L1 = 192 # 256, B % 64]>::to_value(),
            len: 192,
            hole_fill: PaddingKind::Top,
        })
        .unwrap();
    }

    /// A padded axis whose 21 live rows are not three chunks of 8.
    #[test]
    fn padded_axis_whose_live_extent_does_not_divide() {
        config_tile(TileInput {
            index: <m![Z = 21 # 24 / 8]>::to_value(),
            element: <m![Z = 21 # 24, A]>::to_value(),
            expected: <m![1 # 3, Z = 21 # 24 % 8, A]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap();
    }
}

mod rejects {
    use super::*;

    /// An omitted outermost hole is rejected: reaching less far than the buffer is `unpad`'s job.
    #[test]
    fn outermost_hole_omitted() {
        let error = config_tile(TileInput {
            index: <m![I]>::to_value(),
            element: <m![I, A]>::to_value(),
            expected: <m![A]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("does not match the requested view"),
            "{error}"
        );
    }

    /// The same for padding the base layout carried, which is an unpad of the buffer and not a tile.
    #[test]
    fn base_padding_omitted() {
        let error = config_tile(TileInput {
            index: <m![L1 # 256]>::to_value(),
            element: <m![L1 # 256, B % 64]>::to_value(),
            expected: <m![L1, B % 64]>::to_value(),
            len: 192,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("does not match the requested view"),
            "{error}"
        );
    }

    /// Padding on an inner factor may not be dropped, as in a101's `block::linear2` weight tile.
    #[test]
    fn inner_padding_omitted() {
        let error = config_tile(TileInput {
            index: <m![L2 % 96 / 32]>::to_value(),
            element: <m![L5 / 16, L2 / 96, L2 % 96 / 32, L2 % 32 / 4, L5 % 16, L2 % 4]>::to_value(),
            expected: <m![L5 / 16, L2 / 96, L2 % 32 / 4, L5 % 16, L2 % 4]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("does not match the requested view"),
            "{error}"
        );
    }

    /// Naming that inner hole is accepted, so the rejection above concerns the omission alone.
    #[test]
    fn inner_padding_named_is_accepted() {
        config_tile(TileInput {
            index: <m![L2 % 96 / 32]>::to_value(),
            element: <m![L5 / 16, L2 / 96, L2 % 96 / 32, L2 % 32 / 4, L5 % 16, L2 % 4]>::to_value(),
            expected: <m![L5 / 16, L2 / 96, L2 % 96 / 32 = 1 # 3, L2 % 32 / 4, L5 % 16, L2 % 4]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap();
    }

    /// A live axis the split does not produce is a mismatch, not a droppable padding.
    #[test]
    fn wrong_live_axis() {
        let error = config_tile(TileInput {
            index: <m![I]>::to_value(),
            element: <m![I, A]>::to_value(),
            expected: <m![A / 2]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("does not match the requested view"),
            "{error}"
        );
    }

    /// The named padding must be the requested kind: a read tile annotated `Bottom` is rejected.
    #[test]
    fn read_tile_named_hole_with_write_padding() {
        let error = config_tile(TileInput {
            index: <m![I]>::to_value(),
            element: <m![I, A]>::to_value(),
            expected: <m![I = 1 #{!} 3, A]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("does not match the requested view"),
            "{error}"
        );
    }

    /// A write tile omitting its hole is rejected, the hole being what keeps the commit sequencer out.
    #[test]
    fn write_tile_omitted_padding() {
        let error = config_tile(TileInput {
            index: <m![I]>::to_value(),
            element: <m![I, A]>::to_value(),
            expected: <m![A]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("does not match the requested view"),
            "{error}"
        );
    }

    /// An index naming no axis of the buffer, reported as its own error rather than a view mismatch.
    #[test]
    fn index_that_is_not_an_axis_of_the_element() {
        let error = config_tile(TileInput {
            index: <m![L2 % 96 / 32]>::to_value(),
            element: <m![I, A]>::to_value(),
            expected: <m![I, A]>::to_value(),
            len: 1,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(matches!(error, TileError::Split), "{error}");
    }

    /// A window wider than the axis, which `Mapping::resize` asserts on.
    #[test]
    fn window_wider_than_the_axis() {
        for len in [0, 4] {
            let error = config_tile(TileInput {
                index: <m![I]>::to_value(),
                element: <m![I, A]>::to_value(),
                expected: <m![I = 1 # 3, A]>::to_value(),
                len,
                hole_fill: PaddingKind::Top,
            })
            .unwrap_err();
            assert!(matches!(error, TileError::Split), "len {len}: {error}");
        }
    }

    /// The same check on the unit-axis path, which returns before the narrowing.
    #[test]
    fn unit_axis_with_a_wide_window() {
        let error = config_tile(TileInput {
            index: <m![One]>::to_value(),
            element: <m![One, A]>::to_value(),
            expected: <m![One, A]>::to_value(),
            len: 5,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_err();
        assert!(matches!(error, TileError::Split), "{error}");
    }
}
