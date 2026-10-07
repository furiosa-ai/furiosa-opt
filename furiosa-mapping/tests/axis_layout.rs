//! Tests for the `MappingExt` methods that locate an axis and cut it out: `split_at`, `find_axis`,
//! `remove_axis` and `window_axis`. They share one set of axes, so a symbol means one thing
//! throughout. The cases cover what a projection loses about an axis: its padding, the context a
//! modulo discards, and where the buffer keeps its symbols. Each refusal case names the
//! precondition it broke.

use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, m};

axes![
    A = 4,
    A2 = 10,
    B = 6,
    Q = 8,
    D = 16,
    K = 8,
    P = 21,
    T = 3,
    W = 151936,
    Hid = 64,
    H = 4,
    Z = 24
];

#[test]
fn invalid_split_boundaries_return_errors() {
    let mapping = Mapping::Broadcast { size: 6 };
    for target in [0, 4] {
        assert_eq!(mapping.split_at(target), Err(SplitAtError { size: 6, target }));
    }
    assert_eq!(
        Mapping::Broadcast { size: 0 }.split_at(1),
        Err(SplitAtError { size: 0, target: 1 })
    );
}

#[test]
fn test_padded_axis_is_measured_across_its_padding() {
    // `Z = 24` padded to 32, chunked by 8: four chunks of 8 rows, each row `H = 4` cells.
    let table = <m![Z # 32, H]>::to_value();

    let stride = table.find_axis(&<m![Z # 32 / 8]>::to_value());

    assert_eq!(stride, Ok(8 * H::SIZE));
    // The extent comes from the axis and includes padding: three chunks hold live rows, four exist.
    assert_eq!(<m![Z # 32 / 8]>::to_value().size(), 4);
}

#[test]
fn test_live_extent_indivisible_by_the_chunk_still_locates() {
    // Live 21 padded to 24, chunked by 8. The live rows do not divide into three chunks, which
    // `config_tile` refuses, but the axis is still located.
    let table = <m![P # 24, H]>::to_value();

    assert_eq!(table.find_axis(&<m![P # 24 / 8]>::to_value()), Ok(8 * H::SIZE));
}

#[test]
fn test_tutorial_lm_head_shape() {
    // `W = 151936` padded to `155648 = 19 * 8192`, over a hidden size of 64.
    let table = <m![W # 155648, Hid]>::to_value();

    assert_eq!(
        table.find_axis(&<m![W # 155648 / 8192]>::to_value()),
        Ok(8192 * Hid::SIZE)
    );
}

#[test]
fn test_axis_that_is_not_outermost() {
    // `Q` sits between `A` and `D`, so peeling the outermost term would report `A` instead.
    let table = <m![A, Q, D]>::to_value();

    assert_eq!(table.find_axis(&<m![Q / 2]>::to_value()), Ok(2 * D::SIZE));
    assert_eq!(table.find_axis(&<m![Q / 4]>::to_value()), Ok(4 * D::SIZE));
}

#[test]
fn test_axis_spanning_more_than_one_symbol() {
    // `B / 2` cuts inside `B`, so the axis is `A` paired with half of `B` and belongs to neither symbol.
    let table = <m![A, B]>::to_value();

    assert_eq!(table.find_axis(&<m![A, B / 2]>::to_value()), Ok(2));
}

#[test]
fn test_unpadded_axis_is_unaffected() {
    let table = <m![Z, H]>::to_value();

    assert_eq!(table.find_axis(&<m![Z / 8]>::to_value()), Ok(8 * H::SIZE));
}

#[test]
fn test_axis_absent_from_the_buffer_is_refused() {
    let table = <m![A, D]>::to_value();

    assert_eq!(
        table.find_axis(&<m![Q / 2]>::to_value()),
        Err(FindAxisError::NotInMapping)
    );
}

/// Every cell of `(A, D)` is in `m![A, Q, D]`, but at two strides, so no single stride advances the
/// axis: reading it out would have to step over the `Q` cells in between.
#[test]
fn test_an_axis_split_across_the_buffer_is_refused() {
    let axis = <m![A, D]>::to_value();

    assert_eq!(
        <m![A, Q, D]>::to_value().find_axis(&axis),
        Err(FindAxisError::ScatteredInMapping)
    );
    // The same two symbols adjacent are located, so it is the gap that is refused, not the pair.
    assert_eq!(<m![Q, A, D]>::to_value().find_axis(&axis), Ok(1));
}

/// A symbol the buffer lacks reads no memory, so only part of the axis is found: `(B, D)` puts 16 of
/// its 96 cells in the buffer, wherever `B` sits. That is a missing axis, not a scattered one, and it
/// pins the order of the two checks, since the broadcast also leaves the axis in two entries.
#[test]
fn test_a_partly_broadcast_axis_is_refused_as_missing() {
    let table = <m![A, D]>::to_value();

    assert_eq!(
        table.find_axis(&<m![B, D]>::to_value()),
        Err(FindAxisError::NotInMapping)
    );
    assert_eq!(
        table.find_axis(&<m![A, B]>::to_value()),
        Err(FindAxisError::NotInMapping)
    );
}

/// `m![A]` and `m![A # 8]` both step by `H`, and the entry reports the padded 8 for either. Checking
/// that report against `axis.size()` would reject the live form.
#[test]
fn test_live_axis_in_a_buffer_laid_out_for_its_padding() {
    let table = <m![A # 8, H]>::to_value();

    assert_eq!(table.find_axis(&<m![A]>::to_value()), Ok(H::SIZE));
    assert_eq!(table.find_axis(&<m![A # 8]>::to_value()), Ok(H::SIZE));
}

/// The mapping is the padded axis itself, so the entry covers 8 cells while the axis states 4.
#[test]
fn test_a_buffer_that_is_nothing_but_the_padded_axis() {
    let table = <m![A # 8]>::to_value();

    assert_eq!(table.find_axis(&<m![A]>::to_value()), Ok(1));
    assert_eq!(table.find_axis(&<m![A # 8]>::to_value()), Ok(1));
}

/// Locating and windowing differ here. `T = 3` is located at stride `H`, and `window_axis` rejects it
/// because a span of 12 cannot be cut out of 32. `m![T # 8]` reaches the same window.
#[test]
fn test_a_live_extent_that_does_not_divide_is_still_located() {
    let table = <m![T # 8, H]>::to_value();
    let one_position = <m![H # 32]>::to_value().normalize();

    assert_eq!(table.find_axis(&<m![T]>::to_value()), Ok(H::SIZE));
    assert_eq!(
        table.window_axis(H::SIZE, T::SIZE, 1, PaddingKind::Top),
        Err(WindowAxisError::SpanDoesNotDivide {
            stride: H::SIZE,
            size: T::SIZE,
            cells: table.size()
        })
    );

    assert_eq!(table.find_axis(&<m![T # 8]>::to_value()), Ok(H::SIZE));
    assert_eq!(table.window_axis(H::SIZE, 8, 1, PaddingKind::Top), Ok(one_position));
}

#[test]
fn test_a_context_the_buffer_contradicts_is_located_but_not_removable() {
    let table = <m![A2]>::to_value();

    // `% 4` locates a partial axis, but the whole region cannot be removed.
    assert_eq!(table.find_axis(&<m![A2 # 12 % 4]>::to_value()), Ok(1));
    assert_eq!(
        table.remove_axis(&<m![A2 # 12 % 4]>::to_value()),
        Err(SplitAtError {
            size: A2::SIZE,
            target: 4
        }
        .into())
    );
    // Without the modulo the span itself is checked, and locating already fails.
    assert_eq!(
        table.find_axis(&<m![A2 # 12]>::to_value()),
        Err(SplitAtError {
            size: A2::SIZE,
            target: 12
        }
        .into())
    );
}

/// Rejected whether the mapping carries a second axis (`m![A, H]`) or is the axis alone (`m![A]`).
#[test]
fn test_an_axis_wider_than_its_buffer_is_refused() {
    let padded_axis = <m![A # 8]>::to_value();

    assert_eq!(
        <m![A, H]>::to_value().find_axis(&padded_axis),
        Err(SplitAtError {
            size: A::SIZE * H::SIZE,
            target: H::SIZE * 8
        }
        .into())
    );
    assert_eq!(
        <m![A]>::to_value().find_axis(&padded_axis),
        Err(SplitAtError {
            size: A::SIZE,
            target: 8
        }
        .into())
    );
}

#[test]
fn test_the_whole_buffer_is_an_axis_of_stride_one() {
    let table = <m![A, B]>::to_value();

    assert_eq!(table.find_axis(&table), Ok(1));
}

/// Sweeps the postcondition `find_axis` does not assert: for an axis taken by `split_at`, peeling the
/// reported stride back out reproduces that axis.
#[test]
fn test_reported_stride_satisfies_the_windowing_precondition() -> eyre::Result<()> {
    for table in [
        <m![Z # 32, H]>::to_value(),
        <m![A, Q, D]>::to_value(),
        <m![A, B]>::to_value(),
    ] {
        let n = table.size();
        for target in (1..=n).filter(|t| n.is_multiple_of(*t)) {
            let axis = table.split_at(target)?.0;
            let Ok(stride) = table.find_axis(&axis) else { continue };
            let size = axis.size();

            let peeled = table.split_at(stride * size)?.1.split_at(stride)?.0;

            assert_eq!(
                peeled.normalize(),
                axis.normalize(),
                "stride {stride} x {size} does not peel back to {} in {}",
                axis.normalize(),
                table.normalize()
            );
        }
    }
    Ok(())
}

/// Sweeps the single-run rule over every subset of a buffer's terms: an axis is located exactly when
/// its terms are adjacent, and each located one peels back out at the reported stride.
#[test]
fn test_only_adjacent_terms_form_an_axis() -> eyre::Result<()> {
    let terms = [
        <m![A]>::to_value(),
        <m![Q]>::to_value(),
        <m![D]>::to_value(),
        <m![T]>::to_value(),
    ];
    let table = terms.iter().cloned().reduce(|outer, inner| outer.pair(inner)).unwrap();

    for subset in 1..1u32 << terms.len() {
        let picked: Vec<usize> = (0..terms.len()).filter(|i| subset >> i & 1 == 1).collect();
        let axis = picked
            .iter()
            .map(|i| terms[*i].clone())
            .reduce(|outer, inner| outer.pair(inner))
            .unwrap();
        let adjacent = picked.windows(2).all(|step| step[1] == step[0] + 1);

        let located = table.find_axis(&axis);

        if !adjacent {
            assert_eq!(
                located,
                Err(FindAxisError::ScatteredInMapping),
                "{} is scattered in {}",
                axis.normalize(),
                table.normalize()
            );
            continue;
        }
        let Ok(stride) = located else {
            panic!(
                "{} is a run of {} and must be located, got {located:?}",
                axis.normalize(),
                table.normalize()
            )
        };
        let peeled = table.split_at(stride * axis.size())?.1.split_at(stride)?.0;
        assert_eq!(
            peeled.normalize(),
            axis.normalize(),
            "stride {stride} does not peel back to {}",
            axis.normalize()
        );
    }
    Ok(())
}

/// An axis padded past the end of the mapping is rejected. The sequencer absorbs padding on both sides,
/// so it reports a stride over an extent that is not there: 32 cells over 8 positions, in 128.
#[test]
fn test_axis_padded_past_the_buffer_is_refused() {
    let table = <m![Z # 32, H]>::to_value();
    let sane = <m![Z # 32 / 8]>::to_value();
    let over = sane.clone().padding(8, PaddingKind::Top);

    assert_eq!(table.find_axis(&sane), Ok(32), "the 4-chunk axis is there");
    assert_eq!(over.size(), 8, "the widened axis claims twice the chunks");
    assert_eq!(
        table.find_axis(&over),
        Err(SplitAtError {
            size: table.size(),
            target: 32 * 8
        }
        .into()),
        "and a buffer of 128 cells does not hold 8 x 32"
    );
}

/// Padding kind must agree. [`PaddingRule::Exact`] accepts a pad only against the same kind, where a
/// read carve would absorb a `Top` over-read.
#[test]
fn test_padding_kind_must_agree() {
    let buffers = [
        <m![Z # 32, H]>::to_value(),
        <m![Z #{!} 32, H]>::to_value(),
        <m![Z #{0} 32, H]>::to_value(),
    ];
    let spellings = [
        <m![Z # 32 / 8]>::to_value(),
        <m![Z #{!} 32 / 8]>::to_value(),
        <m![Z #{0} 32 / 8]>::to_value(),
    ];

    for (declared, buffer) in buffers.iter().enumerate() {
        for (asked, axis) in spellings.iter().enumerate() {
            let located = buffer.find_axis(axis);
            if declared == asked {
                assert_eq!(located, Ok(32), "{} in {}", axis.normalize(), buffer.normalize());
            } else {
                // The error variant depends on where the disagreement is caught. A `Zero` stream
                // cannot read a `Top` pad at all, so that pair is rejected one step earlier.
                assert!(
                    located.is_err(),
                    "{} must not be located in {}, got {located:?}",
                    axis.normalize(),
                    buffer.normalize()
                );
            }
        }
    }
}

#[test]
fn test_padding_kind_must_agree_for_an_interior_hole() {
    let top_buffer = <m![A # 8, D]>::to_value();
    let bottom_buffer = <m![A #{!} 8, D]>::to_value();
    let top_axis = <m![A # 8]>::to_value();
    let bottom_axis = <m![A #{!} 8]>::to_value();

    assert_eq!(top_buffer.find_axis(&top_axis), Ok(D::SIZE));
    assert_eq!(bottom_buffer.find_axis(&bottom_axis), Ok(D::SIZE));
    assert!(top_buffer.find_axis(&bottom_axis).is_err());
    assert!(bottom_buffer.find_axis(&top_axis).is_err());
}

/// A composite spanning two symbols and divided again that is located, so the gap below is narrower
/// than doubly divided composites in general.
#[test]
fn test_a_divided_composite_that_is_located() {
    let table = <m![P # 24, H]>::to_value();

    assert_eq!(table.find_axis(&<m![[P # 24, H] / 3]>::to_value()), Ok(3));
}

/// Desired behavior, not met: a composite spanning several symbols and divided again
/// (`(A, B / 2) / 2`) is read diagonally, so the sequencer matches no term and declines.
#[test]
#[ignore = "the sequencer matches no term for a doubly divided composite (backlog)"]
fn test_doubly_divided_composites_are_located() -> eyre::Result<()> {
    let cases: [(Mapping, usize); 3] = [
        (<m![A, B]>::to_value(), 4),
        (<m![A, B]>::to_value(), 8),
        (<m![P # 24, H]>::to_value(), 6),
    ];

    for (table, target) in cases {
        let axis = table.split_at(target)?.0;
        let size = axis.size();

        // The axis is present: peeling at the step it would have reproduces it.
        let peeled = table.split_at(target * size)?.1.split_at(target)?.0;
        assert_eq!(
            peeled.normalize(),
            axis.normalize(),
            "{} is a sub-layout of {} at stride {target}",
            axis.normalize(),
            table.normalize()
        );

        assert_eq!(
            table.find_axis(&axis),
            Ok(target),
            "{} in {} should be located at {target}",
            axis.normalize(),
            table.normalize()
        );
    }
    Ok(())
}

#[test]
fn finds_and_removes_one_axis_run() -> eyre::Result<()> {
    assert_eq!(<m![K, D]>::to_value().find_axis(&<m![K]>::to_value()), Ok(D::SIZE));
    assert_eq!(
        <m![K, D]>::to_value().remove_axis(&<m![K]>::to_value())?.normalize(),
        <m![D]>::to_value().normalize()
    );
    Ok(())
}

#[test]
fn preserves_unrelated_padding() -> eyre::Result<()> {
    let remainder = <m![1 # 64, K, D]>::to_value().remove_axis(&<m![K]>::to_value())?;

    assert_eq!(remainder.normalize(), <m![1 # 64, D]>::to_value().normalize());
    Ok(())
}

#[test]
fn rejects_an_axis_absent_from_the_mapping() {
    assert_eq!(
        <m![A, D]>::to_value().remove_axis(&<m![K]>::to_value()),
        Err(FindAxisError::NotInMapping),
    );
}

#[test]
fn rejects_an_axis_split_across_the_mapping() {
    assert_eq!(
        <m![K / 2, D, K % 2]>::to_value().remove_axis(&<m![K]>::to_value()),
        Err(FindAxisError::ScatteredInMapping),
    );
}

#[test]
fn rejects_an_axis_the_mapping_only_partly_holds() {
    assert!(<m![K / 2, D]>::to_value().remove_axis(&<m![K]>::to_value()).is_err());
}

#[test]
fn removing_no_axis_leaves_the_mapping_unchanged() {
    let table = <m![K, D]>::to_value();

    assert_eq!(table.remove_axis(&<m![1]>::to_value()), Ok(table.clone()));
}

#[test]
fn rejects_padding_that_names_no_axis_but_names_cells() {
    assert_eq!(
        <m![K, D]>::to_value().remove_axis(&<m![1 # 64]>::to_value()),
        Err(FindAxisError::NotInMapping),
    );
}

#[test]
fn test_narrows_the_located_axis() -> eyre::Result<()> {
    let table = <m![Z # 32, H]>::to_value();

    let windowed = table.window_axis(8 * H::SIZE, 4, 1, PaddingKind::Top)?;

    assert_eq!(windowed, <m![1 # 4, Z # 32 % 8, H]>::to_value().normalize());
    Ok(())
}

/// A window wider than its axis is rejected for any mapping. `Mapping::resize` asserts on it, so
/// `window_axis` must check it first.
#[test]
fn test_a_window_wider_than_the_axis_is_refused_as_such() {
    let table = <m![Z # 32, H]>::to_value();

    for window in [0, 5] {
        assert_eq!(
            table.window_axis(8 * H::SIZE, 4, window, PaddingKind::Top),
            Err(WindowAxisError::WindowExceedsAxis { window, size: 4 })
        );
    }
}

#[test]
fn test_an_axis_that_does_not_fit_the_mapping_is_refused_as_such() {
    let table = <m![Z # 32, H]>::to_value();

    assert_eq!(
        table.window_axis(8 * H::SIZE, 8, 1, PaddingKind::Top),
        Err(WindowAxisError::SpanDoesNotDivide {
            stride: 32,
            size: 8,
            cells: table.size()
        })
    );
}
