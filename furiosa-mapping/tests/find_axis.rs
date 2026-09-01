//! Tests for `MappingExt::find_axis`. The cases cover what a projection loses about an axis: its
//! padding, the context a modulo discards, and where the buffer keeps its symbols.

use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, m};

axes![
    A = 4,
    A2 = 10,
    B = 6,
    Q = 8,
    D = 16,
    P = 21,
    T = 3,
    W = 151936,
    Hid = 64,
    H = 4,
    Z = 24
];

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
        Err(WindowAxisError::SpanExceedsMapping {
            stride: H::SIZE,
            size: T::SIZE,
            cells: table.size()
        })
    );

    assert_eq!(table.find_axis(&<m![T # 8]>::to_value()), Ok(H::SIZE));
    assert_eq!(table.window_axis(H::SIZE, 8, 1, PaddingKind::Top), Ok(one_position));
}

/// A context the mapping contradicts is not checked once a modulo has discarded it: `A # 12 % 4` keeps
/// only cells 0..3, which are live in a 10-cell `A`.
#[test]
fn test_a_context_the_buffer_contradicts_is_not_noticed() {
    let table = <m![A2]>::to_value();

    assert_eq!(table.find_axis(&<m![A2 # 12 % 4]>::to_value()), Ok(1));
    // Without the modulo the span is checked, and the same claim is rejected.
    assert_eq!(
        table.find_axis(&<m![A2 # 12]>::to_value()),
        Err(FindAxisError::SpanExceedsMapping)
    );
}

/// Rejected whether the mapping carries a second axis (`m![A, H]`) or is the axis alone (`m![A]`).
#[test]
fn test_an_axis_wider_than_its_buffer_is_refused() {
    let padded_axis = <m![A # 8]>::to_value();

    assert_eq!(
        <m![A, H]>::to_value().find_axis(&padded_axis),
        Err(FindAxisError::SpanExceedsMapping)
    );
    assert_eq!(
        <m![A]>::to_value().find_axis(&padded_axis),
        Err(FindAxisError::SpanExceedsMapping)
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
fn test_reported_stride_satisfies_the_windowing_precondition() {
    for table in [
        <m![Z # 32, H]>::to_value(),
        <m![A, Q, D]>::to_value(),
        <m![A, B]>::to_value(),
    ] {
        let n = table.size();
        for target in (1..=n).filter(|t| n.is_multiple_of(*t)) {
            let axis = table.split_at(target).0;
            let Ok(stride) = table.find_axis(&axis) else { continue };
            let size = axis.size();

            let peeled = table.split_at(stride * size).1.split_at(stride).0;

            assert_eq!(
                peeled.normalize(),
                axis.normalize(),
                "stride {stride} x {size} does not peel back to {} in {}",
                axis.normalize(),
                table.normalize()
            );
        }
    }
}

/// Sweeps the single-run rule over every subset of a buffer's terms: an axis is located exactly when
/// its terms are adjacent, and each located one peels back out at the reported stride.
#[test]
fn test_only_adjacent_terms_form_an_axis() {
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
        let peeled = table.split_at(stride * axis.size()).1.split_at(stride).0;
        assert_eq!(
            peeled.normalize(),
            axis.normalize(),
            "stride {stride} does not peel back to {}",
            axis.normalize()
        );
    }
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
        Err(FindAxisError::SpanExceedsMapping),
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

/// Desired behavior, not met for a hole away from the tail: a `Top` axis is located in a
/// `Bottom`-padded mapping, because the carve splits the segment before comparing its padding.
#[test]
#[ignore = "the carve splits a segment before its padding is compared (backlog)"]
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
fn test_doubly_divided_composites_are_located() {
    let cases: [(Mapping, usize); 3] = [
        (<m![A, B]>::to_value(), 4),
        (<m![A, B]>::to_value(), 8),
        (<m![P # 24, H]>::to_value(), 6),
    ];

    for (table, target) in cases {
        let axis = table.split_at(target).0;
        let size = axis.size();

        // The axis is present: peeling at the step it would have reproduces it.
        let peeled = table.split_at(target * size).1.split_at(target).0;
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
}
