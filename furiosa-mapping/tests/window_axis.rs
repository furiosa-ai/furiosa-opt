//! Tests for `MappingExt::window_axis`. Each case names the precondition a refusal broke.

use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, m};

axes![Z = 24, H = 4];

#[test]
fn test_narrows_the_located_axis() {
    let table = <m![Z # 32, H]>::to_value();

    let windowed = table.window_axis(8 * H::SIZE, 4, 1, PaddingKind::Top).unwrap();

    assert_eq!(windowed, <m![1 # 4, Z # 32 % 8, H]>::to_value().normalize());
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
        Err(WindowAxisError::SpanExceedsMapping {
            stride: 32,
            size: 8,
            cells: table.size()
        })
    );
}
