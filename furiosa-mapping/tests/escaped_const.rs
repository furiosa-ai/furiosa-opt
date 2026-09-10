use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, i, m};

/// `m!` accepts an escaped constant `{ EXPR }` wherever a literal number is expected — the
/// numeric operator arguments `/`, `%`, `=`, and padding `#` (including `#{!}` / `#{0}`).
/// Padding may also name an axis; the escaped form must agree with the literal it resolves to.
#[test]
fn test_m_macro_escaped_const_number() {
    axes![A = 512, Out = 64];
    const N: usize = 8;

    // Padding `#`: an associated const (`Out::SIZE`) and a plain const item (`N`) both work.
    assert_eq!(<m![1 # { Out::SIZE }]>::to_value(), <m![1 # 64]>::to_value());
    assert_eq!(<m![A # { N }]>::to_value(), <m![A # 8]>::to_value());

    // Stride `/`, modulo `%`, resize `=`.
    assert_eq!(<m![A / { N }]>::to_value(), <m![A / 8]>::to_value());
    assert_eq!(<m![A % { N }]>::to_value(), <m![A % 8]>::to_value());
    assert_eq!(<m![A = { N }]>::to_value(), <m![A = 8]>::to_value());

    // Fill-kind padding (`#{!}`, `#{0}`) with an escaped const size.
    assert_eq!(<m![A #{!} { Out::SIZE }]>::to_value(), <m![A #{!} 64]>::to_value());
    assert_eq!(<m![A #{0} { N }]>::to_value(), <m![A #{0} 8]>::to_value());

    // A bare `{ .. }` in atom position is still an escaped *mapping*, unaffected.
    assert_eq!(<m![{ m![A] }]>::to_value(), <m![A]>::to_value());
}

/// A padding extent may be an axis, keeping a size derived by the caller in type space instead
/// of reading an associated const in const-argument position.
#[test]
fn test_m_macro_axis_padding_extent() {
    axes![Tiles = 8];

    assert_eq!(<m![1 # Tiles]>::to_value(), <m![1 # 8]>::to_value());
    assert_eq!(<m![1 #{!} Tiles]>::to_value(), <m![1 #{!} 8]>::to_value());
    assert_eq!(<m![1 #{0} Tiles]>::to_value(), <m![1 #{0} 8]>::to_value());
}

fn extent_from_axis<Factor: AxisName>() -> Mapping {
    <m![1 # Factor]>::to_value()
}

/// The factor stays in type space when it comes from a generic caller, so this crate needs no
/// `generic_const_exprs` feature to express it.
#[test]
fn test_m_macro_generic_axis_padding_extent() {
    axes![Tiles = 8];

    assert_eq!(extent_from_axis::<Tiles>(), <m![1 # 8]>::to_value());
}

fn mappings_from_axes<Width: AxisName, Factor: AxisName>() -> (Mapping, Mapping, Mapping, Mapping) {
    (
        <m![Width / Factor]>::to_value(),
        <m![Width % Factor]>::to_value(),
        <m![Width = Factor]>::to_value(),
        <m![1 # Factor]>::to_value(),
    )
}

/// Every mapping factor may be an axis in a generic caller, so no operator has to read an
/// associated const in const-argument position.
#[test]
fn test_m_macro_generic_axis_factors() {
    axes![Width = 64, Factor = 8];

    assert_eq!(
        mappings_from_axes::<Width, Factor>(),
        (
            <m![Width / 8]>::to_value(),
            <m![Width % 8]>::to_value(),
            <m![Width = 8]>::to_value(),
            <m![1 # 8]>::to_value(),
        )
    );
}

fn factors_derived_from_width<Width: AxisName>() -> (Mapping, Mapping) {
    (
        <m![1 # (Width / 512)]>::to_value(),
        <m![Width / (Width / 256)]>::to_value(),
    )
}

/// A factor may be a parenthesized mapping, so a caller that tiles a generic width derives the
/// tile count from the width instead of being handed it.
#[test]
fn test_m_macro_mapping_factors() {
    axes![Width = 2560];

    assert_eq!(
        factors_derived_from_width::<Width>(),
        (<m![1 # 5]>::to_value(), <m![Width / 10]>::to_value()),
    );
}

/// `i!` accepts escaped constants in the same operator positions.
#[test]
fn test_i_macro_escaped_const_number() {
    axes![A = 512];
    const N: usize = 32;

    assert_eq!(i![A / { N } = 8: 0], i![A / 32 = 8: 0]);
    assert_eq!(i![A % { N }: 3], i![A % 32: 3]);
}
