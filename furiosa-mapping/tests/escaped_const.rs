use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, i, m};

/// `m!` accepts an escaped constant `{ EXPR }` wherever a literal number is expected — the
/// operator arguments `/`, `%`, `=`, and `#` (including `#{!}` / `#{0}`). The escaped form must
/// agree with the literal it resolves to.
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

/// `i!` accepts escaped constants in the same operator positions.
#[test]
fn test_i_macro_escaped_const_number() {
    axes![A = 512];
    const N: usize = 32;

    assert_eq!(i![A / { N } = 8: 0], i![A / 32 = 8: 0]);
    assert_eq!(i![A % { N }: 3], i![A % 32: 3]);
}
