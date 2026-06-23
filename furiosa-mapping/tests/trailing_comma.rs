use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, i, m};

/// `axes![]` accepts trailing commas in any position.
#[test]
fn test_axes_macro_trailing_comma_accepted() {
    // Single-element with trailing comma.
    {
        axes![A = 4,];
        assert_eq!(A::SIZE, 4);
    }
    // Multi-element with trailing comma.
    {
        axes![A = 4, B = 2,];
        assert_eq!((A::SIZE, B::SIZE), (4, 2));
    }
}

/// `m![]` accepts trailing commas in any position.
#[test]
fn test_m_macro_trailing_comma_accepted() {
    axes![A = 4, B = 2];

    // Single-element with trailing comma.
    assert_eq!(<m![A,]>::to_value(), <m![A]>::to_value());
    // Multi-element with trailing comma.
    assert_eq!(<m![A, B,]>::to_value(), <m![A, B]>::to_value());
    // Nested with trailing comma.
    assert_eq!(<m![(A, B,)]>::to_value(), <m![A, B]>::to_value());
}

/// `i![]` accepts trailing commas in any position.
#[test]
fn test_i_macro_trailing_comma_accepted() {
    axes![A = 4, B = 2];

    // Index assignments with trailing comma.
    assert_eq!(i![A: 1, B: 0,], i![A: 1, B: 0]);
    // Nested index mapping with trailing comma.
    assert_eq!(i![(A, B,): 5,], i![A, B: 5]);
}
