/// NaN-aware f32 wrapper. NaN == NaN is true.
#[derive(Clone, Copy)]
pub struct F32(pub f32);

impl PartialEq for F32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}

impl std::fmt::Debug for F32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// NaN-aware equality check for f32 tensors via to_buf().
pub fn assert_f32_vec_eq(expected: &[f32], actual: &[f32]) {
    let expected_vec: Vec<F32> = expected.iter().copied().map(F32).collect();
    let actual_vec: Vec<F32> = actual.iter().copied().map(F32).collect();
    assert_eq!(expected_vec, actual_vec);
}
