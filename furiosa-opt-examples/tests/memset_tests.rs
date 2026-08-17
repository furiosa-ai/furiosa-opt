//! Answer-key tests for the `memset` example kernels on the CPU backend.
//!
//! These pin the VISA-level *calculation*: each kernel is launched on the CPU and its readback is
//! checked against a hand-written answer key. (VISA-vs-LIR *parity*, that device translation matches
//! this calculation, lives in `npu-visa-test`.) Every kernel overwrites a **non-fill** input, so a
//! readback equal to the fill value proves the fill fired (a passthrough would echo the input). The
//! input is `m![A, B]`; the 2-cluster relayout reads it back as `m![A / 2, A % 2, B]` (`OutMap`).

use furiosa_opt_examples::memset::{
    A, B, memset_alias_bf16, memset_bf16_one, memset_f8e4m3_one, memset_f32_one_half, memset_i4_neg_one,
    memset_i8_neg_one, memset_i16_300, memset_i32_zero,
};
use furiosa_opt_std::prelude::*;

type OutMap = m![A / 2, A % 2, B];

/// Launches a whole-region `memset` kernel on the CPU over a non-fill `$input` and asserts every
/// readback cell equals `$fill`: the answer key for a fill that overwrites the whole region.
macro_rules! whole_region_answer_test {
    ($name:ident, $d:ty, $fn:ident, $input:expr, $fill:expr $(,)?) => {
        #[tokio::test]
        async fn $name() {
            let mut ctx = Context::acquire();
            let input_vals: Vec<$d> = $input;
            let input = HostTensor::<$d, m![A, B]>::from_vec(input_vals)
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;
            let output = launch($fn, (&mut *ctx, &input)).await;
            let answer: Vec<$d> = std::iter::repeat_n($fill, <OutMap>::SIZE).collect();
            assert_eq!(output.to_host::<OutMap>(&mut ctx.pdma).await.into_vec(), answer);
        }
    };
}

// `PadValue::Zero` path (a zero fill folds to all-zero element bits).
whole_region_answer_test!(
    test_memset_i32_zero,
    i32,
    memset_i32_zero,
    (1..=<m![A, B]>::SIZE as i32).collect(),
    0,
);
// Signed `PadValue::Custom` (low byte `0xff`).
whole_region_answer_test!(
    test_memset_i8_neg_one,
    i8,
    memset_i8_neg_one,
    (0..<m![A, B]>::SIZE).map(|i| (1 + (i % 50)) as i8).collect(),
    -1,
);
// 2-byte `PadValue::Custom`.
whole_region_answer_test!(
    test_memset_i16_300,
    i16,
    memset_i16_300,
    (0..<m![A, B]>::SIZE).map(|i| (i % 200) as i16).collect(),
    300,
);
// 4-byte float `PadValue::Custom`.
whole_region_answer_test!(
    test_memset_f32_one_half,
    f32,
    memset_f32_one_half,
    (0..<m![A, B]>::SIZE).map(|i| 2.0 + i as f32).collect(),
    1.5,
);
// `const { .. }`-folded `bf16`.
whole_region_answer_test!(
    test_memset_bf16_one,
    bf16,
    memset_bf16_one,
    (0..<m![A, B]>::SIZE).map(|i| bf16::from_f32(2.0 + i as f32)).collect(),
    bf16::from_f32(1.0),
);
// `const { .. }`-folded `f8e4m3`.
whole_region_answer_test!(
    test_memset_f8e4m3_one,
    f8e4m3,
    memset_f8e4m3_one,
    (0..<m![A, B]>::SIZE).map(|_| f8e4m3::from_f32(2.0)).collect(),
    f8e4m3::from_f32(1.0),
);
// Sub-byte `i4`: the fill value `-1` is the nibble `0xf`, so every cell decodes to -1.
whole_region_answer_test!(
    test_memset_i4_neg_one,
    i4,
    memset_i4_neg_one,
    (0..<m![A, B]>::SIZE).map(|i| i4::from_i32((i % 7) as i32)).collect(),
    i4::from_i32(-1),
);
// In-place aliasing: the relayout consumes the fill, so the readback is all-`3.0`.
whole_region_answer_test!(
    test_memset_alias_bf16,
    bf16,
    memset_alias_bf16,
    (0..<m![A, B]>::SIZE).map(|_| bf16::from_f32(1.0)).collect(),
    bf16::from_f32(3.0),
);
