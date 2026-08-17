use crate::common::{assert_f32_bits_eq, assert_f32_vec_eq};
use furiosa_opt_examples::negative::vector_engine::ve_elementwise_fxp_chain;
use furiosa_opt_examples::vector_engine::{
    A, B, ve_elementwise_full_pipeline, ve_elementwise_fxp_const, ve_elementwise_logic, ve_elementwise_multi_vrf,
    ve_elementwise_reinterpret_abs_f32, ve_elementwise_reinterpret_chain_f32, ve_elementwise_stash_f32,
    ve_elementwise_stash_i32, ve_elementwise_ternary, ve_elementwise_ternary_stash, ve_elementwise_vrf,
    ve_stash_after_reinterpret_f32, ve_stash_after_widen_f32, ve_stash_fxp_fxp,
};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::panic::{AssertUnwindSafe, catch_unwind};

// =============================================================================
// VE Elementwise Tests (uses A=512, B=256 from vector_engine.rs)
// =============================================================================

#[tokio::test]
async fn test_ve_elementwise_fxp_const() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_fxp_const, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = input + 100
    let expected = input.into_inner().map(|x| x.wrapping_add(100));

    assert_eq!(expected.into_vec(), result.into_vec());
}

/// This test verifies that ALU conflicts are properly detected.
/// AddFxp and SubFxp both use FxpAdd ALU, so chaining them should panic.
#[tokio::test]
async fn test_ve_elementwise_fxp_chain() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut ctx = Context::acquire();

            let mut rng = SmallRng::seed_from_u64(42);
            let input = HostTensor::<i32, m![A]>::rand(&mut rng);

            let input_hbm = input.to_hbm(&mut ctx.pdma).await;

            launch(ve_elementwise_fxp_chain, (&mut *ctx, &input_hbm)).await;
        });
    }));

    assert!(
        result.is_err(),
        "Expected panic due to ALU conflict (FxpAdd used twice)"
    );
}

#[tokio::test]
async fn test_ve_elementwise_full_pipeline() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_full_pipeline, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = clamp(((input + 100) as f32 * 2.5) as i32, 0, 1000)
    let expected = input.into_inner().map(|x| {
        let v = ((x.wrapping_add(100) as f32) * 2.5).round() as i32;
        v.clamp(0, 1000)
    });

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_elementwise_stash_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_stash_f32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = max(input * 2.0, input)
    let expected = input.into_inner().map(|x| f32::max(x * 2.0, x));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_elementwise_stash_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_stash_i32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = max(input * 2, input)
    let expected = input.into_inner().map(|x| i32::max(x.wrapping_mul(2), x));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_stash_fxp_fxp() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_stash_fxp_fxp, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = max(input * 2, input)
    let expected = input.into_inner().map(|x| i32::max(x.wrapping_mul(2), x));

    assert_eq!(expected.into_vec(), result.into_vec());
}

/// Corner cases for the reinterpret kernels: both signed zeros (the sign bit the mask must clear),
/// the smallest normal (where an exponent bump is not a doubling of a normal value), values the
/// clip cuts and values it does not.
const REINTERPRET_CORNERS: [f32; 9] = [0.0, -0.0, -1.0, 1.0, -0.5, 3.25, -80.0, 120.0, f32::MIN_POSITIVE];

/// `min(|x| with exponent + 1, 100.0)` per [`REINTERPRET_CORNERS`] entry, worked out from the bit
/// layout rather than from the kernel's own arithmetic: on `±0.0` the exponent field goes 0 -> 1, so
/// the result is `f32::MIN_POSITIVE` and not `0.0`, and the last entry doubles into
/// `2 * f32::MIN_POSITIVE`.
const REINTERPRET_CHAIN_EXPECTED: [f32; 9] = [
    f32::MIN_POSITIVE,
    f32::MIN_POSITIVE,
    2.0,
    2.0,
    1.0,
    6.5,
    100.0,
    100.0,
    2.0 * f32::MIN_POSITIVE,
];

fn cycled(values: &[f32]) -> Vec<f32> {
    (0..A::SIZE).map(|i| values[i % values.len()]).collect()
}

#[tokio::test]
async fn test_ve_elementwise_reinterpret_abs_f32() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<f32, m![A]>::from_vec(cycled(&REINTERPRET_CORNERS));
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_reinterpret_abs_f32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Clearing the sign bit is `abs`, reached here without a float ALU. Compared bitwise, so `-0.0`
    // reaching the output would fail rather than compare equal to `0.0`.
    let expected = input.into_inner().map(f32::abs);

    assert_f32_bits_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_elementwise_reinterpret_chain_f32() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<f32, m![A]>::from_vec(cycled(&REINTERPRET_CORNERS));
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_reinterpret_chain_f32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    assert_f32_bits_eq(&cycled(&REINTERPRET_CHAIN_EXPECTED), &result.into_vec());
}

/// The stash takes what the stream carries at the write, so the masked `|x|` is what the clip reads
/// back: `2|x|`. Bitwise, so a `-0.0` surviving the mask would fail.
#[tokio::test]
async fn test_ve_stash_after_reinterpret_f32() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<f32, m![A]>::from_vec(cycled(&REINTERPRET_CORNERS));
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_stash_after_reinterpret_f32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    let expected = input.into_inner().map(|x| 2.0 * f32::abs(x));

    assert_f32_bits_eq(&expected.into_vec(), &result.into_vec());
}

/// A 4-way result reaches the stash through the widen: the write rides the first clip op and the
/// read the second, so both sit on 8-way ALUs.
#[tokio::test]
async fn test_ve_stash_after_widen_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_stash_after_widen_f32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = 2 * max(input * 2, 0)
    let expected = input.into_inner().map(|x| 2.0 * f32::max(x * 2.0, 0.0));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_elementwise_logic() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_logic, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = (input & 0xFF) | 0x100
    let expected = input.into_inner().map(|x| (x & 0xFF) | 0x100);

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
#[ignore = "Failing on cpu"]
async fn test_ve_elementwise_vrf() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![B]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_vrf, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A, B]>(&mut ctx.pdma).await;

    // Verify: output = lhs + rhs (broadcasted)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner().transpose(true), |x, y| x + y);

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_elementwise_multi_vrf() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A, B]>::rand(&mut rng);
    let vrf1 = HostTensor::<i32, m![B]>::rand(&mut rng);
    let vrf2 = HostTensor::<i32, m![B]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;
    let vrf1_hbm = vrf1.to_hbm(&mut ctx.pdma).await;
    let vrf2_hbm = vrf2.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_multi_vrf, (&mut *ctx, &input_hbm, &vrf1_hbm, &vrf2_hbm)).await;

    let result = out_hbm.to_host::<m![A, B]>(&mut ctx.pdma).await;

    // Verify: output = ((input + vrf1) * vrf2) + vrf1 (broadcasted)
    let vrf1_inner = vrf1.into_inner();
    let vrf1_t = vrf1_inner.transpose(true);
    let vrf2_t = vrf2.into_inner().transpose(true);
    let expected = input
        .into_inner()
        .zip_with(&vrf1_t, |x, y| x.wrapping_add(y))
        .zip_with(&vrf2_t, |x, y| x.wrapping_mul(y))
        .zip_with(&vrf1_t, |x, y| x.wrapping_add(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

// =============================================================================
// Ternary operation tests
// =============================================================================

#[tokio::test]
async fn test_ve_elementwise_ternary() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_ternary, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: FmaF = input * 2.0 + 3.0
    let expected = input.into_inner().map(|x| x.mul_add(2.0, 3.0));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_elementwise_ternary_stash() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out_hbm = launch(ve_elementwise_ternary_stash, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: FmaF with stash = input * input + 1.0 = input^2 + 1.0
    let expected = input.into_inner().map(|x| x.mul_add(x, 1.0));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}
