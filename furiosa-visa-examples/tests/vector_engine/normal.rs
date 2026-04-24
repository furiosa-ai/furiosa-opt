use crate::common::assert_f32_vec_eq;
use furiosa_visa_examples::vector_engine::{
    A, B, ve_elementwise_full_pipeline, ve_elementwise_fxp_chain, ve_elementwise_fxp_const, ve_elementwise_logic,
    ve_elementwise_multi_vrf, ve_elementwise_stash_f32, ve_elementwise_stash_i32, ve_elementwise_ternary,
    ve_elementwise_ternary_stash, ve_elementwise_vrf, ve_stash_fp_fp, ve_stash_fp_fxp, ve_stash_fxp_fp,
    ve_stash_fxp_fxp,
};
use furiosa_visa_examples::vrf_add::vrf_add;
use furiosa_visa_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::panic::{AssertUnwindSafe, catch_unwind};

// =============================================================================
// VRF Add Tests (uses A=1024, B=512 from vrf_add.rs)
// =============================================================================

#[tokio::test]
async fn test_vrf_add() {
    use furiosa_visa_examples::vrf_add::{A, B};

    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![B]>::rand(&mut rng);

    // Transfer to HBM
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    // Execute the operation
    let out_hbm = launch(vrf_add, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    // copy results into host
    let result = out_hbm.to_host::<m![A, B]>(&mut ctx.pdma).await;

    // Verify: each element should be lhs[i] + rhs[i]
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner().transpose(true), |a, b| a + b);

    assert_eq!(expected.to_buf(), result.to_buf());
}

// =============================================================================
// VE Elementwise Tests (uses A=512, B=256 from vector_engine.rs)
// =============================================================================

#[tokio::test]
async fn test_ve_elementwise_fxp_const() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_fxp_const, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = input + 100
    let expected = input.into_inner().map(|x| x.map(|x| x.wrapping_add(100)));

    assert_eq!(expected.to_buf(), result.to_buf());
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

            let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

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

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_full_pipeline, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = clamp(((input + 100) as f32 * 2.5) as i32, 0, 1000)
    let expected = input.into_inner().map(|x| {
        x.map(|x| {
            let v = ((x.wrapping_add(100) as f32) * 2.5).round() as i32;
            v.clamp(0, 1000)
        })
    });

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_elementwise_stash_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_stash_f32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = max(input * 2.0, input)
    let expected = input.into_inner().map(|x| x.map(|x| f32::max(x * 2.0, x)));

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_elementwise_stash_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_stash_i32, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = max(input * 2, input)
    let expected = input.into_inner().map(|x| x.map(|x| i32::max(x.wrapping_mul(2), x)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_stash_fxp_fxp() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_stash_fxp_fxp, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = max(input * 2, input)
    let expected = input.into_inner().map(|x| x.map(|x| i32::max(x.wrapping_mul(2), x)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

/// This test verifies that cross-type stash reads (i32 -> f32) are properly rejected.
/// Stashing i32 data and trying to read it as f32 should panic with type mismatch.
#[tokio::test]
async fn test_ve_stash_fxp_fp() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut ctx = Context::acquire();

            let mut rng = SmallRng::seed_from_u64(42);
            let input = HostTensor::<i32, m![A]>::rand(&mut rng);

            let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

            launch(ve_stash_fxp_fp, (&mut *ctx, &input_hbm)).await;
        });
    }));

    assert!(
        result.is_err(),
        "Expected panic due to stash type mismatch (stashed i32, read as f32)"
    );
}

#[tokio::test]
async fn test_ve_stash_fp_fp() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_stash_fp_fp, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: stash (input * 2.0), then * 3.0, then max with stashed
    // output = max(input * 2.0 * 3.0, input * 2.0) = max(input * 6.0, input * 2.0)
    let expected = input.into_inner().map(|x| x.map(|x| f32::max(x * 6.0, x * 2.0)));

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

/// This test verifies that cross-type stash reads (f32 -> i32) are properly rejected.
/// Stashing f32 data and trying to read it as i32 should panic with type mismatch.
#[tokio::test]
async fn test_ve_stash_fp_fxp() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut ctx = Context::acquire();

            let mut rng = SmallRng::seed_from_u64(42);
            let input = HostTensor::<f32, m![A]>::rand(&mut rng);

            let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

            launch(ve_stash_fp_fxp, (&mut *ctx, &input_hbm)).await;
        });
    }));

    assert!(
        result.is_err(),
        "Expected panic due to stash type mismatch (stashed f32, read as i32)"
    );
}

#[tokio::test]
async fn test_ve_elementwise_logic() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_logic, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = (input & 0xFF) | 0x100
    let expected = input.into_inner().map(|x| x.map(|x| (x & 0xFF) | 0x100));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_elementwise_vrf() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![B]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_elementwise_vrf, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A, B]>(&mut ctx.pdma).await;

    // Verify: output = lhs + rhs (broadcasted)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner().transpose(true), |a, b| a + b);

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_elementwise_multi_vrf() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A, B]>::rand(&mut rng);
    let vrf1 = HostTensor::<i32, m![B]>::rand(&mut rng);
    let vrf2 = HostTensor::<i32, m![B]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let vrf1_hbm = vrf1.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let vrf2_hbm = vrf2.to_hbm(&mut ctx.pdma, 2 << 28).await;

    let out_hbm = launch(ve_elementwise_multi_vrf, (&mut *ctx, &input_hbm, &vrf1_hbm, &vrf2_hbm)).await;

    let result = out_hbm.to_host::<m![A, B]>(&mut ctx.pdma).await;

    // Verify: output = ((input + vrf1) * vrf2) + vrf1 (broadcasted)
    let vrf1_inner = vrf1.into_inner();
    let vrf1_t = vrf1_inner.transpose(true);
    let vrf2_t = vrf2.into_inner().transpose(true);
    let expected = input
        .into_inner()
        .zip_with(&vrf1_t, |a, b| a.zip_map(b, |x, y| x.wrapping_add(y)))
        .zip_with(&vrf2_t, |a, b| a.zip_map(b, |x, y| x.wrapping_mul(y)))
        .zip_with(&vrf1_t, |a, b| a.zip_map(b, |x, y| x.wrapping_add(y)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

// =============================================================================
// Ternary operation tests
// =============================================================================

#[tokio::test]
async fn test_ve_elementwise_ternary() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_ternary, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: FmaF = input * 2.0 + 3.0
    let expected = input.into_inner().map(|x| x.map(|x| x.mul_add(2.0, 3.0)));

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_elementwise_ternary_stash() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_elementwise_ternary_stash, (&mut *ctx, &input_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: FmaF with stash = input * input + 1.0 = input^2 + 1.0
    let expected = input.into_inner().map(|x| x.map(|x| x.mul_add(x, 1.0)));

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}
