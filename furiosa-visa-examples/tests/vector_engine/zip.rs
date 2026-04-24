use crate::common::assert_f32_vec_eq;
use furiosa_visa_examples::vector_engine::{
    A, ve_group_pair_add, ve_group_pair_chain, ve_group_pair_fp, ve_group_pair_fxp, ve_group_pair_logic,
    ve_group_pair_preprocess_both, ve_group_pair_preprocess_g0, ve_group_pair_preprocess_g1, ve_group_pair_ternary,
    ve_group_pair_ternary_selective, ve_group_pair_unary, ve_group_pair_unary_selective,
};
use furiosa_visa_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// =============================================================================
// VE Binary Split Tests (uses A=512 from vector_engine.rs)
// All binary_split functions take (ctx, lhs, rhs, output)
// =============================================================================

#[tokio::test]
async fn test_ve_group_pair_add() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_add, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = lhs + rhs
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |a, b| a.zip_map(b, |x, y| x.wrapping_add(y)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_preprocess_both() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_preprocess_both, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = (lhs * 2) + (rhs * 3)
    let expected = lhs
        .into_inner()
        .map(|x| x.map(|x| x.wrapping_mul(2)))
        .zip_with(&rhs.into_inner().map(|x| x.map(|x| x.wrapping_mul(3))), |a, b| {
            a.zip_map(b, |x, y| x.wrapping_add(y))
        });

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_preprocess_g0() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_preprocess_g0, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = (lhs * 10) + rhs
    let expected = lhs
        .into_inner()
        .map(|x| x.map(|x| x.wrapping_mul(10)))
        .zip_with(&rhs.into_inner(), |a, b| a.zip_map(b, |x, y| x.wrapping_add(y)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_preprocess_g1() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_preprocess_g1, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = lhs + (rhs * 10)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner().map(|x| x.map(|x| x.wrapping_mul(10))), |a, b| {
            a.zip_map(b, |x, y| x.wrapping_add(y))
        });

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_chain() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_chain, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = ((lhs + 10) * 2) + ((rhs + 20) * 3)
    let lhs_processed = lhs.into_inner().map(|x| x.map(|x| x.wrapping_add(10).wrapping_mul(2)));
    let rhs_processed = rhs.into_inner().map(|x| x.map(|x| x.wrapping_add(20).wrapping_mul(3)));
    let expected = lhs_processed.zip_with(&rhs_processed, |a, b| a.zip_map(b, |x, y| x.wrapping_add(y)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_fxp() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_fxp, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = lhs * rhs
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |a, b| a.zip_map(b, |x, y| x.wrapping_mul(y)));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_logic() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_logic, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: output = lhs ^ rhs (BitXor)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |a, b| a.zip_map(b, |x, y| x ^ y));

    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_fp() {
    let mut ctx = Context::acquire();

    // Input is i32, output is f32
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_fp, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: fxp_to_fp(31) then multiply -> (lhs as f32) * (rhs as f32)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |a, b| a.zip_map(b, |x, y| (x as f32) * (y as f32)));

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_unary() {
    let mut ctx = Context::acquire();

    // Input is i32, output is f32
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_unary, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: fxp_to_fp(31) -> sqrt(both) -> add
    // output = sqrt(lhs as f32) + sqrt(rhs as f32)
    let expected = lhs.into_inner().zip_with(&rhs.into_inner(), |a, b| {
        a.zip_map(b, |x, y| (x as f32).sqrt() + (y as f32).sqrt())
    });

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_unary_selective() {
    let mut ctx = Context::acquire();

    // Input is i32, output is f32
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_unary_selective, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: fxp_to_fp(31) -> exp(g0 only) -> add
    // output = exp(lhs as f32) + (rhs as f32)
    let expected = lhs.into_inner().zip_with(&rhs.into_inner(), |a, b| {
        a.zip_map(b, |x, y| (x as f32).exp() + (y as f32))
    });

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_ternary() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<f32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<f32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_ternary, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: (lhs * 2.0 + 1.0) + (rhs * 3.0 + 2.0)
    let expected = lhs.into_inner().zip_with(&rhs.into_inner(), |a, b| {
        a.zip_map(b, |x, y| x.mul_add(2.0, 1.0) * y.mul_add(3.0, 2.0))
    });

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_group_pair_ternary_selective() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<f32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<f32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out_hbm = launch(ve_group_pair_ternary_selective, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: (lhs * 2.0 + 1.0) / rhs (rhs unchanged, no ternary)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |a, b| a.zip_map(b, |x, y| x.mul_add(2.0, 1.0) / y));

    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}
