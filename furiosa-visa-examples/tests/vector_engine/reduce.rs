use crate::common::assert_f32_vec_eq;
use furiosa_visa_examples::vector_engine::{
    A, R, ve_inter_slice_reduce_add_f32, ve_inter_slice_reduce_add_sat_i32, ve_inter_slice_reduce_max_i32,
    ve_intra_slice_reduce_add_f32, ve_intra_slice_reduce_add_fxp_sat, ve_intra_slice_reduce_max_f32,
    ve_intra_slice_reduce_max_i32, ve_intra_slice_reduce_min_f32, ve_intra_slice_reduce_min_i32,
    ve_intra_slice_reduce_split_time_packet, ve_vru_then_vau_i32,
};
use furiosa_visa_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// =============================================================================
// Intra-Slice Reduce Tests (uses A=512, R=4, S=15 from vector_engine.rs)
// =============================================================================

#[tokio::test]
async fn test_ve_intra_slice_reduce_add_fxp_sat() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A, R]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_add_fxp_sat, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: saturating add across R axis
    let expected: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.saturating_add(y)), Opt::Init(0));
    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_intra_slice_reduce_max_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A, R]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_max_i32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: max across R axis
    let expected: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.max(y)), Opt::Init(i32::MIN));
    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_intra_slice_reduce_min_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A, R]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_min_i32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: min across R axis
    let expected: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.min(y)), Opt::Init(i32::MAX));
    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_intra_slice_reduce_add_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A, R]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_add_f32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: sum across R axis
    let expected: Tensor<f32, m![A]> = input.into_inner().reduce_add();
    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_intra_slice_reduce_max_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A, R]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_max_f32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: max across R axis
    let expected: Tensor<f32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.max(y)), Opt::Init(f32::NEG_INFINITY));
    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_intra_slice_reduce_min_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![A, R]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_min_f32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: min across R axis
    let expected: Tensor<f32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.min(y)), Opt::Init(f32::INFINITY));
    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

#[tokio::test]
async fn test_ve_intra_slice_reduce_split_time_packet() {
    use furiosa_visa_examples::vector_engine::R16 as R;

    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![R, A]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_intra_slice_reduce_split_time_packet, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Verify: saturating add across R axis (R=4, no padding, full reduce)
    let expected: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.saturating_add(y)), Opt::Init(0));
    assert_eq!(expected.to_buf(), result.to_buf());
}

// =============================================================================
// Inter-Slice Reduce Tests (VRU) — uses A=512, R=4 from vector_engine.rs
// =============================================================================

#[tokio::test]
async fn test_ve_inter_slice_reduce_add_sat_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![R, A]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_inter_slice_reduce_add_sat_i32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    let expected: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.saturating_add(y)), Opt::Init(0));
    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_inter_slice_reduce_max_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![R, A]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_inter_slice_reduce_max_i32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    let expected: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.max(y)), Opt::Init(i32::MIN));
    assert_eq!(expected.to_buf(), result.to_buf());
}

#[tokio::test]
async fn test_ve_inter_slice_reduce_add_f32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![R, A]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_inter_slice_reduce_add_f32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    let expected: Tensor<f32, m![A]> = input.into_inner().reduce_add();
    assert_f32_vec_eq(&expected.to_buf(), &result.to_buf());
}

// =============================================================================
// Path 4: VRU → VAU Test
// =============================================================================

#[tokio::test]
async fn test_ve_vru_then_vau_i32() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![R, A]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0 << 28).await;

    let out_hbm = launch(ve_vru_then_vau_i32, (&mut *ctx, &input_hbm)).await;
    let result = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await;

    // Expected: saturating_add across R, then +100 per element
    let reduced: Tensor<i32, m![A]> = input
        .into_inner()
        .reduce(|a, b| a.zip_map(b, |x, y| x.saturating_add(y)), Opt::Init(0));
    let expected: Vec<_> = reduced
        .to_buf()
        .into_iter()
        .map(|v| v.map(|x| x.wrapping_add(100)))
        .collect();
    assert_eq!(expected, result.to_buf());
}

// VCG-required reduce examples (ve_vcg_intra_*) were removed: the example mappings
// I added did not satisfy the Slice=256 hardware constraint or the divide_strict
// invariants. Rewriting them properly requires careful kernel design, deferred to #18.
