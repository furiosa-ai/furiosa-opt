use crate::common::{assert_f32_bits_eq, assert_f32_vec_eq};
use furiosa_opt_examples::vector_engine::{
    A, ve_group_pair_add, ve_group_pair_chain, ve_group_pair_fp, ve_group_pair_fxp, ve_group_pair_logic,
    ve_group_pair_logic_abs_add_f32, ve_group_pair_preprocess_both, ve_group_pair_preprocess_g0,
    ve_group_pair_preprocess_g1, ve_group_pair_reinterpret_scale_f32, ve_group_pair_ternary,
    ve_group_pair_ternary_selective, ve_group_pair_unary, ve_group_pair_unary_selective,
};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// =============================================================================
// VE Binary Split Tests (uses A=512 from vector_engine.rs)
// All binary_split functions take (device, lhs, rhs, output)
// =============================================================================

#[tokio::test]
async fn test_ve_group_pair_add() {
    let mut device = Device::new(ve_group_pair_add.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_add, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = lhs + rhs
    let expected = lhs.into_inner().zip_with(&rhs.into_inner(), |x, y| x.wrapping_add(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_preprocess_both() {
    let mut device = Device::new(ve_group_pair_preprocess_both.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_preprocess_both, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = (lhs * 2) + (rhs * 3)
    let expected = lhs
        .into_inner()
        .map(|x| x.wrapping_mul(2))
        .zip_with(&rhs.into_inner().map(|x| x.wrapping_mul(3)), |x, y| x.wrapping_add(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_preprocess_g0() {
    let mut device = Device::new(ve_group_pair_preprocess_g0.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_preprocess_g0, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = (lhs * 10) + rhs
    let expected = lhs
        .into_inner()
        .map(|x| x.wrapping_mul(10))
        .zip_with(&rhs.into_inner(), |x, y| x.wrapping_add(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_preprocess_g1() {
    let mut device = Device::new(ve_group_pair_preprocess_g1.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_preprocess_g1, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = lhs + (rhs * 10)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner().map(|x| x.wrapping_mul(10)), |x, y| x.wrapping_add(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_chain() {
    let mut device = Device::new(ve_group_pair_chain.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_chain, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = ((lhs + 10) * 2) + ((rhs + 20) * 3)
    let lhs_processed = lhs.into_inner().map(|x| x.wrapping_add(10).wrapping_mul(2));
    let rhs_processed = rhs.into_inner().map(|x| x.wrapping_add(20).wrapping_mul(3));
    let expected = lhs_processed.zip_with(&rhs_processed, |x, y| x.wrapping_add(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_fxp() {
    let mut device = Device::new(ve_group_pair_fxp.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_fxp, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = lhs * rhs
    let expected = lhs.into_inner().zip_with(&rhs.into_inner(), |x, y| x.wrapping_mul(y));

    assert_eq!(expected.into_vec(), result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_logic() {
    let mut device = Device::new(ve_group_pair_logic.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_logic, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: output = lhs ^ rhs (BitXor)
    let expected = lhs.into_inner().zip_with(&rhs.into_inner(), |x, y| x ^ y);

    assert_eq!(expected.into_vec(), result.into_vec());
}

/// `(lhs, rhs)` pairs and the `scale(lhs, 1) + scale(rhs, 2)` the pair kernel must produce, where
/// `scale(x, n)` adds `n` to the exponent field. Worked out from the bit layout, not from the
/// kernel's arithmetic. The signed-zero pairs are the reason the two groups are paired this way: a
/// zero scales into `±f32::MIN_POSITIVE` rather than staying `0.0`, and pairing it against another
/// zero keeps that visible in the sum instead of being swamped by a normal-sized term.
const PAIR_SCALE_CASES: [(f32, f32, f32); 4] = [
    (0.0, -0.0, -f32::MIN_POSITIVE), // MIN_POSITIVE + (-2 * MIN_POSITIVE)
    (-0.0, 0.0, f32::MIN_POSITIVE),  // -MIN_POSITIVE + 2 * MIN_POSITIVE
    (1.0, 1.0, 6.0),                 // 2.0 + 4.0
    (-1.0, -0.5, -4.0),              // -2.0 + (-2.0)
];

#[tokio::test]
async fn test_ve_group_pair_reinterpret_scale_f32() {
    let mut device = Device::new(ve_group_pair_reinterpret_scale_f32.topology()).unwrap();

    let pick = |f: fn(&(f32, f32, f32)) -> f32| {
        (0..A::SIZE)
            .map(|i| f(&PAIR_SCALE_CASES[i % PAIR_SCALE_CASES.len()]))
            .collect::<Vec<_>>()
    };
    let lhs = HostTensor::<f32, m![A]>::from_vec(pick(|case| case.0));
    let rhs = HostTensor::<f32, m![A]>::from_vec(pick(|case| case.1));

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_reinterpret_scale_f32, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    assert_f32_bits_eq(&pick(|case| case.2), &result.into_vec());
}

/// `|a| + |b|` with the sign bit cleared per group. Bitwise, so a `-0.0` surviving the mask fails,
/// and `-0.0 + 0.0` must come out `0.0`.
#[tokio::test]
async fn test_ve_group_pair_logic_abs_add_f32() {
    let mut device = Device::new(ve_group_pair_logic_abs_add_f32.topology()).unwrap();

    let cases: [(f32, f32); 5] = [
        (0.0, -0.0),
        (-0.0, -0.0),
        (-1.0, 2.5),
        (3.25, -80.0),
        (-f32::MIN_POSITIVE, 1.0),
    ];
    let pick = |f: fn(&(f32, f32)) -> f32| (0..A::SIZE).map(|i| f(&cases[i % cases.len()])).collect::<Vec<_>>();
    let lhs = HostTensor::<f32, m![A]>::from_vec(pick(|case| case.0));
    let rhs = HostTensor::<f32, m![A]>::from_vec(pick(|case| case.1));

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_logic_abs_add_f32, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    let expected: Vec<f32> = (0..A::SIZE)
        .map(|i| {
            let (a, b) = cases[i % cases.len()];
            f32::abs(a) + f32::abs(b)
        })
        .collect();

    assert_f32_bits_eq(&expected, &result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_fp() {
    let mut device = Device::new(ve_group_pair_fp.topology()).unwrap();

    // Input is i32, output is f32
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_fp, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: fxp_to_fp(31) then multiply -> (lhs as f32) * (rhs as f32)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |x, y| (x as f32) * (y as f32));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_unary() {
    let mut device = Device::new(ve_group_pair_unary.topology()).unwrap();

    // Input is i32, output is f32
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_unary, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: fxp_to_fp(31) -> sqrt(both) -> add
    // output = sqrt(lhs as f32) + sqrt(rhs as f32)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |x, y| (x as f32).sqrt() + (y as f32).sqrt());

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_unary_selective() {
    let mut device = Device::new(ve_group_pair_unary_selective.topology()).unwrap();

    // Input is i32, output is f32
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_unary_selective, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: fxp_to_fp(31) -> exp(g0 only) -> add
    // output = exp(lhs as f32) + (rhs as f32)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |x, y| (x as f32).exp() + (y as f32));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_ternary() {
    let mut device = Device::new(ve_group_pair_ternary.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<f32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<f32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_ternary, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: (lhs * 2.0 + 1.0) + (rhs * 3.0 + 2.0)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |x, y| x.mul_add(2.0, 1.0) * y.mul_add(3.0, 2.0));

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}

#[tokio::test]
async fn test_ve_group_pair_ternary_selective() {
    let mut device = Device::new(ve_group_pair_ternary_selective.topology()).unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<f32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<f32, m![A]>::rand(&mut rng);

    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out_hbm = launch(ve_group_pair_ternary_selective, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let result = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap();

    // Verify: (lhs * 2.0 + 1.0) / rhs (rhs unchanged, no ternary)
    let expected = lhs
        .into_inner()
        .zip_with(&rhs.into_inner(), |x, y| x.mul_add(2.0, 1.0) / y);

    assert_f32_vec_eq(&expected.into_vec(), &result.into_vec());
}
