//! Smoke tests and deterministic scalar-oracle tests for the transformer decoder-block
//! `#[device]` entry points. The oracle tests run on the CPU backend and cover
//! projection, chunked attention, decoder, final layer, and one connected decoder step.

#![allow(clippy::type_complexity)]

use furiosa_opt_examples::transformer::axes::*;
use furiosa_opt_examples::transformer::ops;
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn seeded_bf16(rng: &mut SmallRng, len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|_| bf16::from_f32(rng.random_range(-scale..scale)))
        .collect()
}

fn assert_close(actual: &[bf16], expected: &[f32], abs: f32, rel: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual.to_f32() - expected).abs();
        assert!(
            diff <= abs + rel * expected.abs(),
            "index {i}: actual={} expected={expected} diff={diff}",
            actual.to_f32()
        );
    }
}

fn rms(values: &[f32], weight: &[bf16], eps: f32) -> Vec<f32> {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let scale = (mean + eps).sqrt();
    values
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| value / scale * weight.to_f32())
        .collect()
}

#[tokio::test]
async fn test_embedding() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(ops::embedding, (&mut *ctx, &input, &mut out)).await;
}

#[tokio::test]
async fn test_projection() {
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_weight = HostTensor::<bf16, m![Q, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_weight = HostTensor::<bf16, m![P, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_weight = HostTensor::<bf16, m![P, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let input_rms_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_rms_weight = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_rms_weight = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let kv_offset = HostTensor::<i32, m![1]>::zero().to_hbm(&mut ctx.pdma).await;
    let cos = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let sin = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut k_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut v_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut q_out = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::projection,
        (
            &mut *ctx,
            &x,
            &q_weight,
            &k_weight,
            &v_weight,
            &input_rms_weight,
            &q_rms_weight,
            &k_rms_weight,
            &kv_offset,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            &mut q_out,
        ),
    )
    .await;
}

#[tokio::test]
async fn test_projection_matches_scalar_reference() {
    let mut rng = SmallRng::seed_from_u64(0x51_7e);
    let mut ctx = Context::acquire();
    let x = seeded_bf16(&mut rng, H::SIZE, 0.25);
    let q_weight = seeded_bf16(&mut rng, Q::SIZE * H::SIZE, 0.1);
    let k_weight = seeded_bf16(&mut rng, P::SIZE * H::SIZE, 0.1);
    let v_weight = seeded_bf16(&mut rng, P::SIZE * H::SIZE, 0.1);
    let input_rms_weight = seeded_bf16(&mut rng, H::SIZE, 1.0);
    let q_rms_weight = seeded_bf16(&mut rng, D::SIZE, 1.0);
    let k_rms_weight = seeded_bf16(&mut rng, D::SIZE, 1.0);
    let x_hbm = HostTensor::<bf16, m![H]>::from_vec(x.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let q_weight_hbm = HostTensor::<bf16, m![Q, H]>::from_vec(q_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let k_weight_hbm = HostTensor::<bf16, m![P, H]>::from_vec(k_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let v_weight_hbm = HostTensor::<bf16, m![P, H]>::from_vec(v_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let input_rms_hbm = HostTensor::<bf16, m![H]>::from_vec(input_rms_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let q_rms_hbm = HostTensor::<bf16, m![D]>::from_vec(q_rms_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let k_rms_hbm = HostTensor::<bf16, m![D]>::from_vec(k_rms_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let offset = HostTensor::<i32, m![1]>::from_vec(vec![0]).to_hbm(&mut ctx.pdma).await;
    let cos = HostTensor::<bf16, m![D]>::from_vec(vec![bf16::from_f32(1.0); D::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let sin = HostTensor::<bf16, m![D]>::from_vec(vec![bf16::from_f32(0.0); D::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut k_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut v_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut q_out = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::projection,
        (
            &mut *ctx,
            &x_hbm,
            &q_weight_hbm,
            &k_weight_hbm,
            &v_weight_hbm,
            &input_rms_hbm,
            &q_rms_hbm,
            &k_rms_hbm,
            &offset,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            &mut q_out,
        ),
    )
    .await;

    let x_f32: Vec<_> = x.iter().map(|value| value.to_f32()).collect();
    let x_norm = rms(&x_f32, &input_rms_weight, 6.25e-8);
    let mut q_expected = vec![0.0; Q::SIZE];
    let mut k_expected = vec![0.0; P::SIZE];
    let mut v_expected = vec![0.0; P::SIZE];
    for row in 0..Q::SIZE {
        q_expected[row] = q_weight[row * H::SIZE..(row + 1) * H::SIZE]
            .iter()
            .zip(&x_norm)
            .map(|(&weight, &value)| weight.to_f32() * value)
            .sum();
    }
    for row in 0..P::SIZE {
        k_expected[row] = k_weight[row * H::SIZE..(row + 1) * H::SIZE]
            .iter()
            .zip(&x_norm)
            .map(|(&weight, &value)| weight.to_f32() * value)
            .sum();
        v_expected[row] = v_weight[row * H::SIZE..(row + 1) * H::SIZE]
            .iter()
            .zip(&x_norm)
            .map(|(&weight, &value)| weight.to_f32() * value)
            .sum();
    }
    for head in 0..(N::SIZE * G::SIZE) {
        let q = rms(
            &q_expected[head * D::SIZE..(head + 1) * D::SIZE],
            &q_rms_weight,
            6.25e-8,
        );
        q_expected[head * D::SIZE..(head + 1) * D::SIZE].copy_from_slice(&q);
    }
    for head in 0..N::SIZE {
        let k = rms(
            &k_expected[head * D::SIZE..(head + 1) * D::SIZE],
            &k_rms_weight,
            6.25e-8,
        );
        k_expected[head * D::SIZE..(head + 1) * D::SIZE].copy_from_slice(&k);
    }
    assert_close(
        &q_out.to_host::<m![N, G, D]>(&mut ctx.pdma).await.into_vec(),
        &q_expected,
        0.08,
        0.08,
    );
    assert_close(
        &k_cache.to_host::<m![T, N, D]>(&mut ctx.pdma).await.into_vec()[..N::SIZE * D::SIZE],
        &k_expected,
        0.08,
        0.08,
    );
    assert_close(
        &v_cache.to_host::<m![T, N, D]>(&mut ctx.pdma).await.into_vec()[..N::SIZE * D::SIZE],
        &v_expected,
        0.08,
        0.08,
    );
}

#[tokio::test]
async fn test_attention_forward_first() {
    let mut ctx = Context::acquire();

    let q = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let k = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let v = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mask = HostTensor::<f32, m![T]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::attention_forward_first,
        (&mut *ctx, &q, &k, &v, &mask, &mut max_hbm, &mut sum_hbm, &mut out_hbm),
    )
    .await;
}

#[tokio::test]
async fn test_attention_forward() {
    let mut ctx = Context::acquire();

    let q = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let k = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let v = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mask = HostTensor::<f32, m![T]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::attention_forward,
        (&mut *ctx, &q, &k, &v, &mask, &mut max_hbm, &mut sum_hbm, &mut out_hbm),
    )
    .await;
}

/// Each `forward` must persist the max it folded, or a later chunk rescales the accumulated sum
/// and output against chunk 0's max. Takes three chunks to see: with two, the stored max and the
/// running one still agree. Only the max is asserted, since that is the state that went missing.
#[tokio::test]
async fn test_attention_forward_persists_running_max() {
    let mut ctx = Context::acquire();

    // Middle chunk's keys of 1 raise the max to `D / sqrt(D)` between two zeroed chunks, so the
    // third chunk sees `sqrt(D)` only if the second one stored it. v/mask stay zero: the max is
    // taken before masking, and nothing else is read back.
    let ones = |n| std::iter::repeat_n(bf16::from_f32(1.0), n);
    let q = HostTensor::<bf16, m![N, G, D]>::from_vec(ones(N::SIZE * G::SIZE * D::SIZE));
    let k_one = HostTensor::<bf16, m![T, N, D]>::from_vec(ones(T::SIZE * N::SIZE * D::SIZE));

    let q = q.to_hbm(&mut ctx.pdma).await;
    let k_one = k_one.to_hbm(&mut ctx.pdma).await;
    let k_zero = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let v = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mask = HostTensor::<f32, m![T]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    for (chunk, k) in [&k_zero, &k_one, &k_zero].into_iter().enumerate() {
        let args = (&mut *ctx, &q, k, &v, &mask, &mut max_hbm, &mut sum_hbm, &mut out_hbm);
        match chunk {
            0 => launch(ops::attention_forward_first, args).await,
            _ => launch(ops::attention_forward, args).await,
        }
    }

    let expected = (D::SIZE as f32).sqrt();
    for max in max_hbm.to_host::<m![N, G]>(&mut ctx.pdma).await.into_vec() {
        assert!((max - expected).abs() < 0.05, "running max {max} is not {expected}");
    }
}

#[tokio::test]
async fn test_attention_two_chunks_matches_scalar_reference() {
    let mut rng = SmallRng::seed_from_u64(0x00a7_7e17);
    let mut ctx = Context::acquire();
    let q = seeded_bf16(&mut rng, N::SIZE * G::SIZE * D::SIZE, 0.2);
    let k0 = seeded_bf16(&mut rng, T::SIZE * N::SIZE * D::SIZE, 0.2);
    let k1 = seeded_bf16(&mut rng, T::SIZE * N::SIZE * D::SIZE, 0.2);
    let v0 = seeded_bf16(&mut rng, T::SIZE * N::SIZE * D::SIZE, 0.2);
    let v1 = seeded_bf16(&mut rng, T::SIZE * N::SIZE * D::SIZE, 0.2);
    let q_hbm = HostTensor::<bf16, m![N, G, D]>::from_vec(q.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let k0_hbm = HostTensor::<bf16, m![T, N, D]>::from_vec(k0.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let k1_hbm = HostTensor::<bf16, m![T, N, D]>::from_vec(k1.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let v0_hbm = HostTensor::<bf16, m![T, N, D]>::from_vec(v0.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let v1_hbm = HostTensor::<bf16, m![T, N, D]>::from_vec(v1.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let mask = HostTensor::<f32, m![T]>::from_vec(vec![1.0; T::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    launch(
        ops::attention_forward_first,
        (
            &mut *ctx,
            &q_hbm,
            &k0_hbm,
            &v0_hbm,
            &mask,
            &mut max_hbm,
            &mut sum_hbm,
            &mut out_hbm,
        ),
    )
    .await;
    launch(
        ops::attention_forward,
        (
            &mut *ctx,
            &q_hbm,
            &k1_hbm,
            &v1_hbm,
            &mask,
            &mut max_hbm,
            &mut sum_hbm,
            &mut out_hbm,
        ),
    )
    .await;

    let mut expected = vec![0.0; N::SIZE * G::SIZE * D::SIZE];
    let mut expected_max = vec![f32::NEG_INFINITY; N::SIZE * G::SIZE];
    let mut expected_sum = [0.0; N::SIZE * G::SIZE];
    for head in 0..N::SIZE * G::SIZE {
        for (keys, _) in [(&k0, &v0), (&k1, &v1)] {
            for token in 0..T::SIZE {
                let score = (0..D::SIZE)
                    .map(|d| {
                        q[head * D::SIZE + d].to_f32() * keys[(token * N::SIZE + head / G::SIZE) * D::SIZE + d].to_f32()
                    })
                    .sum::<f32>()
                    / (D::SIZE as f32).sqrt();
                expected_max[head] = expected_max[head].max(score);
            }
        }
        for (keys, values) in [(&k0, &v0), (&k1, &v1)] {
            for token in 0..T::SIZE {
                let score = (0..D::SIZE)
                    .map(|d| {
                        q[head * D::SIZE + d].to_f32() * keys[(token * N::SIZE + head / G::SIZE) * D::SIZE + d].to_f32()
                    })
                    .sum::<f32>()
                    / (D::SIZE as f32).sqrt();
                let weight = (score - expected_max[head]).exp();
                expected_sum[head] += weight;
                for d in 0..D::SIZE {
                    expected[head * D::SIZE + d] +=
                        weight * values[(token * N::SIZE + head / G::SIZE) * D::SIZE + d].to_f32();
                }
            }
        }
        for d in 0..D::SIZE {
            expected[head * D::SIZE + d] /= expected_sum[head];
        }
    }
    let sums = sum_hbm.to_host::<m![N, G]>(&mut ctx.pdma).await.into_vec();
    let actual = out_hbm.to_host::<m![N, G, D]>(&mut ctx.pdma).await.into_vec();
    let actual_normalized: Vec<_> = actual
        .iter()
        .enumerate()
        .map(|(index, value)| bf16::from_f32(value.to_f32() / sums[index / D::SIZE]))
        .collect();
    assert_close(&actual_normalized, &expected, 0.12, 0.12);
    for (actual, expected) in max_hbm
        .to_host::<m![N, G]>(&mut ctx.pdma)
        .await
        .into_vec()
        .iter()
        .zip(expected_max)
    {
        assert!(
            (actual - expected).abs() < 0.1,
            "attention max actual={actual} expected={expected}"
        );
    }
}

/// Wires one host-prepared token through projection, one attention chunk, and decoder.
/// Zero weights make the independent end-to-end oracle the residual identity while still
/// exercising the real tensor shapes and intermediate HBM boundaries.
#[tokio::test]
async fn test_decoder_step_wires_kernel_boundaries() {
    let mut rng = SmallRng::seed_from_u64(0x_dec0de);
    let mut ctx = Context::acquire();
    let input = seeded_bf16(&mut rng, H::SIZE, 0.2);
    let residual = seeded_bf16(&mut rng, H::SIZE, 0.2);
    let zeros_q = vec![bf16::from_f32(0.0); Q::SIZE * H::SIZE];
    let zeros_p = vec![bf16::from_f32(0.0); P::SIZE * H::SIZE];
    let zeros_h = vec![bf16::from_f32(0.0); H::SIZE];
    let zeros_l = vec![bf16::from_f32(0.0); L::SIZE * H::SIZE];
    let input_hbm = HostTensor::<bf16, m![H]>::from_vec(input).to_hbm(&mut ctx.pdma).await;
    let q_weight = HostTensor::<bf16, m![Q, H]>::from_vec(zeros_q)
        .to_hbm(&mut ctx.pdma)
        .await;
    let k_weight = HostTensor::<bf16, m![P, H]>::from_vec(zeros_p.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let v_weight = HostTensor::<bf16, m![P, H]>::from_vec(zeros_p)
        .to_hbm(&mut ctx.pdma)
        .await;
    let rms_hbm = HostTensor::<bf16, m![H]>::from_vec(vec![bf16::from_f32(1.0); H::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let q_rms = HostTensor::<bf16, m![D]>::from_vec(vec![bf16::from_f32(1.0); D::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let k_rms = HostTensor::<bf16, m![D]>::from_vec(vec![bf16::from_f32(1.0); D::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let offset = HostTensor::<i32, m![1]>::from_vec(vec![0]).to_hbm(&mut ctx.pdma).await;
    let cos = HostTensor::<bf16, m![D]>::from_vec(vec![bf16::from_f32(1.0); D::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let sin = HostTensor::<bf16, m![D]>::from_vec(vec![bf16::from_f32(0.0); D::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut k_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut v_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut q_out = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    launch(
        ops::projection,
        (
            &mut *ctx,
            &input_hbm,
            &q_weight,
            &k_weight,
            &v_weight,
            &rms_hbm,
            &q_rms,
            &k_rms,
            &offset,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            &mut q_out,
        ),
    )
    .await;
    let mask = HostTensor::<f32, m![T]>::from_vec(vec![1.0; T::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut attn_out = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    launch(
        ops::attention_forward_first,
        (
            &mut *ctx,
            &q_out,
            &k_cache,
            &v_cache,
            &mask,
            &mut max_hbm,
            &mut sum_hbm,
            &mut attn_out,
        ),
    )
    .await;
    let mut residual_hbm = HostTensor::<bf16, m![H]>::from_vec(residual.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let o_weight = HostTensor::<bf16, m![H, Q]>::from_vec(vec![bf16::from_f32(0.0); H::SIZE * Q::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let post = HostTensor::<bf16, m![H]>::from_vec(zeros_h).to_hbm(&mut ctx.pdma).await;
    let up = HostTensor::<bf16, m![L, H]>::from_vec(zeros_l.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let gate = HostTensor::<bf16, m![L, H]>::from_vec(zeros_l)
        .to_hbm(&mut ctx.pdma)
        .await;
    let down = HostTensor::<bf16, m![H, L]>::from_vec(vec![bf16::from_f32(0.0); H::SIZE * L::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    launch(
        ops::decoder,
        (
            &mut *ctx,
            &attn_out,
            &sum_hbm,
            &mut residual_hbm,
            &o_weight,
            &post,
            &up,
            &gate,
            &down,
        ),
    )
    .await;
    assert_close(
        &residual_hbm.to_host::<m![H]>(&mut ctx.pdma).await.into_vec(),
        &residual.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
        0.05,
        0.05,
    );
}

#[tokio::test]
async fn test_decoder() {
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut rx_hbm = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let o_weight = HostTensor::<bf16, m![H, Q]>::zero().to_hbm(&mut ctx.pdma).await;
    let post_rms_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let up_weight = HostTensor::<bf16, m![L, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let gate_weight = HostTensor::<bf16, m![L, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let down_weight = HostTensor::<bf16, m![H, L]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::decoder,
        (
            &mut *ctx,
            &x,
            &sum_hbm,
            &mut rx_hbm,
            &o_weight,
            &post_rms_weight,
            &up_weight,
            &gate_weight,
            &down_weight,
        ),
    )
    .await;
}

#[tokio::test]
async fn test_decoder_matches_scalar_reference() {
    let mut rng = SmallRng::seed_from_u64(0x_de_c0_de);
    let mut ctx = Context::acquire();
    let x = seeded_bf16(&mut rng, Q::SIZE, 0.05);
    let residual = seeded_bf16(&mut rng, H::SIZE, 0.05);
    let o_weight = seeded_bf16(&mut rng, H::SIZE * Q::SIZE, 0.02);
    let post_weight = seeded_bf16(&mut rng, H::SIZE, 1.0);
    let up_weight = seeded_bf16(&mut rng, L::SIZE * H::SIZE, 0.01);
    let gate_weight = seeded_bf16(&mut rng, L::SIZE * H::SIZE, 0.01);
    let down_weight = seeded_bf16(&mut rng, H::SIZE * L::SIZE, 0.01);
    let x_hbm = HostTensor::<bf16, m![N, G, D]>::from_vec(x.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let sum_hbm = HostTensor::<f32, m![N, G]>::from_vec(vec![1.0; N::SIZE * G::SIZE])
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut residual_hbm = HostTensor::<bf16, m![H]>::from_vec(residual.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let o_hbm = HostTensor::<bf16, m![H, Q]>::from_vec(o_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let post_hbm = HostTensor::<bf16, m![H]>::from_vec(post_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let up_hbm = HostTensor::<bf16, m![L, H]>::from_vec(up_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let gate_hbm = HostTensor::<bf16, m![L, H]>::from_vec(gate_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let down_hbm = HostTensor::<bf16, m![H, L]>::from_vec(down_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    launch(
        ops::decoder,
        (
            &mut *ctx,
            &x_hbm,
            &sum_hbm,
            &mut residual_hbm,
            &o_hbm,
            &post_hbm,
            &up_hbm,
            &gate_hbm,
            &down_hbm,
        ),
    )
    .await;

    let x_f32: Vec<_> = x.iter().map(|value| value.to_f32()).collect();
    let residual_f32: Vec<_> = residual.iter().map(|value| value.to_f32()).collect();
    let mut after_attention = vec![0.0; H::SIZE];
    for h in 0..H::SIZE {
        after_attention[h] = residual_f32[h]
            + (0..Q::SIZE)
                .map(|q| o_weight[h * Q::SIZE + q].to_f32() * x_f32[q])
                .sum::<f32>();
    }
    let after_norm = rms(&after_attention, &post_weight, 6.25e-8);
    let mut up = vec![0.0; L::SIZE];
    let mut gate = vec![0.0; L::SIZE];
    for row in 0..L::SIZE {
        up[row] = (0..H::SIZE)
            .map(|h| up_weight[row * H::SIZE + h].to_f32() * after_norm[h])
            .sum();
        gate[row] = (0..H::SIZE)
            .map(|h| gate_weight[row * H::SIZE + h].to_f32() * after_norm[h])
            .sum();
    }
    let mut expected = after_attention;
    for h in 0..H::SIZE {
        expected[h] += (0..L::SIZE)
            .map(|row| {
                down_weight[h * L::SIZE + row].to_f32() * (gate[row] * (1.0 + (-gate[row]).exp()).recip() * up[row])
            })
            .sum::<f32>();
    }
    assert_close(
        &residual_hbm.to_host::<m![H]>(&mut ctx.pdma).await.into_vec(),
        &expected,
        0.25,
        0.2,
    );
}

#[tokio::test]
async fn test_final_layer() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let rms_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let lm_head_weight = HostTensor::<bf16, m![W # 155648 / 8192, W # 155648 % 8192, H]>::zero()
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut out = HostTensor::<bf16, m![Wp]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::final_layer,
        (&mut *ctx, &input, &rms_weight, &lm_head_weight, &mut out),
    )
    .await;
}

#[tokio::test]
async fn test_final_layer_matches_scalar_reference() {
    let mut rng = SmallRng::seed_from_u64(0xf1_a1);
    let mut ctx = Context::acquire();
    let input = seeded_bf16(&mut rng, H::SIZE, 0.1);
    let rms_weight = seeded_bf16(&mut rng, H::SIZE, 1.0);
    let mut lm_head_weight = vec![bf16::from_f32(0.0); Wp::SIZE * H::SIZE];
    for row in 0..W::SIZE {
        lm_head_weight[row * H::SIZE] = bf16::from_f32((row % 17) as f32 * 0.01 - 0.08);
    }
    let input_hbm = HostTensor::<bf16, m![H]>::from_vec(input.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let rms_hbm = HostTensor::<bf16, m![H]>::from_vec(rms_weight.clone())
        .to_hbm(&mut ctx.pdma)
        .await;
    let weight_hbm = HostTensor::<bf16, m![W # 155648 / 8192, W # 155648 % 8192, H]>::from_vec(lm_head_weight)
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut out_hbm = HostTensor::<bf16, m![Wp]>::zero().to_hbm(&mut ctx.pdma).await;
    launch(
        ops::final_layer,
        (&mut *ctx, &input_hbm, &rms_hbm, &weight_hbm, &mut out_hbm),
    )
    .await;

    let input_f32: Vec<_> = input.iter().map(|value| value.to_f32()).collect();
    let normalized = rms(&input_f32, &rms_weight, 6.25e-8);
    let expected: Vec<_> = (0..W::SIZE)
        .map(|row| normalized[0] * ((row % 17) as f32 * 0.01 - 0.08))
        .collect();
    let actual = out_hbm.to_host::<m![Wp]>(&mut ctx.pdma).await.into_vec();
    assert_close(&actual[..W::SIZE], &expected, 0.08, 0.08);
}
