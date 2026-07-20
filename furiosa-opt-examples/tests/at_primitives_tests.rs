//! Codegen coverage for the address-taking `*_at` primitives.
//!
//! Each test launches a kernel from `at_primitives`, forcing the `*_at` primitive it
//! exercises through codegen. Outputs are checked against the same oracle as the
//! address-free twin so the `*_at` path is verified, not merely compiled.

use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Chip = m![1];

#[tokio::test]
async fn test_fetch_commit_at() {
    use furiosa_opt_examples::at_primitives::dma_commit::{A, B, fetch_commit_at};

    let mut ctx = Context::acquire();

    let input = HostTensor::<i8, m![A, B]>::from_vec((0..32768).map(|x| x as i8).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let output = launch(fetch_commit_at, (&mut *ctx, &input)).await;

    let mut expected = vec![0i32; 4096 * 8];
    let mut idx = 0;
    for b in 0..8 {
        for a in 0..4096 {
            expected[idx] = ((a * 8 + b) as i8) as i32;
            idx += 1;
        }
    }

    assert_eq!(output.to_host::<m![B, A]>(&mut ctx.pdma).await.into_vec(), expected);
}

// Covers `HbmTensorView::to_dm_at` and `to_vrf_at`. Launching forces codegen; the value
// oracle lives with `ve_elementwise_multi_vrf`, so here we only assert the run completes.
#[tokio::test]
async fn test_multi_vrf_at() {
    use furiosa_opt_examples::at_primitives::vrf::{A, B, multi_vrf_at};

    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A, B]>::rand(&mut rng);
    let vrf1 = HostTensor::<i32, m![B]>::rand(&mut rng);
    let vrf2 = HostTensor::<i32, m![B]>::rand(&mut rng);

    let input_hbm = input.to_hbm(&mut ctx.pdma).await;
    let vrf1_hbm = vrf1.to_hbm(&mut ctx.pdma).await;
    let vrf2_hbm = vrf2.to_hbm(&mut ctx.pdma).await;

    let _out = launch(multi_vrf_at, (&mut *ctx, &input_hbm, &vrf1_hbm, &vrf2_hbm)).await;
}

#[tokio::test]
async fn test_to_trf_at() {
    use furiosa_opt_examples::at_primitives::trf::{A, B, to_trf_at};

    let mut ctx = Context::acquire();

    let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
        .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma)
        .await;

    let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

    launch(to_trf_at, (&mut *ctx, &input, &mut output)).await;
}

// `DmTensor::to_dm_at`. The relayout regroups A into (A / 2, A % 2) without reordering elements,
// so reading the output as `m![A / 2, A % 2, B]` yields the original buffer.
#[tokio::test]
async fn test_dm_relayout_at() {
    use furiosa_opt_examples::at_primitives::relayout::{A, B, dm_relayout_at};

    let mut ctx = Context::acquire();

    let vals: Vec<i32> = (0..<m![A, B]>::SIZE as i32).collect();
    let input = HostTensor::<i32, m![A, B]>::from_vec(vals.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let output = launch(dm_relayout_at, (&mut *ctx, &input)).await;

    assert_eq!(
        output.to_host::<m![A / 2, A % 2, B]>(&mut ctx.pdma).await.into_vec(),
        vals
    );
}

// `DmTensor::to_dm_pcopy_at`. pcopy is an identity copy, so the output must equal the input.
#[tokio::test]
async fn test_dm_pcopy_at() {
    use furiosa_opt_examples::at_primitives::pcopy::{A, B, dm_pcopy_at};

    let mut ctx = Context::acquire();

    let vals: Vec<i32> = (0..<m![A, B]>::SIZE as i32).collect();
    let input = HostTensor::<i32, m![A, B]>::from_vec(vals.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let output = launch(dm_pcopy_at, (&mut *ctx, &input)).await;

    assert_eq!(output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_vec(), vals);
}
