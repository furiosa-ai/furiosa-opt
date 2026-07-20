#![expect(clippy::type_complexity)]

use furiosa_opt_examples::memory_op::{
    A, B, Q, T, commit_view_bottom_pad, dm_pcopy, dm_relayout, dm_view_pcopy, hbm_chip_shuffle,
};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
async fn test_dm_relayout() {
    let mut ctx = Context::acquire();
    let hbm: HbmTensor<i32, m![1], m![A, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
            .await;
    launch(dm_relayout, (&mut *ctx, &hbm)).await;
}

#[tokio::test]
async fn test_dm_pcopy() {
    let mut ctx = Context::acquire();
    let hbm: HbmTensor<i32, m![1], m![A, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
            .await;
    launch(dm_pcopy, (&mut *ctx, &hbm)).await;
}

#[tokio::test]
async fn test_dm_view_pcopy() {
    let mut ctx = Context::acquire();
    let hbm: HbmTensor<i32, m![1], m![A, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
            .await;
    launch(dm_view_pcopy, (&mut *ctx, &hbm)).await;
}

#[tokio::test]
async fn test_hbm_chip_shuffle() {
    let mut ctx = Context::acquire();
    let hbm: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma)
            .await;
    launch(hbm_chip_shuffle, (&mut *ctx, &hbm)).await;
}

/// `commit_view` into a down-padded tiled `view_mut` — checks bottom padding.
#[tokio::test]
async fn test_commit_view_bottom_pad() {
    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![T, Q % 56 = 8]>::rand(&mut rng)
        .to_hbm::<m![1], m![T, Q % 56 = 8]>(&mut ctx.pdma)
        .await;

    launch(commit_view_bottom_pad, (&mut *ctx, &input)).await;
}
