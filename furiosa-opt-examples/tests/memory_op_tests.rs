use furiosa_opt_examples::memory_op::{A, B, Q, T, commit_view_bottom_pad, dm_relayout};
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
