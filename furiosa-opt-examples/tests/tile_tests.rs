use furiosa_opt_examples::tile::{A, B, D, tile_simple, tile_window_commit};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Host function to test the device function.
#[tokio::test]
async fn test_tile_simple_host() {
    // Host operations: create tensors, transfer to device
    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let mut ctx = Context::acquire();
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    // Device operation via launch
    let output_hbm = launch(tile_simple, (&mut ctx, input_hbm.view())).await;

    // Host operation: transfer back
    let output = output_hbm.to_host::<m![B, A]>(&mut ctx.pdma).await;

    assert_eq!(
        input.into_inner().transpose::<m![B, A]>(false).into_vec(),
        output.into_vec(),
        "Transpose should not change the mathematical meaning of tensor"
    );
}

/// Commits a fetched 32-wide window into the upper half of a 64-wide down-padded DM tile: the kernel
/// writes only `result[32..64]` (= `input[0..32]`) and leaves `result[0..32]` unwritten. A freshly
/// allocated destination starts as an all-zero blank canvas (`Backend::uninit`) that only the
/// commit overwrites, so "unwritten" is the concrete, checkable claim "still zero".
#[tokio::test]
async fn test_tile_window_commit_host() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<f32, m![D]>::from_vec((0..64).map(|x| x as f32).collect::<Vec<_>>())
        .to_hbm::<m![1], m![D]>(&mut ctx.pdma)
        .await;

    let output = launch(tile_window_commit, (&mut *ctx, &input)).await;

    // result[32..64] is written from input[0..32]; result[0..32] (the out-of-tile down-pad cells)
    // must stay at the destination's zero-filled default.
    let actual: Vec<f32> = output.to_host::<m![D]>(&mut ctx.pdma).await.into_vec();
    for i in 0..32 {
        assert_eq!(
            actual[i], 0.0,
            "result[{i}] (out-of-tile) must stay unwritten, got {:?}",
            actual[i]
        );
        assert_eq!(actual[32 + i], i as f32, "result[{}] should equal input[{}]", 32 + i, i);
    }
}
