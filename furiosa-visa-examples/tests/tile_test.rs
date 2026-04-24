use furiosa_visa_examples::tile::{A, B, tile_simple};
use furiosa_visa_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Host function to test the device function.
#[tokio::test]
async fn test_tile_simple_host() {
    // Host operations: create tensors, transfer to device
    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let mut ctx = Context::acquire();
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;

    // Device operation via launch
    let output_hbm = launch(tile_simple, (&mut ctx, input_hbm.view())).await;

    // Host operation: transfer back
    let output = output_hbm.to_host::<m![B, A]>(&mut ctx.pdma).await;

    assert_eq!(
        unsafe { input.into_inner().transmute::<m![B, A]>() }.to_buf(),
        output.to_buf(),
        "Transpose should not change the mathematical meaning of tensor"
    );
}
