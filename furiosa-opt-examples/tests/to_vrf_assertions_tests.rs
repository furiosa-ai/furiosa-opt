use furiosa_opt_examples::to_vrf_assertions::{A, to_vrf_at_capacity};
use furiosa_opt_std::prelude::*;

type Chip = m![1];

/// An operand filling the register file exactly is accepted and runs. The rejected case is a
/// compile-time error, so it lives as a `compile_fail` example in the module's docs instead.
#[tokio::test]
async fn test_to_vrf_at_capacity() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<i32, m![A]>::from_vec((0..<m![A]>::SIZE).map(|x| x as i32).collect::<Vec<_>>())
        .to_hbm::<Chip, m![A]>(&mut ctx.pdma)
        .await;

    launch(to_vrf_at_capacity, (&mut *ctx, &input)).await;
}
