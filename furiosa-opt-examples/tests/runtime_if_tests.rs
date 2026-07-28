//! Answer-key tests for the runtime-`if` device functions: run them on the VISA
//! simulator and check the computed output against the oracle in each doc.

use furiosa_opt_examples::runtime_if::{
    W, runtime_if_accumulate, runtime_if_chain, runtime_if_const, runtime_if_two_outputs, runtime_if_value,
};
use furiosa_opt_std::prelude::*;

fn input_host() -> HostTensor<i32, m![W]> {
    HostTensor::<i32, m![W]>::from_vec((0..<m![W]>::SIZE as i32).collect::<Vec<_>>())
}

/// The single dead-carry example: `result` is reassigned without being read, last iteration `i == 1`
/// takes the then-arm, so `output == input + 1`.
#[tokio::test]
async fn test_runtime_if_value() {
    let mut ctx = Context::acquire();
    let input = input_host();
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(runtime_if_value, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![W]>(&mut ctx.pdma).await.into_vec()
    );
}

/// Live loop-carried accumulator (`result` read each iteration): `+2` at `i == 0` then `+1` at
/// `i == 1`, so `output == input + 3`.
#[tokio::test]
async fn test_runtime_if_accumulate() {
    let mut ctx = Context::acquire();
    let input = input_host();
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(runtime_if_accumulate, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 3).into_vec(),
        out.to_host::<m![W]>(&mut ctx.pdma).await.into_vec()
    );
}

/// Branch result consumed by a further op (`mid` then `+10`) over a live carry: `output == input + 23`.
#[tokio::test]
async fn test_runtime_if_chain() {
    let mut ctx = Context::acquire();
    let input = input_host();
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(runtime_if_chain, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 23).into_vec(),
        out.to_host::<m![W]>(&mut ctx.pdma).await.into_vec()
    );
}

/// Statement form: the two arms write two separate output tensors, `input + 1` and `input + 2`.
#[tokio::test]
async fn test_runtime_if_two_outputs() {
    let mut ctx = Context::acquire();
    let input = input_host();
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let (out_then, out_else) = launch(runtime_if_two_outputs, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out_then.to_host::<m![W]>(&mut ctx.pdma).await.into_vec()
    );
    assert_eq!(
        input.clone().into_inner().map(|x| x + 2).into_vec(),
        out_else.to_host::<m![W]>(&mut ctx.pdma).await.into_vec()
    );
}

/// Constant condition (`if true`): the then-arm is always taken, so `output == input + 1`.
#[tokio::test]
async fn test_runtime_if_const() {
    let mut ctx = Context::acquire();
    let input = input_host();
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(runtime_if_const, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![W]>(&mut ctx.pdma).await.into_vec()
    );
}
