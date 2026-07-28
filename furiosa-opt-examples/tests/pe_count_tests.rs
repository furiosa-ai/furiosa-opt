//! Value-oracle tests for the per-config `pe_count` examples: each `*_pe_add` adds 1 end to end on
//! the VISA simulator (which honors the kernel's `#[device(chip, pe)]`), so the arithmetic is
//! value-checked and not only cross-stage consistency-checked (as in the `npu-visa-test`
//! `compare_edf` companions).

use furiosa_opt_examples::pe_count::{
    P1, P2, P2C, P4, P4C, P8, eight_pe_add, four_chip_add, four_pe_add, one_pe_add, two_chip_add, two_pe_add,
};
use furiosa_opt_std::prelude::*;

/// `one_pe_add` computes `output == input + 1` on a 1-PE device.
#[tokio::test]
async fn test_one_pe_add() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<i32, m![P1]>::from_vec((0..<m![P1]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(one_pe_add, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P1]>(&mut ctx.pdma).await.into_vec()
    );
}

/// `two_pe_add` computes `output == input + 1` on a 2-PE device.
#[tokio::test]
async fn test_two_pe_add() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<i32, m![P2]>::from_vec((0..<m![P2]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(two_pe_add, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P2]>(&mut ctx.pdma).await.into_vec()
    );
}

/// `four_pe_add` computes `output == input + 1` on a 4-PE device.
#[tokio::test]
async fn test_four_pe_add() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<i32, m![P4]>::from_vec((0..<m![P4]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(four_pe_add, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P4]>(&mut ctx.pdma).await.into_vec()
    );
}

/// `eight_pe_add` computes `output == input + 1` on an 8-PE device (2 clusters).
#[tokio::test]
async fn test_eight_pe_add() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<i32, m![P8]>::from_vec((0..<m![P8]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(eight_pe_add, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P8]>(&mut ctx.pdma).await.into_vec()
    );
}

/// `two_chip_add` computes `output == input + 1` on a 2-chip 8-PE device.
#[tokio::test]
async fn test_two_chip_add() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<i32, m![P2C]>::from_vec((0..<m![P2C]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(two_chip_add, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P2C]>(&mut ctx.pdma).await.into_vec()
    );
}

/// `four_chip_add` computes `output == input + 1` on a 4-chip 8-PE device.
#[tokio::test]
async fn test_four_chip_add() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<i32, m![P4C]>::from_vec((0..<m![P4C]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;

    let out = launch(four_chip_add, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P4C]>(&mut ctx.pdma).await.into_vec()
    );
}
