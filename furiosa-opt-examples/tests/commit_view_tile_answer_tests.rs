//! Answer-key tests for `commit_view_tile`: run the `#[device]` body under the
//! default (simulation) backend and compare against an independent host oracle.
//! Unlike the `npu-visa-test` `compare_edf` tests (VISA-sim vs LIR/EDF), this
//! pins whether the VISA/device simulation itself is correct, localizing a
//! divergence to either the eDSL/example or the LIR lowering.

use furiosa_opt_examples::commit_view_tile::{A, X, swap_halves};
use furiosa_opt_std::prelude::*;

type Chip = m![1];

/// Ramp `0, 1, 2, ...` as bf16 (values < 256 are exact in bf16, so the moves
/// are byte-exact and `assert_eq!` needs no tolerance).
fn ramp() -> Vec<bf16> {
    (0..<m![A, X]>::SIZE).map(|i| bf16::from_f32(i as f32)).collect()
}

/// Host oracle: swap the two `X`-halves of each `[A, X]` row.
fn swapped_halves(input: &[bf16]) -> Vec<bf16> {
    let x = <m![X]>::SIZE;
    let half = x / 2;
    input
        .chunks_exact(x)
        .flat_map(|row| row[half..].iter().chain(&row[..half]).copied())
        .collect()
}

/// True-math oracle for [`swap_halves`]: the second `X`-half comes first. The
/// read tile of the second half uses the ELEMENT index `16` (not the byte
/// offset `32`); a byte-unit fetch offset would read element `16 / 2 = 8` and
/// fail this check.
#[tokio::test]
async fn answer_swap_halves() {
    let input_vals = ramp();
    let expected = swapped_halves(&input_vals);
    assert_ne!(
        expected, input_vals,
        "the swap must actually move data (not an identity)"
    );

    let mut ctx = Context::acquire();
    let input_hbm = HostTensor::<bf16, m![A, X]>::from_vec(input_vals)
        .to_hbm::<Chip, m![A, X]>(&mut ctx.pdma)
        .await;

    let out = launch(swap_halves, (&mut *ctx, &input_hbm)).await;

    assert_eq!(
        expected,
        out.to_host::<m![A, X]>(&mut ctx.pdma).await.into_vec(),
        "host oracle (left) vs VISA-sim (right)"
    );
}
