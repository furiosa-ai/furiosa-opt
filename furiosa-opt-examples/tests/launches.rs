//! Build-only root for npu-opt's `registry` test: this harness binary's registry must carry
//! exactly the launches below. `missing` must be a returning call: an inline `panic!` diverges
//! and drops the launches out of the monomorphization collector's sight.

use std::future::Future;

use furiosa_opt_examples::generic::{B, C, D, E, bundle_copy, pair_copy};
use furiosa_opt_std::prelude::*;

fn missing<T>() -> T {
    panic!("build-only fixture")
}

async fn launches() {
    type InputC = HbmTensor<i32, m![1], m![B, C]>;
    type InputD = HbmTensor<i32, m![1], m![B, D]>;
    type InputE = HbmTensor<i32, m![1], m![B, E]>;
    let (mut ctx, c, d, e): (Context, InputC, InputD, InputE) = missing();
    launch(bundle_copy, (&mut ctx, &c)).await;
    launch(bundle_copy, (&mut ctx, &d)).await;
    launch(bundle_copy, (&mut ctx, &e)).await;
    launch(pair_copy, (&mut ctx, &c)).await;
    launch(pair_copy, (&mut ctx, &d)).await;
}

#[test]
fn roots_register_their_launches() {
    if std::hint::black_box(false) {
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        let _ = std::pin::pin!(launches()).poll(&mut cx);
    }
}
