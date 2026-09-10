//! A launch whose wait is dropped part-way frees its answer slot and buffers only once the device
//! has answered. Gated to the `npu` backend, where a wait can be dropped mid-flight.
#![cfg(backend = "npu")]

use std::future::Future;
use std::pin::pin;
use std::task::{Device as TaskContext, Waker};

use furiosa_opt_examples::transformer::axes::{H, W, Wp};
use furiosa_opt_examples::transformer::ops;
use furiosa_opt_std::backend::npu;
use furiosa_opt_std::prelude::*;

type Weight = m![W # 155648 / 8192, W # 155648 % 8192, H];

/// More launches than one cluster's answer slots, so a leaked slot would stall the tail.
const DROPPED: usize = 24;

/// The final layer streams hundreds of megabytes of weight, so every drop cuts a running launch.
#[tokio::test]
async fn dropped_wait_frees_output() {
    let mut device = Device::new(ops::final_layer.topology()).unwrap();
    let input = HostTensor::<bf16, m![H]>::from_vec((0..H::SIZE).map(|i| bf16::from_f32((i % 7) as f32 * 0.1)))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let rms = HostTensor::<bf16, m![H]>::from_vec(vec![bf16::from_f32(1.0); H::SIZE])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let mut lm_head = vec![bf16::from_f32(0.0); Wp::SIZE * H::SIZE];
    for row in 0..W::SIZE {
        lm_head[row * H::SIZE] = bf16::from_f32((row % 17) as f32 * 0.01 - 0.08);
    }
    let weight = HostTensor::<bf16, Weight>::from_vec(lm_head)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let mut out = HostTensor::<bf16, m![Wp]>::zero()
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    // Loading the function awaits its own transfers; only a loaded function submits on the first poll.
    launch(ops::final_layer, (&mut device, &input, &rms, &weight, &mut out))
        .await
        .unwrap();
    let expected = out.to_host::<m![Wp]>(&mut device.pdma).await.unwrap().into_vec();
    let function = npu::function(&device, ops::final_layer.path(), &[]).await.unwrap();
    let mut args = Buffers::new();
    (&input, &rms, &weight, &mut out).bind(&mut args);
    let output = args[3].addr();

    let mut cancelled = 0;
    for _ in 0..DROPPED {
        let mut run = pin!(function.run(&args, &[]));
        if run
            .as_mut()
            .poll(&mut TaskContext::from_waker(Waker::noop()))
            .is_pending()
        {
            cancelled += 1;
        }
    }
    assert!(cancelled > 0, "no launch was still running at its first poll");
    // The bound arguments share the allocation, so both handles go before it is free.
    drop(args);
    drop(out);

    let pattern = HostTensor::<bf16, m![Wp]>::from_vec(vec![bf16::from_f32(-1.0); Wp::SIZE])
        .to_hbm::<m![1], m![Wp]>(&mut device.pdma)
        .await
        .unwrap();
    let mut placed = Buffers::new();
    pattern.bind(&mut placed);
    assert_eq!(
        placed[0].addr(),
        output,
        "the dropped launches' output region is reused at once"
    );
    assert_eq!(
        pattern.to_host::<m![Wp]>(&mut device.pdma).await.unwrap().into_vec(),
        vec![bf16::from_f32(-1.0); Wp::SIZE]
    );

    let mut again = HostTensor::<bf16, m![Wp]>::zero()
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    launch(ops::final_layer, (&mut device, &input, &rms, &weight, &mut again))
        .await
        .unwrap();
    assert_eq!(
        again.to_host::<m![Wp]>(&mut device.pdma).await.unwrap().into_vec(),
        expected
    );
}
