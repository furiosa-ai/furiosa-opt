//! Generic `#[device]` fixtures for npu-opt's `registry` test: launched from test-shaped
//! roots, and each root's binary registers exactly its own instantiations. A generic fn has no
//! artifact of its own. Its kernels compile at a root's link.

use furiosa_opt_std::prelude::*;

// B spans the whole device (cluster 2 x slice 256). The other axes are generic key values.
axes![B = 512, C = 32, D = 64, E = 128, F = 16];

#[device(chip = 1)]
pub fn bundle_copy<A>(device: &mut Device, hbm: &HbmTensor<i32, m![1], m![B, A]>) -> HbmTensor<i32, m![1], m![B, A]>
where
    A: AxisName,
{
    let dm: DmTensor<i32, m![1], m![2], m![B / 2], m![A]> = hbm.to_dm(&mut device.tdma);
    dm.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn pair_copy<A: AxisName, R: AxisName>(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![1], m![A, R]>,
) -> HbmTensor<i32, m![1], m![A, R]> {
    let dm: DmTensor<i32, m![1], m![2], m![A / 2], m![R]> = hbm.to_dm(&mut device.tdma);
    dm.to_hbm(&mut device.tdma)
}

// A build-only launch root inside the lib's unit tests: the `--test` harness unit links like
// any other root and its binary must register the launch. Never executed (a launch needs
// hardware): the poll hides behind `black_box(false)`. Polling once pulls the launch into the
// mono set.
#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;

    fn missing<T>() -> T {
        panic!("build-only fixture")
    }

    async fn launches() {
        let (mut device, f): (Device, HbmTensor<i32, m![1], m![B, F]>) = missing();
        launch(bundle_copy, (&mut device, &f)).await.unwrap();
    }

    #[test]
    fn unit_launch_is_registered() {
        if std::hint::black_box(false) {
            let waker = std::task::Waker::noop();
            let mut cx = std::task::Context::from_waker(waker);
            let _ = std::pin::pin!(launches()).poll(&mut cx);
        }
    }
}
