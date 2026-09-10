//! Two contexts of one topology partition a device, each taking only the PEs its function needs.
//! Gated to the `npu` backend: the simulator reserves nothing.
#![cfg(backend = "npu")]

use furiosa_opt_examples::pe_count::{P1, one_pe_add};
use furiosa_opt_std::prelude::*;

/// Each acquisition opens a device of its own, and both hold theirs throughout, so the second
/// one's PEs are chosen while the first one's are taken.
#[tokio::test]
async fn each_acquisition_takes_its_own_pes() {
    let mut contexts = [
        Device::new(one_pe_add.topology()).expect("a 1-PE device"),
        Device::new(one_pe_add.topology()).expect("a 1-PE device beside another"),
    ];
    let input =
        |offset: i32| HostTensor::<i32, m![P1]>::from_vec((offset..offset + <m![P1]>::SIZE as i32).collect::<Vec<_>>());

    let mut outputs = Vec::new();
    for (offset, device) in contexts.iter_mut().enumerate() {
        let input_hbm = input(offset as i32).to_hbm(&mut device.pdma).await.unwrap();
        outputs.push(launch(one_pe_add, (&mut *device, &input_hbm)).await.unwrap());
    }

    for (offset, (device, out)) in contexts.iter_mut().zip(outputs).enumerate() {
        assert_eq!(
            input(offset as i32).into_inner().map(|x| x + 1).into_vec(),
            out.to_host::<m![P1]>(&mut device.pdma).await.unwrap().into_vec(),
            "device {offset}"
        );
    }
}
