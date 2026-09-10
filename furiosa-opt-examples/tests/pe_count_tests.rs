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
    let mut device = Device::new(one_pe_add.topology()).unwrap();
    let input = HostTensor::<i32, m![P1]>::from_vec((0..<m![P1]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(one_pe_add, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P1]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

/// `two_pe_add` computes `output == input + 1` on a 2-PE device.
#[tokio::test]
async fn test_two_pe_add() {
    let mut device = Device::new(two_pe_add.topology()).unwrap();
    let input = HostTensor::<i32, m![P2]>::from_vec((0..<m![P2]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(two_pe_add, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P2]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

/// `four_pe_add` computes `output == input + 1` on a 4-PE device.
#[tokio::test]
async fn test_four_pe_add() {
    let mut device = Device::new(four_pe_add.topology()).unwrap();
    let input = HostTensor::<i32, m![P4]>::from_vec((0..<m![P4]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(four_pe_add, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P4]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

/// `eight_pe_add` computes `output == input + 1` on an 8-PE device (2 clusters).
#[tokio::test]
async fn test_eight_pe_add() {
    let mut device = Device::new(eight_pe_add.topology()).unwrap();
    let input = HostTensor::<i32, m![P8]>::from_vec((0..<m![P8]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(eight_pe_add, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P8]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

/// `two_chip_add` computes `output == input + 1` on a 2-chip 8-PE device.
#[tokio::test]
async fn test_two_chip_add() {
    let mut device = Device::new(two_chip_add.topology()).unwrap();
    let input = HostTensor::<i32, m![P2C]>::from_vec((0..<m![P2C]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(two_chip_add, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P2C]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

/// `four_chip_add` computes `output == input + 1` on a 4-chip 8-PE device.
#[tokio::test]
async fn test_four_chip_add() {
    let mut device = Device::new(four_chip_add.topology()).unwrap();
    let input = HostTensor::<i32, m![P4C]>::from_vec((0..<m![P4C]>::SIZE as i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(four_chip_add, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![P4C]>(&mut device.pdma).await.unwrap().into_vec()
    );
}
