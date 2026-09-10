//! Rejection fixture for a runtime `if` feeding a *vector-engine immediate*: an immediate is encoded
//! in the instruction, so it must const-fold to a compile-time constant, and the `if` result is a
//! runtime scalar that does not.
//!
//! The same runtime scalar IS a valid view/tile index, which is the public twin
//! [`runtime_if_scalar::runtime_if_scalar_index`](crate::runtime_if_scalar::runtime_if_scalar_index).

use furiosa_opt_std::prelude::*;

pub use crate::runtime_if_scalar::N;

type Chip = m![1];

type Row = DmTensor<i32, Chip, m![1], m![N / 4], m![N % 4]>;

/// Runtime `if` selecting a vector-engine IMMEDIATE: `vector_fxp(AddFxp, if i == 1 { 1 } else { 2 })`.
///
/// Expected: `a vector-engine immediate must be a compile-time constant, but `...` does not const-fold`.
///
/// Runs at a 4-PE device (1 cluster) so the `Cluster = m![1]` `Row` is a valid allocation and the
/// compile reaches the vector-engine-immediate check that is the point of this negative example.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_scalar_immediate(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![N]>,
) -> HbmTensor<i32, Chip, m![N]> {
    let input_dm: Row = input.to_dm(&mut device.tdma);
    let mut result: Row = DmTensor::new();
    for i in 0..2 {
        let c: i32 = if i == 1 { 1 } else { 2 };
        result = device
            .main
            .begin(input_dm.view())
            .fetch::<m![1], m![N % 4]>()
            .fetch_cast::<i32>()
            .collect::<m![1], m![N % 4 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, c)
            .vector_final()
            .commit_trim::<m![N % 4]>()
            .commit();
    }
    result.to_hbm(&mut device.tdma)
}

/// An observable overflow flag from `overflowing_add` must not be discarded.
#[device(chip = 1)]
pub fn overflowing_add_flag(device: &mut Device, input_hbm: &HbmTensor<i8, Chip, m![N]>) -> HbmTensor<i8, Chip, m![N]> {
    let input = input_hbm.to_dm::<m![1], m![N / 4], m![N % 4]>(&mut device.tdma);
    let mut output = DmTensor::<i8, Chip, m![1], m![N / 4], m![N % 4]>::new();
    for i in 0..2usize {
        let (off, overflow) = i.overflowing_add(0);
        let off = if overflow { 0 } else { off * 128 };
        let src = input.view().tile::<m![N / 4], 128, m![N = 512 # 1024]>(off);
        let dst = output.view_mut().tile::<m![N / 4], 128, m![N = 512 #{!} 1024]>(off);
        src.to_dm_view(&mut device.tdma, dst);
    }
    output.to_hbm(&mut device.tdma)
}

/// An observable overflow flag from `overflowing_sub` must not be discarded.
#[device(chip = 1)]
pub fn overflowing_sub_flag(device: &mut Device, input_hbm: &HbmTensor<i8, Chip, m![N]>) -> HbmTensor<i8, Chip, m![N]> {
    let input = input_hbm.to_dm::<m![1], m![N / 4], m![N % 4]>(&mut device.tdma);
    let mut output = DmTensor::<i8, Chip, m![1], m![N / 4], m![N % 4]>::new();
    for i in 0..2usize {
        let (off, overflow) = (i * 128).overflowing_sub(0);
        let off = if overflow { 0 } else { off };
        let src = input.view().tile::<m![N / 4], 128, m![N = 512 # 1024]>(off);
        let dst = output.view_mut().tile::<m![N / 4], 128, m![N = 512 #{!} 1024]>(off);
        src.to_dm_view(&mut device.tdma, dst);
    }
    output.to_hbm(&mut device.tdma)
}

/// An observable overflow flag from `overflowing_mul` must not be discarded.
#[device(chip = 1)]
pub fn overflowing_mul_flag(device: &mut Device, input_hbm: &HbmTensor<i8, Chip, m![N]>) -> HbmTensor<i8, Chip, m![N]> {
    let input = input_hbm.to_dm::<m![1], m![N / 4], m![N % 4]>(&mut device.tdma);
    let mut output = DmTensor::<i8, Chip, m![1], m![N / 4], m![N % 4]>::new();
    for i in 0..2usize {
        let (off, overflow) = i.overflowing_mul(128);
        let off = if overflow { 0 } else { off };
        let src = input.view().tile::<m![N / 4], 128, m![N = 512 # 1024]>(off);
        let dst = output.view_mut().tile::<m![N / 4], 128, m![N = 512 #{!} 1024]>(off);
        src.to_dm_view(&mut device.tdma, dst);
    }
    output.to_hbm(&mut device.tdma)
}
