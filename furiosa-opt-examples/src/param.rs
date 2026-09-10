//! Tuple-, struct-, array-, and borrowed-typed `#[device]` parameter examples.
//!
//! Each passthrough returns its input tensor unchanged (HBM -> SRAM -> HBM), so the
//! tests can confirm every parameter shape reaches the kernel by checking
//! the data survives the round-trip.

use furiosa_opt_std::prelude::*;

axes![Layers = 2, A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[derive(DeviceSend)]
pub struct Inputs<'a> {
    pub x: &'a HbmTensor<i8, Chip, m![A, B]>,
}

#[derive(DeviceSend)]
pub struct Weight<'a> {
    pub x: &'a HbmTensor<i8, Chip, m![A, B]>,
}

#[derive(DeviceSend)]
pub struct Layer<'a> {
    pub weight: Weight<'a>,
}

#[derive(DeviceSend)]
pub struct Model<'a, const N: usize> {
    pub layers: [Layer<'a>; N],
}

fn copy(device: &mut Device, input: &HbmTensor<i8, Chip, m![A, B]>) -> HbmTensor<i8, Chip, m![A, B]> {
    input
        .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn tuple_passthrough(
    device: &mut Device,
    inputs: (&HbmTensor<i8, Chip, m![A, B]>,),
) -> HbmTensor<i8, Chip, m![A, B]> {
    copy(device, inputs.0)
}

#[device(chip = 1)]
pub fn struct_passthrough(device: &mut Device, inputs: Inputs<'_>) -> HbmTensor<i8, Chip, m![A, B]> {
    copy(device, inputs.x)
}

#[device(chip = 1)]
pub fn array_passthrough(
    device: &mut Device,
    inputs: [Inputs<'_>; 2],
) -> (HbmTensor<i8, Chip, m![A, B]>, HbmTensor<i8, Chip, m![A, B]>) {
    let [first, second] = inputs;
    (copy(device, first.x), copy(device, second.x))
}

#[device(chip = 1)]
pub fn borrowed_model_passthrough(
    device: &mut Device,
    model: &Model<'_, 2>,
) -> (HbmTensor<i8, Chip, m![A, B]>, HbmTensor<i8, Chip, m![A, B]>) {
    (
        copy(device, model.layers[0].weight.x),
        copy(device, model.layers[1].weight.x),
    )
}

#[device(chip = 1)]
pub fn borrowed_model_layer_loop(
    device: &mut Device,
    model: &Model<'_, 2>,
    output: &mut HbmTensor<i8, Chip, m![Layers, A, B]>,
) {
    for index in 0..2 {
        let output = output.view_mut().tile::<m![Layers], 1, m![1 #{!} 2, A, B]>(index);
        model.layers[index]
            .weight
            .x
            .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut device.tdma)
            .view()
            .to_hbm_view(&mut device.tdma, output);
    }
}

#[device(chip = 1)]
pub fn local_struct_passthrough(
    device: &mut Device,
    x: &HbmTensor<i8, Chip, m![A, B]>,
) -> HbmTensor<i8, Chip, m![A, B]> {
    let inputs = Inputs { x };
    copy(device, inputs.x)
}

#[device(chip = 1)]
pub fn explicit_unit(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![A, B]>,
    output: &mut HbmTensor<i8, Chip, m![A, B]>,
) -> () {
    input
        .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut device.tdma)
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
}
