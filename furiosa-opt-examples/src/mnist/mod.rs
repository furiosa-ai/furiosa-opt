use furiosa_opt_std::prelude::*;

axes![X = 800, H = 256, C = 16, I = 2, Dummy8 = 8, Dummy16 = 16];

type Chip = m![1];
type Cluster = m![1 # 2];

fn fc1_matmul(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![X]>,
    weight: &HbmTensor<bf16, Chip, m![H, X]>,
) -> DmTensor<bf16, Chip, Cluster, m![H], m![1 # 16]> {
    let input_dm: DmTensor<bf16, Chip, Cluster, m![H], m![X]> = input.to_dm(&mut device.tdma);
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![H], m![X]> = weight.to_dm(&mut device.tdma);

    let input_trf: TrfTensor<bf16, Chip, Cluster, m![H], m![1], m![X]> = device
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![X]>()
        .collect::<m![X / 16], m![X % 16]>()
        .to_trf();

    device
        .main
        .begin(weight_dm.view())
        .fetch::<m![X / 16], m![X % 16]>()
        .collect::<m![X / 16], m![X % 16]>()
        .contract_outer::<m![X / 32], m![X % 32], _, _, _>(&input_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 16]>()
        .commit()
}

fn fc1_bias_prepared(
    device: &mut Device,
    bias: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![H], m![1 # 16]> {
    let bias_dm_0: DmTensor<bf16, Chip, Cluster, m![H / 8, 1 # 8], m![H % 8]> = bias.to_dm(&mut device.tdma);
    let bias_dm_1: DmTensor<bf16, Chip, Cluster, m![H / 8, 1 # 8], m![H % 8, 1 # 8]> = device
        .main
        .begin(bias_dm_0.view())
        .fetch::<m![1], m![H % 8]>()
        .collect::<m![1], m![H % 8 # 16]>()
        .transpose::<m![H % 8], m![1 # 16]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let bias_dm_2: DmTensor<bf16, Chip, Cluster, m![H / 8, Dummy8], m![H % 8, 1 # 8]> = unsafe { bias_dm_1.reshape() };
    let bias_dm_3: DmTensor<bf16, Chip, Cluster, m![H], m![Dummy8 # 16]> = device
        .main
        .begin(bias_dm_2.view())
        .fetch::<m![H % 8], m![1 # 8]>()
        .switch::<m![H], m![Dummy8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Dummy8], m![1 # 16]>()
        .transpose::<m![Dummy8 / 4], m![Dummy8 % 4 # 16]>()
        .commit_trim::<m![Dummy8 % 4]>()
        .commit();

    unsafe { bias_dm_3.reshape() }
}

fn fc1_relu(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![X]>,
    weight: &HbmTensor<bf16, Chip, m![H, X]>,
    bias: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![H], m![1 # 4]> {
    let matmul = fc1_matmul(device, input, weight);
    let bias_dm_4 = fc1_bias_prepared(device, bias);

    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(matmul.view(), bias_dm_4.view())
        .fetch::<m![I], m![1 # 4]>()
        .fetch_cast::<f32>()
        .collect::<m![I], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_clip(ClipBinaryOpF32::Max, 0.0f32)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit()
}

fn fc2_matmul(
    device: &mut Device,
    input: DmTensor<bf16, Chip, Cluster, m![H], m![1 # 4]>,
    weight: &HbmTensor<bf16, Chip, m![C, H]>,
) -> DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1 # 16]> {
    let input_dm: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![H]> = fc2_input_prepared(device, input);
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![H]> = weight.to_dm(&mut device.tdma);

    let input_trf: TrfTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1], m![H]> = device
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    device
        .main
        .begin(weight_dm.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .contract_outer::<m![H / 32], m![H % 32], _, _, _>(&input_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 16]>()
        .commit()
}

fn fc2_input_prepared(
    device: &mut Device,
    input: DmTensor<bf16, Chip, Cluster, m![H], m![1 # 4]>,
) -> DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![H]> {
    device
        .main
        .begin(input.view())
        .fetch::<m![1], m![1 # 4]>()
        .switch::<m![C, 1 # 16], m![H]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![H], m![1 # 16]>()
        .transpose::<m![H / 4], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}

fn fc2_bias_prepared(
    device: &mut Device,
    bias: &HbmTensor<bf16, Chip, m![C]>,
) -> DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1 # 16]> {
    let bias_dm_0: DmTensor<bf16, Chip, Cluster, m![1 # 16, 1 # 16], m![C]> = bias.to_dm(&mut device.tdma);
    let bias_dm_1: DmTensor<bf16, Chip, Cluster, m![Dummy16, 1 # 16], m![C]> = unsafe { bias_dm_0.reshape() };
    let bias_dm_2: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![Dummy16]> = device
        .main
        .begin(bias_dm_1.view())
        .fetch::<m![C], m![1 # 4]>()
        .switch::<m![C, 1 # 16], m![Dummy16]>(SwitchConfig::InterTranspose {
            slice1: 16,
            slice0: 16,
            time0: 1,
        })
        .collect::<m![Dummy16], m![1 # 16]>()
        .transpose::<m![Dummy16 / 4], m![Dummy16 % 4 # 16]>()
        .commit_trim::<m![Dummy16 % 4]>()
        .commit();
    unsafe { bias_dm_2.reshape() }
}

fn fc2(
    device: &mut Device,
    input: DmTensor<bf16, Chip, Cluster, m![H], m![1 # 4]>,
    weight: &HbmTensor<bf16, Chip, m![C, H]>,
    bias: &HbmTensor<bf16, Chip, m![C]>,
) -> HbmTensor<bf16, Chip, m![C]> {
    let matmul = fc2_matmul(device, input, weight);
    let bias_dm = fc2_bias_prepared(device, bias);

    let logits: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1 # 16]> = device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(matmul.view(), bias_dm.view())
        .fetch::<m![I], m![1 # 4]>()
        .fetch_cast::<f32>()
        .collect::<m![I], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 16]>()
        .commit();

    logits.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn forward(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![X]>,
    fc1_weight: &HbmTensor<bf16, Chip, m![H, X]>,
    fc1_bias: &HbmTensor<bf16, Chip, m![H]>,
    fc2_weight: &HbmTensor<bf16, Chip, m![C, H]>,
    fc2_bias: &HbmTensor<bf16, Chip, m![C]>,
) -> HbmTensor<bf16, Chip, m![C]> {
    let hidden = fc1_relu(device, input, fc1_weight, fc1_bias);
    fc2(device, hidden, fc2_weight, fc2_bias)
}
