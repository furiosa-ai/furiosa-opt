use furiosa_visa_std::prelude::*;

axes![X = 800, H = 256, C = 16, I = 2, Dummy8 = 8, Dummy16 = 16];

type Chip = m![1];
type Cluster = m![1 # 2];

fn fc1_matmul(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![X]>,
    weight: &HbmTensor<bf16, Chip, m![H, X]>,
) -> DmTensor<bf16, Chip, Cluster, m![H], m![1 # 16]> {
    let input_dm: DmTensor<bf16, Chip, Cluster, m![H], m![X]> = input.to_dm(&mut ctx.tdma, 0);
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![H], m![X]> = weight.to_dm(&mut ctx.tdma, 0x10000);

    let input_trf: TrfTensor<bf16, Chip, Cluster, m![H], m![1], m![X]> = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<bf16, m![1], m![X]>()
        .collect::<m![X / 16], m![X % 16]>()
        .to_trf(TrfAddress::Full);

    ctx.main
        .begin(weight_dm.view())
        .fetch::<bf16, m![X / 16], m![X % 16]>()
        .collect::<m![X / 16], m![X % 16]>()
        .align::<m![X / 32], m![X % 32], _, _>(&input_trf)
        .contract::<m![1]>()
        .accumulate::<m![1], m![1 # 8]>(AccumulationKind::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit(0x20000)
}

fn fc1_bias_prepared(
    ctx: &mut Context,
    bias: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![H], m![1 # 16]> {
    let bias_dm_0: DmTensor<bf16, Chip, Cluster, m![H / 8, 1 # 8], m![H % 8]> = bias.to_dm(&mut ctx.tdma, 0x30000);
    let bias_dm_1: DmTensor<bf16, Chip, Cluster, m![H / 8, 1 # 8], m![H % 8, 1 # 8]> = ctx
        .main
        .begin(bias_dm_0.view())
        .fetch::<bf16, m![1], m![H % 8]>()
        .collect::<m![1], m![H % 8 # 16]>()
        .transpose::<m![H % 8], m![1 # 16]>()
        .commit(0x30000);
    let bias_dm_2: DmTensor<bf16, Chip, Cluster, m![H / 8, Dummy8], m![H % 8, 1 # 8]> = unsafe { bias_dm_1.reshape() };
    let bias_dm_3: DmTensor<bf16, Chip, Cluster, m![H], m![Dummy8 # 16]> = ctx
        .main
        .begin(bias_dm_2.view())
        .fetch::<bf16, m![H % 8], m![1 # 8]>()
        .switch::<m![H], m![Dummy8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Dummy8], m![1 # 16]>()
        .transpose::<m![1], m![Dummy8 # 16]>()
        .commit(0x20000);

    unsafe { bias_dm_3.reshape() }
}

fn fc1_relu(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![X]>,
    weight: &HbmTensor<bf16, Chip, m![H, X]>,
    bias: &HbmTensor<bf16, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    let matmul = fc1_matmul(ctx, input, weight);
    let bias_dm_4 = fc1_bias_prepared(ctx, bias);

    let activated: DmTensor<bf16, Chip, Cluster, m![H], m![1 # 4]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(matmul.view(), bias_dm_4.view())
        .fetch::<f32, m![I], m![1 # 2]>()
        .collect::<m![I], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_clip(ClipBinaryOpF32::Max, 0.0f32)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit(0x20000);

    activated.to_hbm(&mut ctx.tdma, 0x1000_0000)
}

fn fc2_matmul(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![C, H]>,
) -> DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1 # 16]> {
    let input_dm: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![H]> = input.to_dm(&mut ctx.tdma, 0x40000);
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![H]> = weight.to_dm(&mut ctx.tdma, 0x50000);

    let input_trf: TrfTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1], m![H]> = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<bf16, m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf(TrfAddress::Full);

    ctx.main
        .begin(weight_dm.view())
        .fetch::<bf16, m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .align::<m![H / 32], m![H % 32], _, _>(&input_trf)
        .contract::<m![1]>()
        .accumulate::<m![1], m![1 # 8]>(AccumulationKind::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit(0x60000)
}

fn fc2_bias_prepared(
    ctx: &mut Context,
    bias: &HbmTensor<bf16, Chip, m![C]>,
) -> DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1 # 16]> {
    let bias_dm_0: DmTensor<bf16, Chip, Cluster, m![1 # 16, 1 # 16], m![C]> = bias.to_dm(&mut ctx.tdma, 0x70000);
    let bias_dm_1: DmTensor<bf16, Chip, Cluster, m![Dummy16, 1 # 16], m![C]> = unsafe { bias_dm_0.reshape() };
    let bias_dm_2: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![Dummy16 # 16]> = ctx
        .main
        .begin(bias_dm_1.view())
        .fetch::<bf16, m![C], m![1 # 4]>()
        .switch::<m![C, 1 # 16], m![Dummy16]>(SwitchConfig::InterTranspose {
            slice1: 16,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Dummy16], m![1 # 16]>()
        .transpose::<m![1], m![Dummy16 # 16]>()
        .commit(0x71000);
    unsafe { bias_dm_2.reshape() }
}

fn fc2(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![C, H]>,
    bias: &HbmTensor<bf16, Chip, m![C]>,
) -> HbmTensor<bf16, Chip, m![C]> {
    let matmul = fc2_matmul(ctx, input, weight);
    let bias_dm = fc2_bias_prepared(ctx, bias);

    let logits: DmTensor<bf16, Chip, Cluster, m![C, 1 # 16], m![1 # 16]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(matmul.view(), bias_dm.view())
        .fetch::<f32, m![I], m![1 # 2]>()
        .collect::<m![I], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit(0x60000);

    logits.to_hbm(&mut ctx.tdma, 0x1100_0000)
}

#[device(chip = 1)]
pub fn forward(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![X]>,
    fc1_weight: &HbmTensor<bf16, Chip, m![H, X]>,
    fc1_bias: &HbmTensor<bf16, Chip, m![H]>,
    fc2_weight: &HbmTensor<bf16, Chip, m![C, H]>,
    fc2_bias: &HbmTensor<bf16, Chip, m![C]>,
) -> HbmTensor<bf16, Chip, m![C]> {
    let hidden = fc1_relu(ctx, input, fc1_weight, fc1_bias);
    fc2(ctx, &hidden, fc2_weight, fc2_bias)
}
