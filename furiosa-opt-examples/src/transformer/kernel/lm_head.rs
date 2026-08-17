use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::{H, Wp};

pub(crate) type Cluster = m![Wp / 4096 % 2];
pub(crate) type Slice = m![Wp / 16 % 256];

pub(crate) fn forward(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    weight: HbmTensorView<'_, bf16, Chip, m![Wp / 8192, Wp % 8192, H]>,
    out: &mut HbmTensor<bf16, Chip, m![Wp]>,
) {
    let mut logits: DmTensor<bf16, Chip, Cluster, Slice, m![Wp / 8192, Wp % 16]> = DmTensor::new();

    for i in 0..19 {
        let weight: DmTensor<bf16, Chip, Cluster, Slice, m![Wp % 16, H]> = weight
            .tile::<m![Wp / 8192], 1, m![1 # 19, Wp % 8192, H]>(i)
            .to_dm(&mut ctx.tdma);

        let weight_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![Wp % 8], m![Wp / 8 % 2, H]> = ctx
            .sub
            .begin(weight.view())
            .fetch::<m![Wp % 8, Wp / 8 % 2, H / 16], m![H % 16]>()
            .collect::<m![Wp % 8, Wp / 8 % 2, H / 16], m![H % 16]>()
            .to_trf();

        ctx.main
            .begin(input.view())
            .fetch::<m![H / 16], m![H % 16]>()
            .collect::<m![H / 16], m![H % 16]>()
            .contract_outer::<m![H / 32, Wp / 8 % 2], m![H % 32], _, _, _>(&weight_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Wp / 8 % 2]>()
            .contract_lane::<m![Wp / 8 % 2], m![Wp % 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Wp % 8 # 16]>()
            .commit_trim::<m![Wp % 8]>()
            .commit_view(logits.view_mut().tile::<m![Wp / 8192], 1, m![1 #{!} 19, Wp % 16]>(i));
    }

    let logits: DmTensor<bf16, Chip, Cluster, m![Wp / 128 % 32, 1 # 8], m![Wp / 8192, Wp % 128]> = ctx
        .main
        .begin(logits.view())
        .fetch::<m![Wp / 8192], m![Wp % 16]>()
        .switch::<m![Wp / 128 % 32, 1 # 8], m![Wp / 8192, Wp / 16 % 8]>(SwitchConfig::Broadcast1 {
            slice1: 8,
            slice0: 1,
        })
        .collect::<m![Wp / 8192, Wp / 16 % 8], m![Wp % 16]>()
        .commit_trim::<m![Wp % 16]>()
        .commit();

    logits.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}
