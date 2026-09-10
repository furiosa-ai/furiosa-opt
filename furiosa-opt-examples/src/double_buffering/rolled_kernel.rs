use super::{Group, Out, Red, Tok};
use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![Tok / 8 % 2];
type Slice = m![Tok % 8 # 256];

/// Stages and contracts one weight group per rolled iteration.
#[device(chip = 1)]
pub fn rolled(
    device: &mut Device,
    activation: &HbmTensor<bf16, Chip, m![Tok, Red]>,
    weight: &HbmTensor<bf16, Chip, m![Group, Out, Red]>,
) -> HbmTensor<bf16, Chip, m![Tok, Group, Out]> {
    let activation: DmTensor<bf16, Chip, Cluster, Slice, m![Red]> = activation.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, Slice, m![Group, Out, Red]> = weight.to_dm(&mut device.tdma);
    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![Group, Out]> = DmTensor::new();

    for g in 0..Group::SIZE {
        let weight_group = weight.view().tile::<m![Group], 1, m![1 # 20, Out, Red]>(g);
        let weight_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![Out], m![Red]> = device
            .sub
            .begin(weight_group)
            .fetch::<m![Out, Red / 16], m![Red % 16]>()
            .collect::<m![Out, Red / 16], m![Red % 16]>()
            .to_trf();

        device
            .main
            .begin(activation.view())
            .fetch::<m![Red / 16], m![Red % 16]>()
            .collect::<m![Red / 16], m![Red % 16]>()
            .contract_outer::<m![Red / 32], m![Red % 32], _, _, _>(&weight_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![1]>()
            .contract_lane::<m![1], m![Out]>(LaneMode::Interleaved)
            .cast::<bf16, m![Out # 16]>()
            .commit_trim::<m![Out]>()
            .commit_view(output.view_mut().tile::<m![Group], 1, m![1 #{!} 20, Out]>(g));
    }

    let mut result = HbmTensor::<bf16, Chip, m![Tok, Group, Out]>::new();
    output.view().to_hbm_view(&mut device.tdma, result.view_mut());
    result
}
