use super::{Group, Out, Pairs, Red, Tok};
use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![Tok / 8 % 2];
type Slice = m![Tok % 8 # 256];

/// Exposes two weight groups per iteration for software pipelining.
#[device(chip = 1)]
pub fn software_pipelined(
    device: &mut Device,
    activation: &HbmTensor<bf16, Chip, m![Tok, Red]>,
    weight: &HbmTensor<bf16, Chip, m![Group, Out, Red]>,
) -> HbmTensor<bf16, Chip, m![Tok, Group, Out]> {
    let activation: DmTensor<bf16, Chip, Cluster, Slice, m![Red]> = activation.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, Slice, m![Group, Out, Red]> = weight.to_dm(&mut device.tdma);
    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![Group, Out]> = DmTensor::new();

    for pair in 0..Pairs::SIZE {
        let first = pair * 2;
        let second = first + 1;
        let first_weight = weight.view().tile::<m![Group], 1, m![1 # 20, Out, Red]>(first);
        let first_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![Out], m![Red]> = device
            .sub
            .begin(first_weight)
            .fetch::<m![Out, Red / 16], m![Red % 16]>()
            .collect::<m![Out, Red / 16], m![Red % 16]>()
            .to_trf();
        let second_weight = weight.view().tile::<m![Group], 1, m![1 # 20, Out, Red]>(second);
        let second_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![Out], m![Red]> = device
            .sub
            .begin(second_weight)
            .fetch::<m![Out, Red / 16], m![Red % 16]>()
            .collect::<m![Out, Red / 16], m![Red % 16]>()
            .to_trf();

        device
            .main
            .begin(activation.view())
            .fetch::<m![Red / 16], m![Red % 16]>()
            .collect::<m![Red / 16], m![Red % 16]>()
            .contract_outer::<m![Red / 32], m![Red % 32], _, _, _>(&first_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![1]>()
            .contract_lane::<m![1], m![Out]>(LaneMode::Interleaved)
            .cast::<bf16, m![Out # 16]>()
            .commit_trim::<m![Out]>()
            .commit_view(output.view_mut().tile::<m![Group], 1, m![1 #{!} 20, Out]>(first));
        device
            .main
            .begin(activation.view())
            .fetch::<m![Red / 16], m![Red % 16]>()
            .collect::<m![Red / 16], m![Red % 16]>()
            .contract_outer::<m![Red / 32], m![Red % 32], _, _, _>(&second_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![1]>()
            .contract_lane::<m![1], m![Out]>(LaneMode::Interleaved)
            .cast::<bf16, m![Out # 16]>()
            .commit_trim::<m![Out]>()
            .commit_view(output.view_mut().tile::<m![Group], 1, m![1 #{!} 20, Out]>(second));
    }

    let mut result = HbmTensor::<bf16, Chip, m![Tok, Group, Out]>::new();
    output.view().to_hbm_view(&mut device.tdma, result.view_mut());
    result
}
