//! Rejection fixtures for excessive consecutive blocking flits in an `InterTranspose` fetch.

use furiosa_opt_std::prelude::*;

axes![
    Dim1 = 2,
    Swap = 2,
    Accumulate = 2,
    Chunk = 3,
    PacketNine = 144,
    PacketThree = 48,
    OutputLane = 8
];

type Chip = m![1];
type Cluster = m![1 # 2];
type InSlice = m![1 # 128, Dim1];
type OutSlice = m![1 # 128, Swap];
type ReducedSlice = m![1 # 256];

#[device(chip = 1)]
pub fn intertranspose_blocking_flits_18_dim1_2_fpp9(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Dim1, Accumulate, Swap, PacketNine]>,
    weight: &HbmTensor<bf16, Chip, m![OutputLane, PacketNine]>,
) -> HbmTensor<bf16, Chip, m![Dim1, PacketNine / 16, OutputLane]> {
    let input = input.to_dm::<Cluster, InSlice, m![Accumulate, Swap, PacketNine]>(&mut device.tdma);
    let weight = weight.to_dm::<Cluster, OutSlice, m![OutputLane, PacketNine]>(&mut device.tdma);
    let weight: TrfTensor<bf16, Chip, Cluster, OutSlice, m![OutputLane], m![PacketNine]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![OutputLane, PacketNine / 16], m![PacketNine % 16]>()
        .collect::<m![OutputLane, PacketNine / 16], m![PacketNine % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, ReducedSlice, m![Dim1, PacketNine / 16, OutputLane]> = device
        .main
        .begin(input.view())
        .fetch::<m![Accumulate, Swap], m![PacketNine]>()
        .switch::<OutSlice, m![Accumulate, Dim1]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Accumulate, Dim1, PacketNine / 16], m![PacketNine % 16]>()
        .contract_outer::<m![Accumulate, Dim1, PacketNine / 16], m![PacketNine % 16], _, _, _>(&weight)
        .contract_packet::<m![1]>()
        .contract_time::<m![Dim1, PacketNine / 16]>()
        .contract_lane::<m![Dim1, PacketNine / 16], m![OutputLane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<ReducedSlice, m![Dim1, PacketNine / 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![OutputLane # 16]>()
        .commit_trim::<m![OutputLane]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn intertranspose_blocking_flits_18_dim0_chunk_3_fpp3(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Dim1, Accumulate, Chunk, Swap, PacketThree]>,
    weight: &HbmTensor<bf16, Chip, m![OutputLane, PacketThree]>,
) -> HbmTensor<bf16, Chip, m![Dim1, Chunk, PacketThree / 16, OutputLane]> {
    let input = input.to_dm::<Cluster, InSlice, m![Accumulate, Chunk, Swap, PacketThree]>(&mut device.tdma);
    let weight = weight.to_dm::<Cluster, OutSlice, m![OutputLane, PacketThree]>(&mut device.tdma);
    let weight: TrfTensor<bf16, Chip, Cluster, OutSlice, m![OutputLane], m![PacketThree]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![OutputLane, PacketThree / 16], m![PacketThree % 16]>()
        .collect::<m![OutputLane, PacketThree / 16], m![PacketThree % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, ReducedSlice, m![Dim1, Chunk, PacketThree / 16, OutputLane]> = device
        .main
        .begin(input.view())
        .fetch::<m![Accumulate, Swap, Chunk], m![PacketThree]>()
        .switch::<OutSlice, m![Accumulate, Chunk, Dim1]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 3,
        })
        .collect::<m![Accumulate, Chunk, Dim1, PacketThree / 16], m![PacketThree % 16]>()
        .contract_outer::<m![Accumulate, Chunk, Dim1, PacketThree / 16], m![PacketThree % 16], _, _, _>(&weight)
        .contract_packet::<m![1]>()
        .contract_time::<m![Chunk, Dim1, PacketThree / 16]>()
        .contract_lane::<m![Chunk, Dim1, PacketThree / 16], m![OutputLane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<ReducedSlice, m![Chunk, Dim1, PacketThree / 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![OutputLane # 16]>()
        .commit_trim::<m![OutputLane]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}
