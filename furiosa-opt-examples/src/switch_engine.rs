//! Switch Engine examples and tests.

use furiosa_opt_std::prelude::*;

axes![A = 64, B = 8, V = 16, Y = 4];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![A, 1 # 4];
type OutSlice = m![A, Y];

#[device(chip = 1)]
pub fn custom_broadcast(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, B, V]>,
) -> HbmTensor<bf16, Chip, m![A, Y, B, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![B, V]> = input.to_dm::<Cluster, Slice, m![B, V]>(&mut ctx.tdma, 0);

    let result: DmTensor<bf16, Chip, Cluster, OutSlice, m![B, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<OutSlice, m![B]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![B], m![V]>()
        .commit_trim::<m![V]>()
        .commit(0x1000);

    result.to_hbm::<m![A, Y, B, V]>(&mut ctx.tdma, 0x4000)
}
