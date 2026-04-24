#![expect(clippy::type_complexity)]

use furiosa_visa_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn fetch_commit_simple(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i32, m![1], m![B, A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, B, A % 8]>(&mut ctx.tdma, 0);

    let fetch_and_commit_tensor: DmTensor<i32, Chip, Cluster, m![A / 16], m![A / 8 % 2, B, A % 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![A / 8 % 2], m![B, A % 8]>()
        .collect::<m![A / 8 % 2, B], m![A % 8]>()
        .commit(0);

    fetch_and_commit_tensor.to_hbm(&mut ctx.tdma, 0x3000)
}
