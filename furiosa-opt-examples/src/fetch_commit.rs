#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn fetch_commit_simple(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i32, m![1], m![B, A]> {
    // Element's innermost axis must match the source's innermost (B, stride 1) so that the
    // DMA tail contains the full B axis (8 × i8 = 8 bytes), satisfying min_align = 8.
    let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma, 0);

    let fetch_and_commit_tensor: DmTensor<i32, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![A / 8 % 2], m![A % 8, B]>()
        .collect::<m![A / 8 % 2, A % 8], m![B]>()
        .commit(0);

    fetch_and_commit_tensor.to_hbm(&mut ctx.tdma, 0x3000)
}
