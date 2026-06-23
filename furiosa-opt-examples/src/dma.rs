//! Kernels for DMA tests.

use furiosa_opt_std::prelude::*;

axes![A = 65536, B = 1024];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Tries to transpose element with to_dm.
#[device(chip = 1)]
pub fn invalid_hbm_to_dm(ctx: &mut Context, input: &HbmTensor<i8, Chip, m![A, B]>) -> HbmTensor<i8, Chip, m![B, A]> {
    let output_dm: DmTensor<i8, Chip, Cluster, m![B / 4], m![B % 4, A]> = input.to_dm(&mut ctx.tdma, 0x20000);

    output_dm.to_hbm(&mut ctx.tdma, 0x1000_0000)
}
