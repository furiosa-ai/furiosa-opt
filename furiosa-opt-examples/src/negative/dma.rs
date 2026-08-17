//! Rejection fixture for the DMA transpose restriction.

use furiosa_opt_std::prelude::*;

pub use crate::dma::{A, B};

type Chip = m![1];
type Cluster = m![1 # 2];

/// Transposes the element axes in a `to_dm`, which the DMA cannot do.
#[device(chip = 1)]
pub fn invalid_hbm_to_dm(ctx: &mut Context, input: &HbmTensor<i8, Chip, m![A, B]>) -> HbmTensor<i8, Chip, m![B, A]> {
    let output_dm: DmTensor<i8, Chip, Cluster, m![B / 4], m![B % 4, A]> = input.to_dm(&mut ctx.tdma);

    output_dm.to_hbm(&mut ctx.tdma)
}
