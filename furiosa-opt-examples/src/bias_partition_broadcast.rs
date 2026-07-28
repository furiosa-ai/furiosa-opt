//! Fills an under-filled partition to the slice count with a non-target-`Tile` `Broadcast`.

use furiosa_opt_std::prelude::*;

// `Feat = 128` is the real partition axis; `Quad = 4` is a real inner (8 bf16 bytes).
axes![Feat = 128, Quad = 4];

type Chip = m![1];
// `Feat`(128) * `Broadcast`(2) = 256, the slice count; the bare-`2` is the partition fill.
type Slice = m![Feat, 2];
type Cl = m![2];

/// Stages `[Feat, Quad]` onto a 128-wide `Slice` and writes the 256-slice DM back.
#[device(chip = 1)]
pub fn bias_partition_broadcast(
    ctx: &mut Context,
    bias: &HbmTensor<bf16, Chip, m![Feat, Quad]>,
) -> HbmTensor<bf16, Chip, m![Feat, 2, Quad]> {
    // The output keeps the `2` axis, so the read-only fill needs no write-side collapse.
    let bias_dm: DmTensor<bf16, Chip, Cl, Slice, m![Quad]> = bias.to_dm(&mut ctx.tdma);
    bias_dm.to_hbm(&mut ctx.tdma)
}
