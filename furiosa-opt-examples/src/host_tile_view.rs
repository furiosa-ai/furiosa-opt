//! Repro for a host-side tile whose offset the Npu backend drops: a view reaches the device as its
//! address alone, so a tile recorded only in the mapping arrives as tile 0.

use furiosa_opt_std::prelude::*;

axes![C = 64, H = 64];

const _: () = assert!(C::SIZE == 64);

pub type Chip = m![1];

/// One row of the `[C, H]` table: a single live `C` position, the other 63 as padding.
pub type SmallRow = m![1 # 64, H];

/// [`SmallRow`] as a write destination, where the out-of-tile cells must be down (Bottom) padding so
/// the commit sequencer leaves them unwritten.
pub type SmallRowMut = m![1 #{!} 64, H];

/// Moves one row between two tables, both ends tiled by the caller on the host, so one kernel
/// covers both conversions `launch` performs and so both `address` implementations.
#[device(chip = 1)]
pub fn tile_move(
    ctx: &mut Context,
    input: HbmTensorView<'_, bf16, Chip, SmallRow>,
    out: HbmTensorViewMut<'_, bf16, Chip, SmallRowMut>,
) {
    input.to_hbm_view(&mut ctx.tdma, out);
}
