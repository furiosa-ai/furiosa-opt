//! Kernels for DMA tests.

use furiosa_opt_std::prelude::*;

axes![A = 65536, B = 1024, PA = 256, PC = 5];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Tries to transpose element with to_dm.
#[device(chip = 1)]
pub fn invalid_hbm_to_dm(ctx: &mut Context, input: &HbmTensor<i8, Chip, m![A, B]>) -> HbmTensor<i8, Chip, m![B, A]> {
    let output_dm: DmTensor<i8, Chip, Cluster, m![B / 4], m![B % 4, A]> = input.to_dm(&mut ctx.tdma);

    output_dm.to_hbm(&mut ctx.tdma)
}

/// The innermost in-slice tail axis `PC = 5` (i32 = 20 bytes, NOT a multiple of the DMA min-align)
/// is padded to `8` (32 bytes, aligned) on BOTH the HBM and DM sides. This compiles only because the
/// DMA tail is aligned to the padded packet: the tail-shape alignment is pinned to the Access-class
/// packet size (8), so the burst strides the padded 8-cell packet. Without that pinning the tail would
/// collapse to the 5 live cells (20 bytes) and fail the `tail_size % min_align` check. Regression guard
/// for that alignment path. (A dense `m![PA, PC]` source is rejected at the load -- its DRAM stride 20
/// is unaligned -- so the padding must be declared on the HBM side too.)
#[device(chip = 1)]
pub fn padded_tail_alignment(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![PA, PC # 8]>,
) -> HbmTensor<i32, Chip, m![PA, PC # 8]> {
    let dm: DmTensor<i32, Chip, Cluster, m![PA], m![PC # 8]> = input.to_dm(&mut ctx.tdma);
    dm.to_hbm(&mut ctx.tdma)
}
