//! Minimal kernels that exercise a specific (chip, PE) device config end to end, so CI verifies the
//! `global_config` plumbing is wired for that topology (not just the common 8-PE config).
//!
//! Each kernel adds 1 on the vector engine over a DM layout sized to its device's chip x cluster x
//! slice grid, so a mis-wired config produces a mismatched layout the EDF comparison catches. The
//! value oracle (`output == input + 1`) lives in the `furiosa-opt-examples` `pe_count` tests; the
//! cross-stage `compare_edf` companions live in `npu-visa-test`.
//!
//! In-slice extent is fixed at `P / 4 % 4` = 4 i32 (16 B, 8-byte aligned) across every config; only
//! the slice count changes with the topology.

use furiosa_opt_std::prelude::*;

// One axis per config, sized to fill its device grid: total = chips x clusters x slices x 4.
//   1 PE       = 1 chip  x 1 cluster  x  64 slices x 4 =  256
//   2 PE       = 1 chip  x 1 cluster  x 128 slices x 4 =  512
//   4 PE       = 1 chip  x 1 cluster  x 256 slices x 4 = 1024
//   8 PE       = 1 chip  x 2 clusters x 256 slices x 4 = 2048
//   2 chip 8PE = 2 chips x 2 clusters x 256 slices x 4 = 4096
//   4 chip 8PE = 4 chips x 2 clusters x 256 slices x 4 = 8192
axes![P1 = 256, P2 = 512, P4 = 1024, P8 = 2048, P2C = 4096, P4C = 8192];

type Chip = m![1];

/// Adds 1 on the vector engine at a 1-PE device (1 cluster x 64 slices).
///
/// Oracle: `output == input + 1`.
#[device(chip = 1, pe = 1)]
pub fn one_pe_add(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![P1]>) -> HbmTensor<i32, Chip, m![P1]> {
    let input_dm: DmTensor<i32, Chip, m![1], m![P1 / 4], m![P1 % 4]> = input.to_dm(&mut ctx.tdma);
    let result: DmTensor<i32, Chip, m![1], m![P1 / 4], m![P1 % 4]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P1 % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P1 % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P1 % 4]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}

/// Adds 1 on the vector engine at a 2-PE device (1 cluster x 128 slices).
///
/// Oracle: `output == input + 1`.
#[device(chip = 1, pe = 2)]
pub fn two_pe_add(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![P2]>) -> HbmTensor<i32, Chip, m![P2]> {
    let input_dm: DmTensor<i32, Chip, m![1], m![P2 / 4], m![P2 % 4]> = input.to_dm(&mut ctx.tdma);
    let result: DmTensor<i32, Chip, m![1], m![P2 / 4], m![P2 % 4]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P2 % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P2 % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P2 % 4]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}

/// Adds 1 on the vector engine at a 4-PE device (1 cluster x 256 slices).
///
/// Oracle: `output == input + 1`.
#[device(chip = 1, pe = 4)]
pub fn four_pe_add(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![P4]>) -> HbmTensor<i32, Chip, m![P4]> {
    let input_dm: DmTensor<i32, Chip, m![1], m![P4 / 4], m![P4 % 4]> = input.to_dm(&mut ctx.tdma);
    let result: DmTensor<i32, Chip, m![1], m![P4 / 4], m![P4 % 4]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P4 % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P4 % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P4 % 4]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}

/// Adds 1 on the vector engine at an 8-PE device (2 clusters x 256 slices). The DM layout now
/// carries a 2-wide cluster dimension (`P8 / 1024`), unlike the single-cluster configs above.
///
/// Oracle: `output == input + 1`.
#[device(chip = 1, pe = 8)]
pub fn eight_pe_add(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![P8]>) -> HbmTensor<i32, Chip, m![P8]> {
    let input_dm: DmTensor<i32, Chip, m![P8 / 1024], m![P8 / 4 % 256], m![P8 % 4]> = input.to_dm(&mut ctx.tdma);
    let result: DmTensor<i32, Chip, m![P8 / 1024], m![P8 / 4 % 256], m![P8 % 4]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P8 % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P8 % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P8 % 4]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}

/// Adds 1 on the vector engine at a 2-chip 8-PE device (2 chips x 2 clusters x 256 slices). The DM
/// layout now carries a 2-wide chip dimension (`P2C / 2048`) above the cluster dimension, so the HBM
/// input must be chip-distributed too (a 1-chip `Broadcast` input cannot spread across chips).
///
/// Oracle: `output == input + 1`.
#[device(chip = 2, pe = 8)]
pub fn two_chip_add(
    ctx: &mut Context,
    input: &HbmTensor<i32, m![P2C / 2048], m![P2C % 2048]>,
) -> HbmTensor<i32, m![P2C / 2048], m![P2C % 2048]> {
    let input_dm: DmTensor<i32, m![P2C / 2048], m![P2C / 1024 % 2], m![P2C / 4 % 256], m![P2C % 4]> =
        input.to_dm(&mut ctx.tdma);
    let result: DmTensor<i32, m![P2C / 2048], m![P2C / 1024 % 2], m![P2C / 4 % 256], m![P2C % 4]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P2C % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P2C % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P2C % 4]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}

/// Adds 1 on the vector engine at a 4-chip 8-PE device (4 chips x 2 clusters x 256 slices), the
/// widest topology: a 4-wide chip dimension (`P4C / 2048`) over the 2-wide cluster dimension.
///
/// Oracle: `output == input + 1`.
#[device(chip = 4, pe = 8)]
pub fn four_chip_add(
    ctx: &mut Context,
    input: &HbmTensor<i32, m![P4C / 2048], m![P4C % 2048]>,
) -> HbmTensor<i32, m![P4C / 2048], m![P4C % 2048]> {
    let input_dm: DmTensor<i32, m![P4C / 2048], m![P4C / 1024 % 2], m![P4C / 4 % 256], m![P4C % 4]> =
        input.to_dm(&mut ctx.tdma);
    let result: DmTensor<i32, m![P4C / 2048], m![P4C / 1024 % 2], m![P4C / 4 % 256], m![P4C % 4]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P4C % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P4C % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P4C % 4]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}
