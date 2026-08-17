//! Kernels for DMA tests.

use furiosa_opt_std::prelude::*;

axes![A = 65536, B = 1024, PA = 256, PC = 5, PD = 8, PV = 2];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Copies `lhs` and `rhs` to two separate HBM outputs. Guards vISA multiple-output lowering;
/// distinct content pins tuple order `(lhs, rhs)`. `PD = 8` is DMA-aligned, so the round trip
/// stays dense.
#[device(chip = 1)]
pub fn dup_two(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![PA, PD]>,
    rhs: &HbmTensor<i32, Chip, m![PA, PD]>,
) -> (HbmTensor<i32, Chip, m![PA, PD]>, HbmTensor<i32, Chip, m![PA, PD]>) {
    let dl: DmTensor<i32, Chip, Cluster, m![PA], m![PD]> = lhs.to_dm(&mut ctx.tdma);
    let dr: DmTensor<i32, Chip, Cluster, m![PA], m![PD]> = rhs.to_dm(&mut ctx.tdma);
    (dl.to_hbm(&mut ctx.tdma), dr.to_hbm(&mut ctx.tdma))
}

/// The dense `i32` HBM shape shared by the multiple-output copy kernels, and its DM landing shape.
type DupHbm = HbmTensor<i32, Chip, m![PA, PD]>;
type DupDm = DmTensor<i32, Chip, Cluster, m![PA], m![PD]>;

/// Single-output copy: exercises the `compare_edf!` byte-image path on a bare (non-tuple) return.
#[device(chip = 1)]
pub fn dup_one(ctx: &mut Context, input: &DupHbm) -> DupHbm {
    let dm: DupDm = input.to_dm(&mut ctx.tdma);
    dm.to_hbm(&mut ctx.tdma)
}

/// Eight outputs alternating between two inputs (`a, b, a, b, ...`) so adjacent outputs differ.
/// Stresses multiple-output lowering and pins order beyond two outputs.
#[device(chip = 1)]
#[expect(clippy::type_complexity)]
pub fn dup_many(
    ctx: &mut Context,
    a: &DupHbm,
    b: &DupHbm,
) -> (DupHbm, DupHbm, DupHbm, DupHbm, DupHbm, DupHbm, DupHbm, DupHbm) {
    let da: DupDm = a.to_dm(&mut ctx.tdma);
    let db: DupDm = b.to_dm(&mut ctx.tdma);
    (
        da.to_hbm(&mut ctx.tdma),
        db.to_hbm(&mut ctx.tdma),
        da.to_hbm(&mut ctx.tdma),
        db.to_hbm(&mut ctx.tdma),
        da.to_hbm(&mut ctx.tdma),
        db.to_hbm(&mut ctx.tdma),
        da.to_hbm(&mut ctx.tdma),
        db.to_hbm(&mut ctx.tdma),
    )
}

/// Mixes both in-place output forms with a regular returned tensor. Each output copies a distinct
/// input so a permutation of the lowered output order is observable.
#[device(chip = 1)]
pub fn mixed_inplace_and_returned_outputs(
    ctx: &mut Context,
    for_ref: &DupHbm,
    for_view: &DupHbm,
    for_return: &DupHbm,
    by_ref: &mut DupHbm,
    by_view: HbmTensorViewMut<'_, i32, Chip, m![1 #{!} 2, PA, PD]>,
) -> DupHbm {
    let for_ref: DupDm = for_ref.to_dm(&mut ctx.tdma);
    let for_view: DupDm = for_view.to_dm(&mut ctx.tdma);
    let for_return: DupDm = for_return.to_dm(&mut ctx.tdma);
    for_ref.view().to_hbm_view(&mut ctx.tdma, by_ref.view_mut());
    for_view.view().to_hbm_view(&mut ctx.tdma, by_view);
    for_return.to_hbm(&mut ctx.tdma)
}

/// Regression guard for the padded-tail DMA path: an in-slice tail axis whose live size is not
/// DMA-aligned round-trips only when padded to an aligned packet. Here `PC = 5` (i32 = 20 bytes)
/// pads to `8` (32 bytes) on both the HBM and DM sides; the tail alignment pins to the Access-class
/// packet size, so the burst strides the padded 8-cell packet instead of collapsing to the 5 live
/// cells (20 bytes) and failing the `tail_size % min_align` check. The HBM side must declare the
/// padding too, because a dense `m![PA, PC]` source (DRAM stride 20) is rejected at the load.
#[device(chip = 1)]
pub fn padded_tail_alignment(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![PA, PC # 8]>,
) -> HbmTensor<i32, Chip, m![PA, PC # 8]> {
    let dm: DmTensor<i32, Chip, Cluster, m![PA], m![PC # 8]> = input.to_dm(&mut ctx.tdma);
    dm.to_hbm(&mut ctx.tdma)
}
