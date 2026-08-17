//! Negative fixture for the "vISA scalar operand must be a compile-time constant"
//! diagnostic (npu-visa-translate `eval_cast`).
//!
//! The kernel feeds a bare `<m![A]>::SIZE as f32` into a vISA scalar op without the
//! required `const { .. }` wrapper. The inner `<m![A]>::SIZE` const-folds, but the
//! surrounding `as f32` cast survives into the MIR-AST as `Expression::Cast`, which
//! MIR->vISA translation rejects with the const-block guidance.
//!
//! Expected to FAIL at MIR->vISA translation; the regression gate pins it via a
//! `Reasoned` snapshot entry.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
axes![A = 512];

/// Divides each element by the axis size via a bare `<m![A]>::SIZE as f32` cast (see module doc).
#[device(chip = 1)]
pub fn scalar_cast_missing_const(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![A % 2 # 4]>()
        .vector_fp_div(<m![A]>::SIZE as f32)
        .vector_widen_pad::<m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
