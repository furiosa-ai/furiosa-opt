//! Rejection fixture for the vector-engine ALU conflict check: two ops on one chain that need the
//! same ALU cannot be scheduled together, and the builder must say so instead of emitting a chain
//! that silently drops one. The rest of the vector-engine coverage is the public
//! [`vector_engine`](crate::vector_engine) module.

use furiosa_opt_std::prelude::*;

pub use crate::vector_engine::A;

type Chip = m![1];
type Cluster = m![1 # 2];

/// Three fxp ops on one chain, two of which want the same ALU: `AddFxp` takes FxpAdd, `MulInt` takes
/// FxpMul, then `SubFxp` asks for FxpAdd again.
///
/// Expected: the vISA-to-LIR lowering reports `6 is not available for op Binary(SubFxp)`, which the
/// snapshot pins. Cpu panics on the same conflict, which the answer-key test pins by catching it.
#[device(chip = 1)]
pub fn ve_elementwise_fxp_chain(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 10)
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_fxp(FxpBinaryOp::SubFxp, 5)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
