//! Masked softmax with 7-way GQA head grouping.
//!
//! PyTorch: softmax(scores + mask, dim=-1) per attention head.
//! GQA groups G=7 heads share the same KV, so softmax runs independently per group.
//!
//! Numerically stable: max-subtract → exp → sum → div.

use crate::transformer::axes::{
    G,              // = 7, gqa_ratio = 14 / 2
    N,              // = 2, num_kv_heads
    S_prefill as S, // = 1024, query sequence length (prefill)
    T,              // = 1024, key/value sequence length
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

pub(super) fn softmax(
    ctx: &mut Context,
    qk_scores: &DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]>,
    attention_mask: &HbmTensor<i32, Chip, m![1, S, T]>,
) -> DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]> {
    let mask_dm: DmTensor<i32, Chip, Cluster, m![S / 8, N], m![S % 8, T]> = attention_mask.to_dm(&mut ctx.tdma, 0x3e00); // Load attention mask to DM.

    // Build mask branches in VRF for masked softmax.
    let _mask_vrf: VrfTensor<i32, Chip, Cluster, m![S / 8, N], m![S % 8, T]> = ctx
        .sub
        .begin(mask_dm.view())
        .fetch::<i32, m![S % 8], m![T]>()
        .collect::<m![S % 8], m![T]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Comparison([
            InputCmp::I32(InputCmpI32::Equal { boundary: 0 }),
            InputCmp::I32(InputCmpI32::True),
            InputCmp::I32(InputCmpI32::True),
            InputCmp::I32(InputCmpI32::True),
        ]))
        .vector_final()
        .to_vrf(0);
    // BranchMode::Vrf reads this branch state implicitly.

    // Pre-allocate full output tensor. Each group commits into its tile address.
    let softmax_out: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]> =
        unsafe { DmTensor::from_addr(0x8000) };

    // Copy score tensor into a working buffer for per-group processing.
    let mut qk_scores_buf: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]> =
        unsafe { DmTensor::from_addr(0x28000) };
    qk_scores.to_dm_pcopy(&mut ctx.sub, &mut qk_scores_buf);

    // Run masked softmax independently for each GQA group.

    // Base SRAM addresses for each GQA group workspace.
    const GROUP_SRAM_ADDRS: [u64; 7] = [0x8000, 0xc000, 0x10000, 0x14000, 0x18000, 0x1c000, 0x20000];
    for (g, addr) in GROUP_SRAM_ADDRS.iter().enumerate() {
        // Extract one group tile into its workspace.
        let mut group: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, 1, T]> =
            unsafe { DmTensor::from_addr(*addr) };
        qk_scores_buf
            .view()
            .tile::<m![G], 1, m![S % 8, 1, T]>(g)
            .to_dm_view_pcopy(&mut ctx.sub, group.view_mut());

        // Apply the mask, filling masked positions with a large negative value.
        let masked: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, T]> = ctx
            .main
            .begin(group.view())
            .fetch::<f32, m![S % 8], m![T]>()
            .collect::<m![S % 8], m![T]>()
            .vector_init()
            .vector_intra_slice_branch(BranchMode::Vrf)
            .vector_logic(LogicBinaryOpF32::BitAnd, -3.3895314e38f32)
            .vector_final()
            .cast::<bf16, m![T % 8 # 16]>()
            .commit(*addr);

        // Compute per-row max for numerical stability.
        let max_vrf: VrfTensor<f32, Chip, Cluster, m![S / 8, N], m![S % 8]> = ctx
            .main
            .begin(masked.view())
            .fetch::<f32, m![S % 8, T / 8 % 128], m![T % 8]>()
            .collect::<m![S % 8, T / 8 % 128], m![T % 8]>()
            .vector_init()
            .vector_intra_slice_branch(BranchMode::Unconditional)
            .vector_split::<m![S % 8, T / 8 % 128, T / 4 % 2], m![T % 4]>()
            .vector_intra_slice_reduce::<T, m![S % 8], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_pad_way8::<m![1 # 8]>()
            .vector_final()
            .to_vrf(0);

        // Compute sum(exp(x - max)) per row.
        let sum_exp_vrf: VrfTensor<f32, Chip, Cluster, m![S / 8, N], m![S % 8]> = ctx
            .main
            .begin(masked.view())
            .fetch::<f32, m![S % 8, T / 8 % 128], m![T % 8]>()
            .collect::<m![S % 8, T / 8 % 128], m![T % 8]>()
            .vector_init()
            .vector_intra_slice_branch(BranchMode::Unconditional)
            .vector_split::<m![S % 8, T / 8 % 128, T / 4 % 2], m![T % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_intra_slice_reduce::<T, m![S % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_pad_way8::<m![1 # 8]>() // TODO: use filter to move Time -> Packet
            .vector_final()
            .to_vrf(1);

        // Normalize each row to produce probabilities.
        let _softmax: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, T]> = ctx
            .main
            .begin(masked.view())
            .fetch::<f32, m![S % 8, T / 8 % 128], m![T % 8]>()
            .collect::<m![S % 8, T / 8 % 128], m![T % 8]>()
            .vector_init()
            .vector_intra_slice_branch(BranchMode::Unconditional)
            .vector_split::<m![S % 8, T / 8 % 128, T / 4 % 2], m![T % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_fp_div(
                FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode01),
                ArrayVec::new([VeBranchOperand::always(VeRhs::vrf(&sum_exp_vrf))]),
            )
            .vector_concat::<m![S % 8, T / 8 % 128], m![T % 8]>()
            .vector_final()
            .cast::<bf16, m![T % 8 # 16]>()
            .commit(*addr);
    }

    // Return the concatenated softmax output view across all groups.
    softmax_out
}
