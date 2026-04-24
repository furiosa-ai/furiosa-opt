//! Rotary Position Embedding (RoPE).
//!
//! This kernel applies RoPE to both K and Q projections in four stages:
//! 1. Load `position_ids`, scale them to byte offsets, and gather per-token
//!    2x2 rotation coefficients from `rope_table`.
//! 2. Reshape gathered coefficients into an execution-friendly layout and load
//!    them into TRF for reuse.
//! 3. Apply the same RoPE contraction pattern to K: multiply each 2-element
//!    rotation pair by its token-specific coefficient matrix, then flatten back
//!    to contiguous K layout.
//! 4. Apply the same contraction to Q and flatten to contiguous Q layout.

use crate::transformer::axes::{
    D,             // = 64, head_dim
    K,             // = 128, k_proj
    P,             // = 32768, max_position
    Q,             // = 896, q_proj
    R,             // = 2, rope_rot
    S_decode as S, // = 128, sequence length
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// Gather rope coefficients and apply RoPE to Q and K tensors.
///
/// The flow is:
/// - position IDs -> scaled offsets -> rope table gather,
/// - rope coefficient reshape and TRF load,
/// - K RoPE contraction and flatten,
/// - Q RoPE contraction and flatten.
pub(crate) fn rope(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![S, Q]>,
    k: &HbmTensor<bf16, Chip, m![S, K]>,
    rope_table: &HbmTensor<bf16, Chip, m![P, D / 2, R, R]>,
    position_ids: &HbmTensor<i32, Chip, m![S]>,
    out_q: &mut HbmTensor<bf16, Chip, m![S, Q]>,
    out_k: &mut HbmTensor<bf16, Chip, m![S, K]>,
) {
    // Position IDs -> rope-table gather.

    // Load position IDs into DM tiles used by the offset-scaling step.
    let pos_dm: DmTensor<i32, Chip, Cluster, m![1 # 128, S / 64], m![S % 64]> =
        position_ids.to_dm(&mut ctx.tdma, 0x2300);

    // Convert each position ID into a byte offset for rope-table row access.
    let pos_scaled: DmTensor<i32, Chip, Cluster, m![1 # 128, S / 64], m![S % 64]> = ctx
        .main
        .begin(pos_dm.view())
        .fetch::<i32, m![S / 8 % 8], m![S % 8]>()
        .collect::<m![S / 8 % 8], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, 256)
        .vector_final()
        .commit::<m![S % 64]>(0x2300);

    // Reshape offsets into the layout expected by `dma_gather` indexing.
    let pos_reshaped: DmTensor<i32, Chip, Cluster, m![1 # 128, S / 64], m![S / 32 % 2, S % 16, S / 16 % 2]> = ctx
        .main
        .begin(pos_scaled.view())
        .fetch::<i32, m![S / 32 % 2, S / 16 % 2], m![S % 16]>()
        .collect::<m![S / 32 % 2, S / 16 % 2], m![S % 16]>()
        .commit(0x2300);

    // Spill reshaped offsets to HBM and use them as gather indices.
    let pos_hbm: HbmTensor<i32, Chip, m![S]> = pos_reshaped.to_hbm(&mut ctx.tdma, 0x10e36000);

    // Gather per-position RoPE coefficients from `rope_table`.
    let rope_dm: DmTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![D % 16, R, R]> =
        rope_table.dma_gather(&pos_hbm, 0x500);

    // RoPE coefficient reshape + TRF load.

    // Move the rotation dimensions into the layout used for TRF-backed contraction.
    let rope_reshaped: DmTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![R, R, D % 16]> = ctx
        .main
        .begin(rope_dm.view())
        .fetch::<bf16, m![R, R], m![D % 16]>()
        .collect::<m![R, R], m![D % 16]>()
        .commit(0x700);

    // Load rope coefficients into TRF FirstHalf for reuse across Q and K.
    let rope_trf: TrfTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![R], m![R, D % 16]> = ctx
        .sub
        .begin(rope_reshaped.view())
        .fetch::<bf16, m![R], m![R, D % 16]>()
        .collect::<m![R], m![R, D % 16]>()
        .to_trf(TrfAddress::FirstHalf);

    // K RoPE: load K in rotation-pair layout and place it in TRF SecondHalf.
    let k_dm: DmTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![R, R, D % 16]> = k.to_dm(&mut ctx.tdma, 0x0);
    let k_trf: TrfTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![R], m![R, D % 16]> = ctx
        .sub
        .begin(k_dm.view())
        .fetch::<bf16, m![R], m![R, D % 16]>()
        .collect::<m![R], m![R, D % 16]>()
        .to_trf(TrfAddress::SecondHalf);

    // Apply RoPE to K by contracting each 2x2 coefficient matrix with K pairs.
    let k_rotated: DmTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![R, R, D % 16]> = ctx
        .main
        .begin(rope_reshaped.view())
        .fetch::<bf16, m![R, R], m![D % 16]>()
        .collect::<m![R, R], m![D % 16]>()
        .align::<m![R], m![R, D % 16], _, _>(&k_trf)
        .contract::<m![1]>()
        .accumulate::<m![R, R], m![D % 16]>(AccumulationKind::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![S, D / 16 % 2], m![R, R, D % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 16]>()
        .commit(0x0);

    // Flatten rotated K from decomposed tiles back to contiguous K layout.
    let k_d0b: DmTensor<bf16, Chip, Cluster, m![S, 1 # 2], m![K]> = ctx
        .main
        .begin(k_rotated.view())
        .fetch::<bf16, m![R, R], m![D % 16]>()
        .switch::<m![S, 1 # 2], m![R, R]>(SwitchConfig::Broadcast01 {
            slice1: 2, // D/16%2 = dim0_volume
            slice0: 1,
            time0: 1,
        })
        .collect::<m![R, R], m![D % 16]>()
        .commit(0x100);
    // Write K result to output buffer.
    k_d0b.view().to_hbm_view(&mut ctx.tdma, out_k.view_mut());

    // Q RoPE.

    // Reload Q into the layout used by the RoPE contraction path.
    let q_dm_reload: DmTensor<bf16, Chip, Cluster, m![S / 16, D % 16, D / 16 % 2], m![Q / 64, R, S % 16]> =
        q.to_dm(&mut ctx.tdma, 0x200);

    // Reorder Q so the sequence sub-axis and head-group axis match downstream access.
    let q_te: DmTensor<bf16, Chip, Cluster, m![S / 16, D % 16, D / 16 % 2], m![R, S % 16, Q / 64 # 16]> = ctx
        .main
        .begin(q_dm_reload.view())
        .fetch::<bf16, m![R, Q / 64], m![S % 16]>()
        .collect::<m![R, Q / 64], m![S % 16]>()
        .commit(0x600);

    // Move Q into [Q-head-group, rotation, packet] form for contraction.
    let q_it: DmTensor<bf16, Chip, Cluster, m![S / 16, S % 16, D / 16 % 2], m![R, D % 16, Q / 64]> = ctx
        .main
        .begin(q_te.view())
        .fetch::<bf16, m![R, S % 16], m![Q / 64 # 16]>()
        .switch::<m![S / 16, S % 16, D / 16 % 2], m![R, D % 16]>(SwitchConfig::InterTranspose {
            slice1: 16,
            slice0: 2,
            time0: 1,
        })
        .collect::<m![R, D % 16], m![Q / 64]>()
        .commit(0x200);

    // Reshape q_it Slice to match rope_trf Slice for alignment.
    let q_it: DmTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![R, D % 16, Q / 64]> = unsafe { q_it.reshape() };

    // Apply RoPE to Q with the same contraction pattern used for K.
    let q_rotated: DmTensor<bf16, Chip, Cluster, m![S, D / 16 % 2], m![Q / 64, R, D % 16]> = ctx
        .main
        .begin(q_it.view())
        .fetch::<bf16, m![Q / 64, R], m![D % 16]>()
        .collect::<m![Q / 64, R], m![D % 16]>()
        .align::<m![R], m![R, D % 16], _, _>(&rope_trf)
        .contract::<m![1]>()
        .accumulate::<m![Q / 64, R], m![D % 16]>(AccumulationKind::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![S, D / 16 % 2], m![Q / 64, R, D % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 16]>()
        .commit(0x600);

    // Flatten rotated Q from decomposed tiles back to contiguous Q layout.
    let q_d0b: DmTensor<bf16, Chip, Cluster, m![S, 1 # 2], m![Q]> = ctx
        .main
        .begin(q_rotated.view())
        .fetch::<bf16, m![Q / 64, R], m![D % 16]>()
        .switch::<m![S, 1 # 2], m![Q / 64, R]>(SwitchConfig::Broadcast01 {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Q / 64, R], m![D % 16]>()
        .commit(0xa00);
    // Write Q result to output buffer.
    q_d0b.view().to_hbm_view(&mut ctx.tdma, out_q.view_mut());
}
