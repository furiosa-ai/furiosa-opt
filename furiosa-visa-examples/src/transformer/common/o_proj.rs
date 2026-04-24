//! Attention output projection.
//!
//! PyTorch: output = o_proj(attn_output)  — Linear [H=896 → H=896, bias=False]
//!
//! DPE: input [S=128, H=896] is smaller → TRF FirstHalf; weight [896, 896] streams from DM.
//! Uses 112-width tiling (H/112=8, GReduceAdd ratio=8), vs embedding’s 56-width.

use crate::transformer::axes::{
    H,             // = 896, hidden_size = 14 * 64
    S_decode as S, // = 128, sequence length
    X,             // = 16, broadcast
    Z,             // = 2, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// O-projection: `attn_output × O_proj_weight → [S, H]`.
///
/// Reversed matmul: O_proj weight [896,896] is too large for TRF FirstHalf.
/// The kernel places input in TRF FirstHalf and streams weight from SRAM.
pub(crate) fn o_proj(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![S, H]>,
    weight: &HbmTensor<bf16, Chip, m![H, H]>,
) -> HbmTensor<bf16, Chip, m![S, H]> {
    // Load input to SRAM with 112-wide hidden tiling.
    let input_dm: DmTensor<bf16, Chip, Cluster, m![Z, H / 112, S / 8], m![S % 8, H % 112]> =
        input.to_dm(&mut ctx.tdma, 0xbb00);

    // Load projection weights to SRAM.
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![H / 28, H / 448, H / 7 % 4], m![H % 7, H % 448]> =
        weight.to_dm(&mut ctx.tdma, 0x8900);

    // Reorder weight tiles for contraction over 112 hidden channels.
    let weight_it: DmTensor<bf16, Chip, Cluster, m![H / 28, H / 448, H / 112 % 4], m![H / 7 % 4, H % 7, H % 112]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<bf16, m![H % 7, H / 112 % 4], m![H % 112]>()
        .switch::<m![H / 28, H / 448, H / 112 % 4], m![H / 7 % 4, H % 7]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![H / 7 % 4, H % 7], m![H % 112]>()
        .commit(0xa200);

    // Stage input in TRF FirstHalf for reversed matmul ordering.
    let input_trf: TrfTensor<
        bf16,
        Chip,
        Cluster,
        m![H / 448, X, H / 112],
        m![S % 8],
        m![H / 16 % 7, S / 8, S % 8 # 16],
    > = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<bf16, m![S % 8], m![H % 112]>()
        .switch::<m![H / 448, X, H / 112], m![S % 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 8 })
        .collect::<m![S % 8], m![H / 16 % 7, S / 8, S % 8 # 16]>()
        .to_trf(TrfAddress::FirstHalf);

    // Reshape weight Slice to match TRF Slice for alignment.
    let weight_it: DmTensor<bf16, Chip, Cluster, m![H / 448, X, H / 112], m![H / 7 % 4, H % 7, H % 112]> =
        unsafe { weight_it.reshape() };

    // Contract weights with TRF-staged input and reduce across hidden tiles.
    let result: DmTensor<bf16, Chip, Cluster, m![H / 7, S / 64], m![H % 7, S % 64]> = ctx
        .main
        .begin(weight_it.view())
        .fetch::<bf16, m![H / 7 % 4, H % 7], m![H / 16 % 7, H % 16]>()
        .collect::<m![H / 7 % 4, H % 7], m![H / 16 % 7, H % 16]>()
        .align::<m![H / 7 % 4, H % 7], m![H / 16 % 7, H % 16], _, _>(&input_trf)
        .contract::<m![1]>()
        .accumulate::<m![H % 7, S / 8 % 8], m![S % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![H / 7, S / 64], m![H % 7, S % 64]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 64]>()
        .commit(0x8400);

    // Store projected output to HBM.
    result.to_hbm(&mut ctx.tdma, 0x10e36000)
}
