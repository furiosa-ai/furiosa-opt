//! MLP block (SwiGLU activation).
//!
//! PyTorch Qwen2MLP.forward():
//!   x = silu(gate_proj(x)) * up_proj(x)    # SiLU(x) = x * sigmoid(x)
//!   x = down_proj(x)
//!
//! This kernel computes gate and up projections, applies SwiGLU, then runs
//! down projection back to hidden size.

use crate::transformer::axes::{
    H,             // = 896, hidden_size = 14 * 64
    M,             // = 4864, mlp_intermediate_size
    S_decode as S, // = 128, sequence length
    X,             // = 16, broadcast
    Y,             // = 4, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// MLP (SwiGLU): normalized [S, H] → mlp_output [S, H] in DRAM.
pub(crate) fn mlp(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]>,
    gate_weight: &HbmTensor<bf16, Chip, m![M, H]>,
    up_weight: &HbmTensor<bf16, Chip, m![M, H]>,
    down_weight: &HbmTensor<bf16, Chip, m![H, M]>,
) -> HbmTensor<bf16, Chip, m![S, H]> {
    let gate_dm: DmTensor<bf16, Chip, Cluster, m![M / 76, M / 19 % 4], m![M % 19, H]> =
        gate_weight.to_dm(&mut ctx.tdma, 0x0);

    let gate_it: DmTensor<bf16, Chip, Cluster, m![M / 76, H / 224], m![M / 19 % 4, M % 19, H % 224]> = ctx
        .main
        .begin(gate_dm.view())
        .fetch::<bf16, m![M % 19, H / 224], m![H % 224]>()
        .switch::<m![M / 76, H / 224], m![M / 19 % 4, M % 19]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![M / 19 % 4, M % 19, H / 16 % 14], m![H % 16]>()
        .commit(0x10000);

    // Reshape gate_it Slice to match TRF Slice for alignment.
    let gate_it: DmTensor<bf16, Chip, Cluster, m![M / 1216, X, H / 224], m![M / 19 % 4, M % 19, H % 224]> =
        unsafe { gate_it.reshape() };

    let input_trf_second: TrfTensor<
        bf16,
        Chip,
        Cluster,
        m![M / 1216, X, H / 224],
        m![S % 8],
        m![H / 8 % 7, H / 56 % 4, S / 32, S % 8 # 16],
    > = ctx
        .sub
        .begin(input.view())
        .fetch::<bf16, m![S % 8], m![H / 8 % 7, H % 8]>()
        .switch::<m![M / 1216, X, H / 224], m![S % 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 8 })
        .collect::<m![S % 8], m![H / 8 % 7, H / 56 % 4, S / 32, S % 8 # 16]>()
        .to_trf(TrfAddress::SecondHalf);

    let input_trf_first: TrfTensor<
        bf16,
        Chip,
        Cluster,
        m![M / 1216, X, H / 224],
        m![S % 8],
        m![H / 8 % 7, H / 56 % 4, S / 32, S % 8 # 16],
    > = ctx
        .sub
        .begin(input.view())
        .fetch::<bf16, m![S % 8], m![H / 8 % 7, H % 8]>()
        .switch::<m![M / 1216, X, H / 224], m![S % 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 8 })
        .collect::<m![S % 8], m![H / 8 % 7, H / 56 % 4, S / 32, S % 8 # 16]>()
        .to_trf(TrfAddress::FirstHalf);

    let gate_second: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(gate_it.view())
        .fetch::<bf16, m![M / 19 % 4, M % 19], m![H / 16 % 14, H % 16]>()
        .collect::<m![M / 19 % 4, M % 19, H / 16 % 14], m![H % 16]>()
        .align::<m![M / 19 % 4, M % 19], m![H / 16 % 14, H % 16], _, _>(&input_trf_second)
        .contract::<m![1]>()
        .accumulate::<m![M % 19, S / 8 % 16], m![S % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![M / 19, S / 128], m![M % 19, S % 128]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 128]>()
        .commit(0x0);

    let gate_first_raw: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(gate_it.view())
        .fetch::<bf16, m![M / 19 % 4, M % 19], m![H / 16 % 14, H % 16]>()
        .collect::<m![M / 19 % 4, M % 19, H / 16 % 14], m![H % 16]>()
        .align::<m![M / 19 % 4, M % 19], m![H / 16 % 14, H % 16], _, _>(&input_trf_first)
        .contract::<m![1]>()
        .accumulate::<m![M % 19, S / 8 % 16], m![S % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![M / 19, S / 128], m![M % 19, S % 128]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 128]>()
        .commit(0x10000);

    let gate_second_vrf: VrfTensor<f32, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .sub
        .begin(gate_second.view())
        .fetch::<f32, m![M % 19], m![S % 128]>()
        .collect::<m![M % 19], m![S % 128]>()
        .to_vrf(0);

    let gate_sum: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(gate_first_raw.view())
        .fetch::<f32, m![M % 19, S / 8 % 16], m![S % 8]>()
        .collect::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![M % 19, S / 8 % 16, S / 4 % 2], m![S % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &gate_second_vrf)
        .vector_concat::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_final()
        .cast::<bf16, m![S % 8 # 16]>()
        .commit(0x10000);

    let gate_sigmoid: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(gate_sum.view())
        .fetch::<f32, m![M % 19, S / 8 % 16], m![S % 8]>()
        .collect::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S / 4 % 2, M % 19, S / 8 % 16], m![S % 4]>()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_concat::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_final()
        .cast::<bf16, m![S % 8 # 16]>()
        .commit(0x18000);

    let up_dm: DmTensor<bf16, Chip, Cluster, m![M / 76, M / 19 % 4], m![M % 19, H]> =
        up_weight.to_dm(&mut ctx.tdma, 0x20000);

    let up_it: DmTensor<bf16, Chip, Cluster, m![M / 76, H / 224], m![M / 19 % 4, M % 19, H % 224]> = ctx
        .main
        .begin(up_dm.view())
        .fetch::<bf16, m![M % 19, H / 224], m![H % 224]>()
        .switch::<m![M / 76, H / 224], m![M / 19 % 4, M % 19]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![M / 19 % 4, M % 19, H / 16 % 14], m![H % 16]>()
        .commit(0x28000);

    // Reshape up_it Slice to match TRF Slice for alignment.
    let up_it: DmTensor<bf16, Chip, Cluster, m![M / 1216, X, H / 224], m![M / 19 % 4, M % 19, H % 224]> =
        unsafe { up_it.reshape() };

    let up_second: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(up_it.view())
        .fetch::<bf16, m![M / 19 % 4, M % 19], m![H / 16 % 14, H % 16]>()
        .collect::<m![M / 19 % 4, M % 19, H / 16 % 14], m![H % 16]>()
        .align::<m![M / 19 % 4, M % 19, H / 32 % 7], m![H % 32], _, _>(&input_trf_second)
        .contract::<m![1]>()
        .accumulate::<m![M % 19, S / 8 % 16], m![S % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![M / 19, S / 128], m![M % 19, S % 128]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 128]>()
        .commit(0x20000);

    let up_first_raw: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(up_it.view())
        .fetch::<bf16, m![M / 19 % 4, M % 19], m![H / 16 % 14, H % 16]>()
        .collect::<m![M / 19 % 4, M % 19, H / 16 % 14], m![H % 16]>()
        .align::<m![M / 19 % 4, M % 19, H / 32 % 7], m![H % 32], _, _>(&input_trf_first)
        .contract::<m![1]>()
        .accumulate::<m![M % 19, S / 8 % 16], m![S % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![M / 19, S / 128], m![M % 19, S % 128]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 128]>()
        .commit(0x30000);

    let up_second_vrf: VrfTensor<f32, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .sub
        .begin(up_second.view())
        .fetch::<f32, m![M % 19], m![S % 128]>()
        .collect::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .to_vrf(0);

    let up_sum: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(up_first_raw.view())
        .fetch::<f32, m![M % 19, S / 8 % 16], m![S % 8]>()
        .collect::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![M % 19, S / 8 % 16, S / 4 % 2], m![S % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &up_second_vrf)
        .vector_concat::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_final()
        .cast::<bf16, m![S % 8 # 16]>()
        .commit(0x38000);

    let up_vrf: VrfTensor<f32, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .sub
        .begin(up_sum.view())
        .fetch::<f32, m![M % 19], m![S % 128]>()
        .collect::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .to_vrf(0);

    let gated: DmTensor<bf16, Chip, Cluster, m![M / 19, S / 128], m![M % 19, S % 128]> = ctx
        .main
        .begin(gate_sigmoid.view())
        .fetch::<f32, m![M % 19, S / 8 % 16], m![S % 8]>()
        .collect::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![M % 19, S / 8 % 16, S / 4 % 2], m![S % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_vrf)
        .vector_concat::<m![M % 19, S / 8 % 16], m![S % 8]>()
        .vector_final()
        .cast::<bf16, m![S % 8 # 16]>()
        .commit(0x0);

    let gated_hbm: HbmTensor<bf16, Chip, m![M, S]> = gated.to_hbm(&mut ctx.tdma, 0x256d000);

    let gated_retiled: DmTensor<bf16, Chip, Cluster, m![Y, M / 76], m![M % 76, S]> =
        gated_hbm.to_dm(&mut ctx.tdma, 0x9000);

    // Reshape gated activation Slice for down-projection TRF alignment.
    let gated_retiled: DmTensor<bf16, Chip, Cluster, m![H / 224, M / 152, M / 76 % 2], m![M % 76, S]> =
        unsafe { gated_retiled.reshape() };

    // Load gated activation into TRF for down projection.
    let gated_trf: TrfTensor<bf16, Chip, Cluster, m![H / 224, M / 152, M / 76 % 2], m![M % 76], m![S]> = ctx
        .sub
        .begin(gated_retiled.view())
        .fetch::<bf16, m![M % 76], m![S]>()
        .collect::<m![M % 76], m![S]>()
        .to_trf(TrfAddress::Full);
    let down_dm: DmTensor<bf16, Chip, Cluster, m![H / 224, M / 152, H / 112 % 2], m![H % 112, M % 152]> =
        down_weight.to_dm(&mut ctx.tdma, 0x30000);
    let down_it: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![H / 224, M / 152, M / 76 % 2],
        m![H / 112 % 2, H % 112, M % 76 # 80],
    > = ctx
        .main
        .begin(down_dm.view())
        .fetch::<bf16, m![H % 112, M / 76 % 2], m![M % 76]>()
        .switch::<m![H / 224, M / 152, M / 76 % 2], m![H / 112 % 2, H % 112]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![H / 112 % 2, H % 112], m![M % 76 # 80]>()
        .commit(0x34000);
    // Reorder down-projection weight tiles for contraction alignment.
    let down_transposed: DmTensor<bf16, Chip, Cluster, m![H / 224, M / 152, M / 76 % 2], m![M % 76, H % 224]> = ctx
        .main
        .begin(down_it.view())
        .fetch::<bf16, m![H / 112 % 2, H % 112], m![M % 76]>()
        .collect::<m![M % 76], m![H % 224]>()
        .commit(0x20000);

    // Compute down projection with gated activation staged in TRF.
    let down_raw: DmTensor<bf16, Chip, Cluster, m![H / 56, S / 16], m![H % 56, S % 16]> = ctx
        .main
        .begin(down_transposed.view())
        .fetch::<bf16, m![M % 76], m![H % 224]>()
        .collect::<m![M % 76], m![H % 224]>()
        .align::<m![M % 76], m![H % 224], _, _>(&gated_trf)
        .contract::<m![1]>()
        .accumulate::<m![M % 76], m![1 # 8]>(AccumulationKind::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![H / 56, S / 16], m![H % 56, S % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 16]>()
        .commit(0x8000);

    let down_result: DmTensor<bf16, Chip, Cluster, m![H / 56, S / 16], m![S % 32, H % 56]> = ctx
        .main
        .begin(down_raw.view())
        .fetch::<bf16, m![H % 56, S / 16 % 2], m![S % 16]>()
        .collect::<m![S % 32], m![H % 56]>()
        .commit(0x0);

    // Write final MLP output tensor to HBM.
    down_result.to_hbm(&mut ctx.tdma, 0x10e50000)
}
