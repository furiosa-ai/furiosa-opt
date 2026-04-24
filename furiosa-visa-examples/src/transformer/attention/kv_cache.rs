use crate::transformer::axes::{
    C_kvcache as C, // = 1124, kv_cache_len
    D,              // = 64, head_dim
    K,              // = 128, kv_proj_size = N * D
    N,              // = 2, num_kv_heads
    T,              // = 1024, key/value sequence length
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

pub(super) fn cache_kv(
    ctx: &mut Context,
    attn_k: &HbmTensor<bf16, Chip, m![T, K]>,
    attn_v: &HbmTensor<bf16, Chip, m![T, K]>,
    k_scatter_index: &HbmTensor<i32, Chip, m![1, T]>,
    v_scatter_index: &HbmTensor<i32, Chip, m![1, T]>,
    out_k_cache: &mut HbmTensor<bf16, Chip, m![C, 1, N, D]>,
    out_v_cache: &mut HbmTensor<bf16, Chip, m![C, 1, N, D]>,
) {
    // V scatter index path.

    // Load V scatter indices from HBM to DM.
    let v_idx_dm: DmTensor<i32, Chip, Cluster, m![T / 32, 1 # 8], m![T % 32]> =
        v_scatter_index.to_dm(&mut ctx.tdma, 0x0);

    // Reorder index tiles to match the downstream scatter-friendly layout.
    let v_idx_it: DmTensor<i32, Chip, Cluster, m![T / 32, T / 4 % 8], m![T % 4 # 32]> = ctx
        .main
        .begin(v_idx_dm.view())
        .fetch::<i32, m![T / 4 % 8], m![T % 4]>()
        .switch::<m![T / 32, T / 4 % 8], m![1 # 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 8], m![T % 4]>()
        .commit(0x100);

    // Restore compact T%4 index lanes before scaling.
    let v_idx_stripped: DmTensor<i32, Chip, Cluster, m![T / 32, T / 4 % 8], m![T % 4]> = ctx
        .main
        .begin(v_idx_it.view())
        .fetch::<i32, m![1], m![T % 4]>()
        .collect::<m![1], m![T % 4]>()
        .commit(0x0);

    // Convert row indices to byte offsets (row stride = K * sizeof(bf16) = 256).
    let v_idx_scaled: DmTensor<i32, Chip, Cluster, m![T / 32, T / 4 % 8], m![T % 4 # 8]> = ctx
        .main
        .begin(v_idx_stripped.view())
        .fetch::<i32, m![1], m![T % 4]>()
        .collect::<m![1], m![T % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, 256)
        .vector_final()
        .commit::<m![T % 4 # 8]>(0x0);

    // Pack offsets into a contiguous T-major layout for DMA spill.
    let v_idx_d0b: DmTensor<i32, Chip, Cluster, m![T / 64, 1 # 16], m![T % 64]> = ctx
        .main
        .begin(v_idx_scaled.view())
        .fetch::<i32, m![1], m![T % 4]>()
        .switch::<m![T / 64, 1 # 16], m![1]>(SwitchConfig::Broadcast01 {
            slice1: 16,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1], m![T % 4]>()
        .commit(0x100);

    // Spill prepared V offsets back to HBM for dma_scatter.
    let v_idx_hbm: HbmTensor<i32, Chip, m![T]> = v_idx_d0b.to_hbm(&mut ctx.tdma, 0x10e36000);

    // Load V rows into SRAM, then reinterpret as [N, D] blocks for cache scatter.
    attn_v.to_dm::<Cluster, m![T / 16], m![T % 16, K]>(&mut ctx.tdma, 0x200);
    let v_scatter_reshaped: DmTensor<bf16, Chip, Cluster, m![T / 16], m![T % 16, N, D]> =
        unsafe { DmTensor::from_addr(0x200) };

    // Scatter V rows into cache positions selected by the prepared indices.
    v_scatter_reshaped.dma_scatter::<m![T], _, _>(&v_idx_hbm, out_v_cache, true);

    // K scatter index path.

    // Load K scatter indices from HBM to DM.
    let k_idx_dm: DmTensor<i32, Chip, Cluster, m![T / 32, 1 # 8], m![T % 32]> =
        k_scatter_index.to_dm(&mut ctx.tdma, 0x0);

    // Reorder K index tiles to the scatter-friendly layout.
    let k_idx_it: DmTensor<i32, Chip, Cluster, m![T / 32, T / 4 % 8], m![T % 4 # 32]> = ctx
        .main
        .begin(k_idx_dm.view())
        .fetch::<i32, m![T / 4 % 8], m![T % 4]>()
        .switch::<m![T / 32, T / 4 % 8], m![1 # 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 8], m![T % 4]>()
        .commit(0x100);

    // Restore compact T%4 index lanes before scaling.
    let k_idx_stripped: DmTensor<i32, Chip, Cluster, m![T / 32, T / 4 % 8], m![T % 4]> = ctx
        .main
        .begin(k_idx_it.view())
        .fetch::<i32, m![1], m![T % 4]>()
        .collect::<m![1], m![T % 4]>()
        .commit(0x0);

    // Convert K row indices to byte offsets.
    let k_idx_scaled: DmTensor<i32, Chip, Cluster, m![T / 32, T / 4 % 8], m![T % 4 # 8]> = ctx
        .main
        .begin(k_idx_stripped.view())
        .fetch::<i32, m![1], m![T % 4]>()
        .collect::<m![1], m![T % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, 256)
        .vector_final()
        .commit::<m![T % 4 # 8]>(0x0);

    // Pack K offsets into a contiguous T-major layout for DMA spill.
    let k_idx_d0b: DmTensor<i32, Chip, Cluster, m![T / 64, 1 # 16], m![T % 64]> = ctx
        .main
        .begin(k_idx_scaled.view())
        .fetch::<i32, m![1], m![T % 4]>()
        .switch::<m![T / 64, 1 # 16], m![1]>(SwitchConfig::Broadcast01 {
            slice1: 16,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1], m![T % 4]>()
        .commit(0x100);

    // Spill prepared K offsets back to HBM for dma_scatter.
    let k_idx_hbm: HbmTensor<i32, Chip, m![T]> = k_idx_d0b.to_hbm(&mut ctx.tdma, 0x10e40000);

    // Load K rows into SRAM, then reinterpret as [N, D] blocks for cache scatter.
    attn_k.to_dm::<Cluster, m![T / 4], m![T % 4, K]>(&mut ctx.tdma, 0x5e00);
    let k_scatter_reshaped: DmTensor<bf16, Chip, Cluster, m![T / 4], m![T % 4, N, D]> =
        unsafe { DmTensor::from_addr(0x5e00) };

    // Scatter K rows into cache positions selected by the prepared indices.
    k_scatter_reshaped.dma_scatter::<m![T], _, _>(&k_idx_hbm, out_k_cache, true);
}
