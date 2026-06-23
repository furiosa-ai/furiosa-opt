use furiosa_opt_std::prelude::*;

axes![I = 512, J = 512, K = 64];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
// Distribute output dimensions `I` and `J` across slices
pub type Slice = m![I / 32, J / 32]; // Each slice handles a 16 × 16 output tile
pub type Lane = m![J % 8];

#[device(chip = 1)]
pub fn gemm_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![I, K]>,
    b: &HbmTensor<bf16, Chip, m![J, K]>,
) -> HbmTensor<bf16, Chip, m![I, J]> {
    // Move data from HBM to DM
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![I % 32, K]> = a.to_dm(&mut ctx.tdma, 0);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![J % 32, K]> = b.to_dm(&mut ctx.tdma, 1 << 12);

    // Load matrix B into TRF
    // Switch Engine distributes B across 256 slices
    // Each slice gets the full `K` dimension but only its (16 × 16) output tile
    // See: Switch Engine topologies for details on distribution
    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J / 8 % 4, K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![J % 8, J / 8 % 4], m![K]>()
        .collect::<m![J % 8, J / 8 % 4, K / 16], m![K % 16]>()
        .to_trf(TrfAddress::Full);

    // Compute GEMM: A × B
    // Switch Engine ensures matching (`I / 32`, `J / 32`) slice distribution
    // Contraction reduces along `K`, preserves `I` and `J`
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![I % 32, J % 32]> = ctx
        .main
        .begin(a.view())
        .fetch::<m![I % 32, J / 8 % 4], m![K]>()
        .collect::<m![I % 32, J / 8 % 4, K / 16], m![K % 16]>()
        .contract_outer::<m![I % 32, J / 8 % 4, K / 32], m![K % 32], _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![I % 32, J / 8 % 4]>()
        .contract_lane::<m![I % 32, J / 8 % 4], m![J % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![J % 8 # 16]>()
        .commit_trim::<m![J % 8]>()
        .commit(0);

    // Transfer result to HBM
    result.to_hbm(&mut ctx.tdma, 2 << 28)
}
