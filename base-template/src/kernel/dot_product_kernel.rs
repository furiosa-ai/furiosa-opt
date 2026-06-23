use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![1 # 256]; // 1 active slice; m![A / 8 # 256] would distribute across all 256
pub type Time = m![1]; // No temporal iteration
pub type Lane = m![1]; // No lane parallelism

#[device(chip = 1)]
pub fn dot_product_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {
    // HBM → DM
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = lhs.to_dm(&mut ctx.tdma, 0);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = rhs.to_dm(&mut ctx.tdma, 1 << 12);

    // Sub context: load rhs into TRF (TrfAddress::Full dedicates the entire TRF to this tensor)
    let rhs: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![A]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![{ Time }, A / 16], m![A % 16]>()
        .to_trf(TrfAddress::Full);

    // Main context: stream lhs through the Contraction Engine, reduce along A
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![A / 16], m![A % 16]>()
        // Pair consecutive 32-byte flits into 64-byte packets, halving time steps (A/16 → A/32)
        .contract_outer::<m![A / 32], m![A % 32], _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>() // cast f32 accumulator output back to bf16
        .commit_trim::<m![1 # 8]>()
        .commit(1 << 13);

    // DM → HBM
    result.to_hbm(&mut ctx.tdma, 2 << 28)
}
