use furiosa_opt_std::prelude::*;

axes![I = 256, J = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![I]; // Distribute output dimension across slices
pub type Time = m![J / 32]; // Temporal iterations for reduction dimension
pub type Packet = m![J % 32]; // Packet size for reduction dimension
pub type Lane = m![1];

#[device(chip = 1)]
pub fn gemv_kernel(
    ctx: &mut Context,
    matrix: &HbmTensor<bf16, Chip, m![I, J]>,
    vector: &HbmTensor<bf16, Chip, m![J]>,
) -> HbmTensor<bf16, Chip, m![I]> {
    // Move data from HBM to DM
    let matrix: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = matrix.to_dm(&mut ctx.tdma, 0);
    let vector: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = vector.to_dm(&mut ctx.tdma, 1 << 12);

    // Load vector into TRF
    let vector_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J]> = ctx
        .sub
        .begin(vector.view())
        .fetch::<m![1], m![J]>()
        // Collect Engine: split into 32-byte flits.
        .collect::<m![J / 16], m![J % 16]>()
        .to_trf(TrfAddress::Full);

    // Compute GEMV: matrix × vector
    // Key difference: `I` maps to slice (preserved), `J` gets reduced
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 4]> = ctx
        .main
        .begin(matrix.view())
        .fetch::<m![J / 16], m![J % 16]>()
        .collect::<m![J / 16], m![J % 16]>()
        .contract_outer::<Time, Packet, _, _>(&vector_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit(0);

    // Transfer result to HBM
    result.to_hbm(&mut ctx.tdma, 2 << 28)
}
