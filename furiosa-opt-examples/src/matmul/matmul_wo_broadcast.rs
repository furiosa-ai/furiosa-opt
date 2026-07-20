use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1 # 4];
type Cluster = m![1 # 2];

/// Mulitply matrices of size \[32768\] * \[32768\] -> \[1\]
#[device(chip = 4)]
pub fn matmul_wo_broadcast(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, Chip, m![A, B]>,
    rhs: &HbmTensor<i8, Chip, m![A, B]>,
) -> HbmTensor<i8, Chip, m![1]> {
    let lhs = lhs.to_dm::<Cluster, m![A / 16], m![A % 16, B]>(&mut ctx.tdma);
    let rhs = rhs.to_dm::<Cluster, m![A / 16], m![A % 16, B]>(&mut ctx.tdma);
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 16], m![1], m![A % 16, B]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<m![1], m![A % 16, B]>()
        .collect::<m![A / 4], m![A % 4, B]>()
        .to_trf();

    let matmul_result: DmTensor<i8, Chip, Cluster, m![1 # 256], m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<m![A / 4], m![A % 4, B]>()
        .collect::<m![A / 4], m![A % 4, B]>()
        .contract_outer::<m![A / 8 % 2], m![A % 8, B], _, _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .cast::<i8, m![1 # 32]>()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // write back to HBM.
    matmul_result.to_hbm(&mut ctx.tdma)
}
