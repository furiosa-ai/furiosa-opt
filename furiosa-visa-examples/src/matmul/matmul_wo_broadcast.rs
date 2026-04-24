use furiosa_visa_std::prelude::*;

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
    let lhs = lhs.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, B, A % 8]>(&mut ctx.tdma, 0);
    let rhs = rhs.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, B, A % 8]>(&mut ctx.tdma, 0);
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 16], m![1], m![A / 8 % 2, B, A % 8]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<i8, m![1], m![A / 8 % 2, B, A % 8]>()
        .collect::<m![A / 8 % 2, B / 4], m![B % 4, A % 8]>()
        .to_trf(TrfAddress::Full);

    let matmul_result: DmTensor<i8, Chip, Cluster, m![1 # 256], m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<i8, m![A / 8 % 2], m![B, A % 8]>()
        .collect::<m![A / 8 % 2, B / 4], m![B % 4, A % 8]>()
        .align::<m![A / 8 % 2], m![B % 8, A % 8], _, _>(&rhs)
        .contract::<m![1]>()
        .accumulate::<m![1], m![1 # 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .cast::<i8, m![1 # 32]>()
        .commit(0);

    // write back to HBM.
    matmul_result.to_hbm(&mut ctx.tdma, 0x3000)
}
