//! Matrix multiplication of size 4096 x 4096.

use furiosa_visa_std::prelude::*;

axes![A = 4096, B = 4096, X = 128];

type Chip = m![1];
type Cluster = m![A / 2048 % 2];

/// Mulitply matrices of size \[4096, 4096\] * \[4096\] -> \[4096\]
#[device(chip = 1)]
pub fn matmul_4096(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B]>,
) -> HbmTensor<i8, m![1], m![A]> {
    let lhs = lhs.to_dm::<Cluster, m![A / 1024 % 2, B / 32], m![A % 1024, B % 32]>(&mut ctx.tdma, 0);
    let rhs = rhs.to_dm::<Cluster, m![A / 1024 % 2, B / 32], m![B % 32]>(&mut ctx.tdma, 0);
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 1024 % 2, B / 32], m![1], m![B % 32]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<i8, m![1], m![B % 32]>()
        .collect::<m![1], m![B % 32]>()
        .to_trf(TrfAddress::Full);

    let matmul_result: DmTensor<i8, Chip, Cluster, m![A / 1024 % 2, X], m![A / 2 % 512, A % 2 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<i8, m![A % 1024], m![B % 32]>()
        .collect::<m![A % 1024], m![B % 32]>()
        .align::<m![A / 2 % 512], m![A % 2, B % 32], _, _>(&rhs)
        .contract::<m![A % 2]>()
        .accumulate::<m![A / 2 % 512], m![A % 2 # 8]>(AccumulationKind::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 1024 % 2, X], m![A / 2 % 512]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .cast::<i8, m![A % 2 # 32]>()
        .commit(0);

    // write back to HBM.
    let mut out = unsafe { HbmTensor::<i8, m![1], m![A]>::from_addr(0x3000) };
    matmul_result
        .view()
        .slice_tile::<m![X], 1, m![A / 1024 % 2, 1 # 128]>(0)
        .to_hbm_view(&mut ctx.tdma, out.view_mut());
    out
}
