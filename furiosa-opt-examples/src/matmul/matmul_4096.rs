//! Matrix-vector multiplication with 4096 rows and columns.

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 4096];

type Chip = m![1];
type Cluster = m![A / 2048 % 2];

/// Multiplies a `[4096, 4096]` matrix by a `[4096]` vector.
#[device(chip = 1)]
pub fn matmul_4096(
    device: &mut Device,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B]>,
) -> HbmTensor<i8, m![1], m![A]> {
    let lhs = lhs.to_dm::<Cluster, m![A / 1024 % 2, B / 32], m![A % 1024, B % 32]>(&mut device.tdma);
    let rhs = rhs.to_dm::<Cluster, m![A / 1024 % 2, B / 32], m![B % 32]>(&mut device.tdma);
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 1024 % 2, B / 32], m![1], m![B % 32]> = device
        .sub
        .begin(rhs.view())
        .fetch::<m![1], m![B % 32]>()
        .collect::<m![1], m![B % 32]>()
        .to_trf();

    let matmul_result: DmTensor<i8, Chip, Cluster, m![A / 1024 % 2, 1 # 128], m![A / 2 % 512, A % 2 # 8]> = device
        .main
        .begin(lhs.view())
        .fetch::<m![A % 1024], m![B % 32]>()
        .collect::<m![A % 1024], m![B % 32]>()
        .contract_outer::<m![A / 2 % 512], m![A % 2, B % 32], _, _, _>(&rhs)
        .contract_packet::<m![A % 2]>()
        .contract_time::<m![A / 2 % 512]>()
        .contract_lane::<m![A / 2 % 512], m![A % 2 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 1024 % 2, 1 # 128], m![A / 2 % 512]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .cast::<i8, m![A % 2 # 32]>()
        .commit_trim::<m![A % 2 # 8]>()
        .commit();

    let mut out = HbmTensor::<i8, m![1], m![A]>::new();
    matmul_result.view().to_hbm_view(&mut device.tdma, out.view_mut());
    out
}
