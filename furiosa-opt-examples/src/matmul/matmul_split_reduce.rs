//! Matrix-vector multiplication with four partial contractions.

use furiosa_opt_std::prelude::*;

axes![A = 1024, B = 2048, I = 2];

type Chip = m![1];
type Cluster = m![A / 512 % 2];
type Accumulator = DmTensor<i32, Chip, Cluster, m![A / 64 % 8, 1 # 32], m![A / 4 % 16, A % 4 # 8]>;

fn add_split_contractions(device: &mut Device, lhs: &Accumulator, rhs: &Accumulator) -> Accumulator {
    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view())
        .fetch::<m![A / 4 % 16, I], m![A % 4 # 8]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 4 % 16, I], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![A / 4 % 16]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 4 # 8]>()
        .commit()
}

fn narrow_split_contractions(
    device: &mut Device,
    lhs: &Accumulator,
    rhs: &Accumulator,
) -> DmTensor<i8, Chip, Cluster, m![A / 64 % 8, 1 # 32], m![A / 4 % 16, A % 4 # 8]> {
    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view())
        .fetch::<m![A / 4 % 16, I], m![A % 4 # 8]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 4 % 16, I], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![A / 4 % 16]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .cast::<i8, m![A % 4 # 8 # 32]>()
        .commit_trim::<m![A % 4 # 8]>()
        .commit()
}

fn contraction_tile(
    device: &mut Device,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B]>,
    tile: usize,
) -> Accumulator {
    let lhs = lhs
        .view()
        .tile::<m![B / 512], 1, m![A, 1 # 4, B % 512]>(tile)
        .to_dm::<Cluster, m![A / 64 % 8, B / 16 % 32], m![A % 64, B % 16]>(&mut device.tdma);
    let rhs = rhs
        .view()
        .tile::<m![B / 512], 1, m![1 # 4, B % 512]>(tile)
        .to_dm::<Cluster, m![A / 64 % 8, B / 16 % 32], m![B % 16]>(&mut device.tdma);
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 64 % 8, B / 16 % 32], m![1], m![B % 16 # 32]> = device
        .sub
        .begin(rhs.view())
        .fetch::<m![1], m![B % 16]>()
        .fetch_cast::<i8>()
        .collect::<m![1], m![B % 16 # 32]>()
        .to_trf();

    device
        .main
        .begin(lhs.view())
        .fetch::<m![A / 2 % 32], m![A % 2, B % 16]>()
        .fetch_cast::<i8>()
        .collect::<m![A / 2 % 32], m![A % 2, B % 16]>()
        .contract_outer::<m![A / 4 % 16], m![A % 4, B % 16], _, _, _>(&rhs)
        .contract_packet::<m![A % 4]>()
        .contract_time::<m![A / 4 % 16]>()
        .contract_lane::<m![A / 4 % 16], m![A % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 64 % 8, 1 # 32], m![A / 4 % 16]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .commit_trim::<m![A % 4]>()
        .commit()
}

/// Multiplies a 1024 x 2048 matrix by a 2048-element vector using four partial contractions.
#[device(chip = 1)]
pub fn matmul_with_split_reduce(
    device: &mut Device,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B]>,
) -> HbmTensor<i8, m![1], m![A]> {
    let partial0 = contraction_tile(device, lhs, rhs, 0);
    let partial1 = contraction_tile(device, lhs, rhs, 1);
    let sum01 = add_split_contractions(device, &partial0, &partial1);
    let partial2 = contraction_tile(device, lhs, rhs, 2);
    let sum012 = add_split_contractions(device, &sum01, &partial2);
    let partial3 = contraction_tile(device, lhs, rhs, 3);
    let output = narrow_split_contractions(device, &sum012, &partial3);

    let mut out = HbmTensor::<i8, m![1], m![A]>::new();
    output.view().to_hbm_view(&mut device.tdma, out.view_mut());
    out
}
