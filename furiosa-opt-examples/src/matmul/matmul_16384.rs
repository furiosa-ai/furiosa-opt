//! Matrix multiplication of two 16384 x 16384 matrices.

use furiosa_opt_std::prelude::*;

axes![A = 16384, B = 16384, C = 16384, I = 2];

type Chip = m![1];
type Cluster = m![B / 2048 % 2];

// Each cluster contracts one 2048-element half of the 4096-element B tile.
fn contraction_over_b_2048(
    device: &mut Device,
    lhs: HbmTensorView<'_, i8, Chip, m![1 # 16, A % 1024, 1 # 4, B % 4096]>,
    rhs: HbmTensorView<'_, i8, Chip, m![1 # 4, B % 4096, 1 # 256, C % 64]>,
    out: DmTensorViewMut<'_, i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>,
) {
    let lhs = lhs.to_dm::<Cluster, m![A / 64 % 16, B / 128 % 16], m![A % 64, B % 128]>(&mut device.tdma);
    let lhs: DmTensor<i8, Chip, Cluster, m![A / 128 % 8, B / 64 % 2, B / 128 % 16], m![A % 128, B % 64]> = device
        .main
        .begin(lhs.view())
        .fetch::<m![A % 64, B / 32 % 4], m![B % 32]>()
        .switch::<m![A / 128 % 8, B / 64 % 2, B / 128 % 16], m![A % 64, B / 32 % 2, A / 64 % 2]>(
            SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            },
        )
        .collect::<m![A % 64, B / 32 % 2, A / 64 % 2], m![B % 32]>()
        .commit_trim::<m![B % 32]>()
        .commit();

    // Broadcast the RHS across the accumulator's slice mapping before transposing its element.
    let rhs = rhs.to_dm::<Cluster, m![B / 8 % 256], m![B % 8, C % 64]>(&mut device.tdma);
    let rhs = device
        .main
        .begin(rhs.view())
        .fetch::<m![B % 8, C / 32 % 2], m![C % 32]>()
        .fetch_cast::<i8>()
        .switch::<m![A / 128 % 8, B / 64 % 2, B / 128 % 16], m![B % 8, C / 32 % 2, B / 8 % 8]>(
            SwitchConfig::CustomBroadcast { ring_size: 256 },
        )
        .collect::<m![B % 8, C / 32 % 2, B / 8 % 8], m![C % 32]>()
        .commit_trim::<m![C % 32]>()
        .commit::<m![B % 64, C % 64]>();
    let rhs: DmTensor<i8, Chip, Cluster, m![A / 128 % 8, B / 64 % 2, B / 128 % 16], m![C % 64, B % 64]> = device
        .main
        .begin(rhs.view())
        .fetch::<m![C / 8 % 8, B % 64], m![C % 8]>()
        .collect::<m![C / 8 % 8, B % 64], m![C % 8 # 32]>()
        .transpose::<m![C / 8 % 8, B / 8 % 8, C % 8], m![B % 8 # 32]>()
        .commit_trim::<m![B % 8]>()
        .commit();
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 128 % 8, B / 64 % 2, B / 128 % 16], m![C / 8 % 8], m![C % 8, B % 64]> =
        device
            .sub
            .begin(rhs.view())
            .fetch::<m![C % 64], m![B % 64]>()
            .collect::<m![C % 64, B % 64 / 32], m![B % 32]>()
            .to_trf();

    device
        .main
        .begin(lhs.view())
        .fetch::<m![C % 8, A % 128, B / 32 % 2], m![B % 32]>()
        .collect::<m![C % 8, A % 128, B / 32 % 2], m![B % 32]>()
        .contract_outer::<m![C % 8, A % 128], m![B % 64], _, _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![C % 8, A % 128]>()
        .contract_lane::<m![C % 8, A % 128], m![C / 8 % 8]>(LaneMode::Interleaved)
        // Reduce the sixteen B chunks assigned to each cluster.
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 128 % 8, 1 # 32], m![C % 8, A % 128]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .cast::<i8, m![C / 8 % 8 # 32]>()
        .commit_trim::<m![C / 8 % 8]>()
        .commit_view(out);
}

fn contraction_tile(
    device: &mut Device,
    lhs: &HbmTensorView<'_, i8, Chip, m![1 # 16, A % 1024, B]>,
    rhs: &HbmTensorView<'_, i8, Chip, m![B, 1 # 256, C % 64]>,
    tile: usize,
    out: DmTensorViewMut<'_, i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>,
) {
    let lhs = lhs.tile::<m![B / 4096], 1, m![1 # 16, A % 1024, 1 # 4, B % 4096]>(tile);
    let rhs = rhs.tile::<m![B / 4096], 1, m![1 # 4, B % 4096, 1 # 256, C % 64]>(tile);
    contraction_over_b_2048(device, lhs, rhs, out);
}

/// Adds two partial contractions split over the B dimension.
fn add_split_contractions(
    device: &mut Device,
    lhs: &DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>,
    rhs: &DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>,
    out: DmTensorViewMut<'_, i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>,
) {
    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view())
        .fetch::<m![C % 8, A % 128, I], m![C / 8 % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C % 8, A % 128, I], m![C / 8 % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 8, A % 128]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .cast::<i8, m![C / 8 % 8 # 32]>()
        .commit_trim::<m![C / 8 % 8]>()
        .commit_view(out)
}

/// Reduces the partial contractions across the cluster dimension.
fn reduce_over_cluster(
    device: &mut Device,
    tensor: &DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>,
) -> DmTensor<i8, Chip, m![C / 4 % 2], m![A / 128 % 8, 1 # 32], m![C % 4, A % 128, C / 8 % 8]> {
    let sliced0: DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 4, A % 128, C / 8 % 8]> =
        tensor.asymmetric_cluster_slice::<m![C / 4 % 2], _>(&mut device.sub, &[0, 1]);

    let sliced1: DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 4, A % 128, C / 8 % 8]> =
        tensor.asymmetric_cluster_slice::<m![C / 4 % 2], _>(&mut device.sub, &[1, 0]);

    let shuffled1: DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 4, A % 128, C / 8 % 8]> =
        sliced1.view().cluster_swap().to_dm(&mut device.tdma);

    let mut reduced: DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 4, A % 128, C / 8 % 8]> =
        DmTensor::new();

    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(sliced0.view(), shuffled1.view())
        .fetch::<m![C % 4, A % 128, I], m![C / 8 % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C % 4, A % 128, I], m![C / 8 % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 4, A % 128]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .cast::<i8, m![C / 8 % 8 # 32]>()
        .commit_trim::<m![C / 8 % 8]>()
        .commit_view(reduced.view_mut());

    // Both mappings move the same size-2 factor from Cluster to the C dimension.
    let reshaped: DmTensor<i8, Chip, m![C / 4 % 2], m![A / 128 % 8, 1 # 32], m![C % 4, A % 128, C / 8 % 8]> =
        unsafe { reduced.reshape() };

    reshaped
}

/// Multiplies two 16384 x 16384 matrices using 1024 x 64 output tiles.
#[device(chip = 1)]
pub fn matmul_16384(
    device: &mut Device,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B, C]>,
) -> HbmTensor<i8, m![1], m![A, C]> {
    let mut out = HbmTensor::<i8, m![1], m![A, C]>::new();

    // Split C first so each contraction result fits in DM and can be stored directly to HBM.
    for i in 0..256 {
        let rhs_tile = rhs.view().tile::<m![C / 64], 1, m![B, 1 # 256, C % 64]>(i);
        for a in 0..16 {
            let lhs_tile = lhs.view().tile::<m![A / 1024], 1, m![1 # 16, A % 1024, B]>(a);

            type AccDmTensor = DmTensor<i8, Chip, Cluster, m![A / 128 % 8, 1 # 32], m![C % 8, A % 128, C / 8 % 8]>;
            let mut result0 = AccDmTensor::new();
            let mut result1 = AccDmTensor::new();
            let mut temp = AccDmTensor::new();

            // Split B into four 4096-element tiles so the right-hand operand fits in TRF.
            contraction_tile(device, &lhs_tile, &rhs_tile, 0, result0.view_mut());
            contraction_tile(device, &lhs_tile, &rhs_tile, 1, temp.view_mut());
            add_split_contractions(device, &result0, &temp, result1.view_mut());
            contraction_tile(device, &lhs_tile, &rhs_tile, 2, temp.view_mut());
            add_split_contractions(device, &result1, &temp, result0.view_mut());
            contraction_tile(device, &lhs_tile, &rhs_tile, 3, temp.view_mut());
            add_split_contractions(device, &result0, &temp, result1.view_mut());

            let reduced = reduce_over_cluster(device, &result1);

            let out_tile = out
                .view_mut()
                .tile::<m![A / 1024], 1, m![1 #{!} 16, A % 1024, C]>(a)
                .tile::<m![C / 64], 1, m![1 #{!} 16, A % 1024, 1 #{!} 256, C % 64]>(i);
            reduced.view().to_hbm_view(&mut device.tdma, out_tile);
        }
    }

    out
}
