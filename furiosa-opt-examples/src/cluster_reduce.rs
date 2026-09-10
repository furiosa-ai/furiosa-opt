//! Cluster-axis reductions and their redistribution steps.

#![expect(clippy::type_complexity)]

// ANCHOR: axes
use furiosa_opt_std::prelude::*;

axes![A = 2, B = 2, C = 8, D = 8, I = 2];
// ANCHOR_END: axes

// ANCHOR: add_pair
/// Adds two `[C, D]` DM tensors with the Vector Engine.
pub fn add_pair(
    device: &mut Device,
    lhs: &DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]>,
    rhs: &DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]>,
    output: &mut DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]>,
) {
    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view())
        .fetch::<m![C, I], m![D]>()
        .collect::<m![C, I], m![D]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![D]>()
        .commit_view(output.view_mut());
}
// ANCHOR_END: add_pair

fn reduce_scatter_dm(
    device: &mut Device,
    full: &DmTensor<i32, m![1], m![A], m![1 # 256], m![B, C, D]>,
) -> DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]> {
    let local = full.asymmetric_cluster_slice::<m![B], m![C, D]>(&mut device.sub, &[0, 1]);
    let remote = full.asymmetric_cluster_slice::<m![B], m![C, D]>(&mut device.sub, &[1, 0]);
    let remote = remote.cluster_swap().to_dm::<m![C, D]>(&mut device.tdma);

    let mut shard = DmTensor::new();
    add_pair(device, &remote, &local, &mut shard);
    shard
}

fn all_gather_dm<ShardDimension: M, OutputElement: M>(
    dma: &mut DmaContext<{ Dma::Tensor }>,
    shard: &DmTensor<i32, m![1], ShardDimension, m![1 # 256], m![C, D]>,
) -> DmTensor<i32, m![1], m![2], m![1 # 256], OutputElement> {
    shard.to_dm::<m![1], m![2], m![1 # 256], OutputElement>(dma)
}

// ANCHOR: reduce_scatter
/// Cluster `i` keeps shard `i` of the sum over both clusters.
#[device(chip = 1)]
pub fn reduce_scatter(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![1], m![A, B, C, D]>,
) -> HbmTensor<i32, m![1], m![A, C, D]> {
    let full: DmTensor<i32, m![1], m![A], m![1 # 256], m![B, C, D]> = hbm.to_dm(&mut device.tdma);
    reduce_scatter_dm(device, &full).to_hbm(&mut device.tdma)
}
// ANCHOR_END: reduce_scatter

// ANCHOR: all_gather
/// Both clusters receive every shard along `A`.
#[device(chip = 1)]
pub fn all_gather(device: &mut Device, hbm: &HbmTensor<i32, m![1], m![A, C, D]>) -> HbmTensor<i32, m![1], m![A, C, D]> {
    let shard: DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]> = hbm.to_dm(&mut device.tdma);
    all_gather_dm::<m![A], m![A, C, D]>(&mut device.tdma, &shard).to_hbm(&mut device.tdma)
}
// ANCHOR_END: all_gather

// ANCHOR: all_gather_after_reduce_scatter
/// Composes ReduceScatter and AllGather so both clusters receive the reduced shards.
#[device(chip = 1)]
pub fn all_gather_after_reduce_scatter(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![1], m![A, B, C, D]>,
) -> HbmTensor<i32, m![1], m![B, C, D]> {
    let full: DmTensor<i32, m![1], m![A], m![1 # 256], m![B, C, D]> = hbm.to_dm(&mut device.tdma);
    let shard = reduce_scatter_dm(device, &full);
    // Target cluster b owns coordinate b, so changing the axis name preserves wire order.
    let shard = unsafe { shard.reshape::<m![1], m![B], m![1 # 256], m![C, D]>() };
    all_gather_dm::<m![B], m![B, C, D]>(&mut device.tdma, &shard).to_hbm(&mut device.tdma)
}
// ANCHOR_END: all_gather_after_reduce_scatter

// ANCHOR: all_reduce
/// Adds the local and swapped cluster partials so both clusters receive the full reduction.
#[device(chip = 1)]
pub fn all_reduce(device: &mut Device, hbm: &HbmTensor<i32, m![1], m![A, C, D]>) -> HbmTensor<i32, m![1], m![A, C, D]> {
    let partials: DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]> = hbm.to_dm(&mut device.tdma);
    let remote = partials.cluster_swap().to_dm::<m![C, D]>(&mut device.tdma);

    let mut sum: DmTensor<i32, m![1], m![A], m![1 # 256], m![C, D]> = DmTensor::new();
    add_pair(device, &partials, &remote, &mut sum);
    sum.to_hbm(&mut device.tdma)
}
// ANCHOR_END: all_reduce
