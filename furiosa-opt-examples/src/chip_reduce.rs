//! Chip-axis reductions and their redistribution steps.

#![expect(clippy::type_complexity)]

// ANCHOR: axes
use furiosa_opt_std::prelude::*;

axes![A = 4, B = 4, C = 8, D = 8, I = 2, X = 2];
// ANCHOR_END: axes

pub const MIDDLE_TARGET_POSITION: usize = 2;

// ANCHOR: add_pair
/// Adds two `[C, D]` DM tensors with the Vector Engine.
pub fn add_pair(
    device: &mut Device,
    lhs: &DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]>,
    rhs: &DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]>,
    output: &mut DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]>,
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
    full: &DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, C, D]>,
) -> DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> {
    // Round k routes source chip (target + k) % 4 to each target chip.
    let round0 = full
        .chip_shuffle([0, 1, 2, 3])
        .chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm::<m![C, D]>(&mut device.tdma);
    let round1 = full
        .chip_shuffle([1, 2, 3, 0])
        .chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm::<m![C, D]>(&mut device.tdma);
    let round2 = full
        .chip_shuffle([2, 3, 0, 1])
        .chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm::<m![C, D]>(&mut device.tdma);
    let round3 = full
        .chip_shuffle([3, 0, 1, 2])
        .chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm::<m![C, D]>(&mut device.tdma);

    let mut sum01 = DmTensor::new();
    let mut sum012 = DmTensor::new();
    let mut shard = DmTensor::new();
    add_pair(device, &round0, &round1, &mut sum01);
    add_pair(device, &round2, &sum01, &mut sum012);
    add_pair(device, &round3, &sum012, &mut shard);
    shard
}

fn all_gather_dm<ShardDimension: M, OutputElement: M>(
    dma: &mut DmaContext<{ Dma::Tensor }>,
    shard: &DmTensor<i32, ShardDimension, m![1 # 2], m![1 # 256], m![C, D]>,
) -> DmTensor<i32, m![4], m![1 # 2], m![1 # 256], OutputElement> {
    shard.to_dm::<m![4], m![1 # 2], m![1 # 256], OutputElement>(dma)
}

// ANCHOR: reduce_scatter
/// Chip `i` keeps shard `i` of the sum over all chips.
#[device(chip = 4)]
pub fn reduce_scatter(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, C, D]>,
) -> HbmTensor<i32, m![A], m![C, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, C, D]> = hbm.to_dm(&mut device.tdma);
    reduce_scatter_dm(device, &full).to_hbm(&mut device.tdma)
}
// ANCHOR_END: reduce_scatter

// ANCHOR: all_gather
/// Every chip receives every shard along `A`.
#[device(chip = 4)]
pub fn all_gather(device: &mut Device, hbm: &HbmTensor<i32, m![A], m![C, D]>) -> HbmTensor<i32, m![4], m![A, C, D]> {
    let shard: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> = hbm.to_dm(&mut device.tdma);
    all_gather_dm::<m![A], m![A, C, D]>(&mut device.tdma, &shard).to_hbm(&mut device.tdma)
}
// ANCHOR_END: all_gather

// ANCHOR: all_gather_after_reduce_scatter
/// Composes ReduceScatter and AllGather so every chip receives the reduced shards.
#[device(chip = 4)]
pub fn all_gather_after_reduce_scatter(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, C, D]>,
) -> HbmTensor<i32, m![4], m![B, C, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, C, D]> = hbm.to_dm(&mut device.tdma);
    let shard = reduce_scatter_dm(device, &full);
    // Target chip b owns coordinate b, so changing the axis name preserves wire order.
    let shard = unsafe { shard.reshape::<m![B], m![1 # 2], m![1 # 256], m![C, D]>() };
    all_gather_dm::<m![B], m![B, C, D]>(&mut device.tdma, &shard).to_hbm(&mut device.tdma)
}
// ANCHOR_END: all_gather_after_reduce_scatter

// ANCHOR: all_reduce
/// Sums four cyclic chip rotations so every chip receives the full reduction.
#[device(chip = 4)]
pub fn all_reduce(device: &mut Device, hbm: &HbmTensor<i32, m![A], m![C, D]>) -> HbmTensor<i32, m![A], m![C, D]> {
    let round0: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> = hbm.to_dm(&mut device.tdma);
    let round1 = round0.chip_shuffle([1, 2, 3, 0]).to_dm::<m![C, D]>(&mut device.tdma);
    let round2 = round0.chip_shuffle([2, 3, 0, 1]).to_dm::<m![C, D]>(&mut device.tdma);
    let round3 = round0.chip_shuffle([3, 0, 1, 2]).to_dm::<m![C, D]>(&mut device.tdma);

    let mut sum01 = DmTensor::new();
    let mut sum012 = DmTensor::new();
    let mut sum = DmTensor::new();
    add_pair(device, &round0, &round1, &mut sum01);
    add_pair(device, &round2, &sum01, &mut sum012);
    add_pair(device, &round3, &sum012, &mut sum);
    sum.to_hbm(&mut device.tdma)
}
// ANCHOR_END: all_reduce

// ANCHOR: butterfly_all_reduce
/// Reduces across chips with XOR-paired butterfly rounds.
#[device(chip = 4)]
pub fn butterfly_all_reduce(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![C, D]>,
) -> HbmTensor<i32, m![A], m![C, D]> {
    let partials: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> = hbm.to_dm(&mut device.tdma);
    let swapped = partials.chip_shuffle([1, 0, 3, 2]).to_dm::<m![C, D]>(&mut device.tdma);
    let mut first: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> = DmTensor::new();
    add_pair(device, &swapped, &partials, &mut first);

    let swapped = first.chip_shuffle([2, 3, 0, 1]).to_dm::<m![C, D]>(&mut device.tdma);
    let mut sum: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> = DmTensor::new();
    add_pair(device, &swapped, &first, &mut sum);
    sum.to_hbm(&mut device.tdma)
}
// ANCHOR_END: butterfly_all_reduce

/// Slices an axis inside `Element`, which the ParallelCopy path cannot address.
#[device(chip = 4)]
pub fn shuffle_slice_middle(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, C, D]>,
) -> HbmTensor<i32, m![A], m![C, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, B, D]> =
        hbm.to_hbm::<_, m![C, B, D]>(&mut device.tdma).to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, D]> = DmTensor::new();
    full.chip_shuffle([1, 2, 3, 0])
        .chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm_view(&mut device.tdma, output.view_mut());
    output.to_hbm(&mut device.tdma)
}

/// Writes an inner-axis selection into one position of the source axis.
#[device(chip = 4)]
pub fn shuffle_slice_middle_place(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, C, D]>,
) -> HbmTensor<i32, m![A], m![C, B, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, B, D]> =
        hbm.to_hbm::<_, m![C, B, D]>(&mut device.tdma).to_dm(&mut device.tdma);
    let mut placed: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![C, B, D]> = DmTensor::new();
    placed.view_mut().memset(0, &mut device.sub);
    full.chip_shuffle([1, 2, 3, 0])
        .chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm_view(
            &mut device.tdma,
            placed
                .view_mut()
                .tile::<m![B], 1, m![C, 1 #{!} 4, D]>(MIDDLE_TARGET_POSITION),
        );
    placed.to_hbm(&mut device.tdma)
}

/// Keeps each chip local and selects its same-numbered `B` position.
#[device(chip = 4)]
pub fn shuffle_slice_identity(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, C, D]>,
) -> HbmTensor<i32, m![A], m![C, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, C, D]> = hbm.to_dm(&mut device.tdma);
    full.chip_slice::<m![B], m![C, D]>([0, 1, 2, 3])
        .to_dm::<m![C, D]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

/// Redistributes a source view after selecting one `Element` tile.
#[device(chip = 4)]
pub fn redistribute_element_tile(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, X, C, D]>,
) -> HbmTensor<i32, m![A], m![C, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, X, C, D]> = hbm.to_dm(&mut device.tdma);
    full.view()
        .tile::<m![X], 1, m![B, X = 1 # 2, C, D]>(1)
        .chip_slice::<m![B], m![X = 1 # 2, C, D]>([0, 1, 2, 3])
        .to_dm::<m![X = 1 # 2, C, D]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

// ANCHOR: noncontiguous_axes
/// Selects two independently located axes in one shuffle DMA.
#[device(chip = 4)]
pub fn shuffle_slice_noncontiguous_axes(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, X, C, D]>,
) -> HbmTensor<i32, m![A], m![X, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, X, C, D]> = hbm.to_dm(&mut device.tdma);
    full.chip_shuffle([1, 2, 3, 0])
        .chip_slice::<m![B], m![X, C, D]>([0, 1, 2, 3])
        .chip_slice::<m![C], m![X, D]>([3, 0, 5, 7])
        .to_dm::<m![X, D]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}
// ANCHOR_END: noncontiguous_axes

/// Keeps padding in the unsliced tail while removing the leading shard axis.
#[device(chip = 4)]
pub fn shuffle_slice_padded_tail(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A], m![B, C, D]>,
) -> HbmTensor<i32, m![A], m![C, D]> {
    let full: DmTensor<i32, m![A], m![1 # 2], m![1 # 256], m![B, C, D # 16]> = hbm.to_dm(&mut device.tdma);
    full.chip_shuffle([0, 1, 2, 3])
        .chip_slice::<m![B], m![C, D # 16]>([0, 1, 2, 3])
        .to_dm::<m![C, D # 16]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}
