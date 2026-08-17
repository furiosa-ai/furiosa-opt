//! Examples exercising memory-movement primitives that lack dedicated examples
//! elsewhere: DM↔DM relayout ([`DmTensor::to_dm`]) and a `commit_view` into a
//! down-padded tiled `view_mut` ([`commit_view_bottom_pad`]).
//!
//! Parallel copy and HBM chip shuffle are recognized at MIR but not translated further, so they live
//! in `npu-opt-examples::unsupported::memory_op`.

#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 256, B = 4096];
axes![Q = 896, T = 512];

/// `DmTensor::to_dm`: relayout a DM tensor's `Element` payload (DM → DM DMA copy, auto-placed).
/// The `Slice` size is preserved (`m![A]` and `m![1 # 2, A / 2]` are both 256).
#[device(chip = 1)]
pub fn dm_relayout(
    ctx: &mut Context,
    hbm: &HbmTensor<i32, m![1], m![A, B]>,
) -> HbmTensor<i32, m![1], m![A / 2, A % 2, B]> {
    let dm: DmTensor<i32, m![1], m![1 # 2], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
    let relaid: DmTensor<i32, m![1], m![1 # 2], m![1 # 2, A / 2], m![A % 2, B]> = dm.to_dm(&mut ctx.tdma);
    relaid.to_hbm(&mut ctx.tdma)
}

type QChip = m![1];
type QCluster = m![1 # 2];
type QSlice = m![T / 32 # 256];

/// `commit_view` into a tiled `view_mut` whose tile is padded back up to the
/// destination axis width as down padding (`#{!} 56`): an 8-wide chunk is
/// committed into a 56-wide destination axis, the rest down padding, so the
/// commit sequencer never writes the cells outside the chunk. Minimal
/// fetch/commit pipeline exercising that `view_mut().tile()` fills with
/// [`PaddingKind::Bottom`].
#[device(chip = 1)]
pub fn commit_view_bottom_pad(
    ctx: &mut Context,
    input_hbm: &HbmTensor<f32, QChip, m![T, Q % 56 = 8]>,
) -> HbmTensor<f32, QChip, m![T, Q % 56]> {
    let input: DmTensor<f32, QChip, QCluster, QSlice, m![T % 32, Q % 56 = 8]> = input_hbm.to_dm(&mut ctx.tdma);

    let mut result: DmTensor<f32, QChip, QCluster, QSlice, m![T % 32, Q % 56]> = DmTensor::new();

    ctx.main
        .begin(input.view())
        .fetch::<m![T % 32], m![Q % 56 = 8]>()
        .collect::<m![T % 32], m![Q % 56 = 8]>()
        .commit_trim::<m![Q % 56 = 8]>()
        .commit_view(
            result
                .view_mut()
                .tile::<m![Q % 56], 8, m![T % 32, Q % 56 = 8 #{!} 56]>(0),
        );

    result.to_hbm(&mut ctx.tdma)
}
