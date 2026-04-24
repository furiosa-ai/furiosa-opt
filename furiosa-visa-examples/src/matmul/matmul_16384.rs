//! Matrix multiplication of size 4096 x 4096.

use furiosa_visa_std::prelude::*;

axes![A = 16384, B = 16384, C = 16384, X = 32, I = 2];

type Chip = m![1];
type Cluster = m![B / 2048 % 2];

// contraction over B=2048 axis (which is split into B/64=32 in Slice, B=64 in Element)
fn contraction_over_b_2048(
    ctx: &mut Context,
    lhs: HbmTensorView<i8, Chip, m![A, 1 # 4, B % 4096]>,
    rhs: HbmTensorView<i8, Chip, m![1 # 4, B % 4096, 1 # 32, C % 512]>,
    out: DmTensorViewMut<i8, Chip, Cluster, m![A / 2048, X], m![C % 64, A % 2048, C / 64 % 8]>,
) {
    // Load lhs into data memory.
    let lhs = lhs.to_dm::<Cluster, m![A / 64], m![A % 64, B % 2048]>(&mut ctx.tdma, 0);
    let lhs: DmTensor<i8, Chip, Cluster, m![A / 2048, B / 64 % 32], m![A % 2048, B % 64]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<i8, m![A % 64, B / 32 % 64], m![B % 32]>()
        .switch::<m![A / 2048, B / 64 % 32], m![A % 64, B / 32 % 2, A / 64 % 32]>(SwitchConfig::InterTranspose {
            slice1: 32,
            slice0: 1,
            time0: 2,
        })
        .collect::<m![A % 64, B / 32 % 2, A / 64 % 32], m![B % 32]>()
        .commit(128 * 1024);

    // Load rhs into TRF.
    // first TensorUnit execution: apply custom broadcast to match Slice shape
    let rhs = rhs.to_dm::<Cluster, m![B / 8 % 256], m![B % 8, C % 512]>(&mut ctx.tdma, 256 * 1024);
    let rhs = ctx
        .main
        .begin(rhs.view())
        .fetch::<i8, m![B % 8, C / 32 % 16], m![C % 32]>()
        // Custom broadcast needed. TODO: custom broadcast needs custom bitmap, which needs DMA/subcontext stosfr..
        .switch::<m![A / 2048, B / 64 % 32], m![B % 8, C / 32 % 16, B / 8 % 8]>(SwitchConfig::CustomBroadcast {
            ring_size: 256,
        })
        .collect::<m![B % 8, C / 32 % 16, B / 8 % 8], m![C % 32]>()
        .commit::<m![B % 64, C % 512]>(260 * 1024);
    // second TensorUnit execution: apply TransposeEngine to match Element shape
    let rhs: DmTensor<i8, Chip, Cluster, m![A / 2048, B / 64 % 32], m![C % 512, B % 64]> = ctx
        .main
        .begin(rhs.view())
        .fetch::<i8, m![C / 8 % 64, B % 64], m![C % 8]>()
        .collect::<m![C / 8 % 64, B % 64], m![C % 8 # 32]>()
        .transpose::<m![C / 8 % 64, B / 8 % 8, C % 8], m![B % 8 # 32]>()
        .commit::<m![C % 512, B % 64]>(292 * 1024);
    // Loading into TRF. TRF is 64KB per slice, 32KB when using half mode.
    let rhs: TrfTensor<i8, Chip, Cluster, m![A / 2048, B / 64 % 32], m![C / 64 % 8], m![C % 64, B % 64]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<i8, m![C % 512], m![B % 64]>()
        .collect::<m![C % 512, B % 64 / 32], m![B % 32]>()
        .to_trf(TrfAddress::Full);

    // Perform contraction.
    ctx.main
        .begin(lhs.view())
        .fetch::<i8, m![C % 64, A % 2048, B / 32 % 2], m![B % 32]>()
        .collect::<m![C % 64, A % 2048, B / 32 % 2], m![B % 32]>()
        .align::<m![C % 64, A % 2048], m![B % 64], _, _>(&rhs)
        .contract::<m![1]>()
        .accumulate::<m![C % 64, A % 2048], m![C / 64 % 8]>(AccumulationKind::Interleaved)
        // reduce engine: reduce B/64=32 and broadcast it. Element is additionally given, in order to express optional axis promotion from Element
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 2048, X], m![C % 64, A % 2048]>(InterSliceReduceOpI32::Add)
        .vector_final()
        .cast::<i8, m![C / 64 % 8 # 32]>()
        .commit_view(out);
}

/// add two contraction results which are split over B=2048 axis.
fn add_split_contractions(
    ctx: &mut Context,
    lhs: &DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C % 64, A % 2048, C / 64 % 8]>,
    rhs: &DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C % 64, A % 2048, C / 64 % 8]>,
    out: DmTensorViewMut<i8, Chip, Cluster, m![A / 2048, X], m![C % 64, A % 2048, C / 64 % 8]>,
) {
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view())
        .fetch::<i32, m![C % 64, A % 2048, I], m![C / 64 % 8]>()
        .collect::<m![C % 64, A % 2048, I], m![C / 64 % 8]>()
        // rhs will have Group 0, lhs will have Group 1
        // I=2 axis is reduced by binary add in VE
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 64, A % 2048, 1 # 2], m![C % 64, A % 2048]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .cast::<i8, m![C / 64 % 8 # 32]>()
        .commit_view(out)
}

/// Reduce over cluster axis.
fn reduce_over_cluster(
    ctx: &mut Context,
    tensor: &DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C % 64, A % 2048, C / 64 % 8]>,
) -> DmTensor<i8, Chip, m![C / 32 % 2], m![A / 2048, X], m![C % 32, A % 2048, C / 64 % 8]> {
    // ReduceScatter for Cluster=2
    // Input: Cluster dimension B/2048%2 = 2
    // Output: C/32%2 promoted to Cluster dimension
    //
    // For 2-cluster ReduceScatter:
    // Step 1: Create T0 (with slice (0, 1)) and T1 (with slice(1,0) + shuffle(1,0))
    // Step 2: Add T0 + T1 to reduce cluster axis

    // Step 1a: Use ParallelCopy (via sub context with stos) to create sliced version
    // This performs asymmetric cluster slice operation
    let sliced0: DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C / 32 % 2, C % 32, A % 2048, C / 64 % 8]> = ctx
        .sub
        .parallel_copy_cluster_slice::<2, m![C / 32 % 2], m![1 # 2, C % 32, A % 2048, C / 64 % 8], _, _, _, _, _, _>(
            tensor.view(),
            &[0, 1],
        );

    // Step 1b: Use DmaCommand to shuffle the sliced data across clusters
    // This swaps data between Cluster 0 and Cluster 1
    let sliced1: DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C / 32 % 2, C % 32, A % 2048, C / 64 % 8]> = ctx
        .sub
        .parallel_copy_cluster_slice::<2, m![C / 32 % 2], m![1 # 2, C % 32, A % 2048, C / 64 % 8], _, _, _, _, _, _>(
            tensor.view(),
            &[1, 0],
        );

    let shuffled1: DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C / 32 % 2, C % 32, A % 2048, C / 64 % 8]> =
        ctx.tdma.dm_cluster_shuffle::<_, 2, _, _, _, _>(sliced1.view(), &[1, 0]);

    // Step 2: Binary add T0 + T1 to reduce the cluster dimension
    // Use interleaved fetch with I=2 to read both tensors
    let mut reduced: DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C % 32, A % 2048, C / 64 % 8]> =
        unsafe { DmTensor::from_addr(0) };

    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(sliced0.view(), shuffled1.view())
        .fetch::<i32, m![C % 32, A % 2048, I], m![C / 64 % 8]>()
        .collect::<m![C % 32, A % 2048, I], m![C / 64 % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 32, A % 2048, 1 # 2], m![C % 32, A % 2048]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .cast::<i8, m![C / 64 % 8 # 32]>()
        .commit_view(reduced.view_mut());

    let reshaped: DmTensor<i8, Chip, m![C / 32 % 2], m![A / 2048, X], m![C % 32, A % 2048, C / 64 % 8]> =
        unsafe { reduced.reshape() };

    reshaped
}

/// Mulitply matrices of size 16384 * 16384.
#[device(chip = 1)]
pub fn matmul_16384(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B, C]>,
) -> HbmTensor<i8, m![1], m![A, C]> {
    let mut out = unsafe { HbmTensor::<i8, m![1], m![A, C]>::from_addr(0x3000) };

    // Since lhs and rhs are 256MB and don't fit in 1-chip data memory (256MB), we load them in smaller chunks.
    // First, we split AB * BC into AB * B(C%512) 32 times. We split C first because the contraction result can be stored directly to HBM.
    for i in 0..32 {
        let rhs_tile = rhs.view().tile::<m![C / 512], 1, m![B, 1 # 32, C % 512]>(i);

        // Intermediate result after contracting over B axis.
        // Reason for broadcast dimension (X=32): B/64%32 is assigned to the Slice dimension and reduced using GAT, which broadcasts the result.
        type AccDmTensor = DmTensor<i8, Chip, Cluster, m![A / 2048, X], m![C % 64, A % 2048, C / 64 % 8]>;
        let mut result0 = unsafe { AccDmTensor::from_addr(0) };
        let mut result1 = unsafe { AccDmTensor::from_addr(0) };
        let mut temp = unsafe { AccDmTensor::from_addr(0) };

        // Next, we split AB * B(C%512) into A(B%4096) * (B%4096)(C%512) 4 times and accumulate the results.
        // The reason for splitting rhs into smaller pieces is that it must fit in the smaller TRF. TRF capacity is 64KB/32KB per slice (full/half mode).
        // TODO: More advanced optimizations such as software pipelining are needed to achieve optimal performance.
        for j in 0..4 {
            let lhs_tile = lhs.view().tile::<m![B / 4096], 1, m![A, 1 # 4, B % 4096]>(j);
            let rhs_subtile = rhs_tile.tile::<m![B / 4096], 1, m![1 # 4, B % 4096, 1 # 32, C % 512]>(j);

            // Accumulate contraction results for each chunk (reduce over B=2048 axis).
            if j == 0 {
                // For the first iteration, store the result directly to result0.
                contraction_over_b_2048(ctx, lhs_tile, rhs_subtile, result0.view_mut());
            } else {
                contraction_over_b_2048(ctx, lhs_tile, rhs_subtile, temp.view_mut());

                // From the second iteration onwards, accumulate results alternating between result0 and result1.
                if j % 2 == 1 {
                    add_split_contractions(ctx, &result0, &temp, result1.view_mut());
                } else {
                    add_split_contractions(ctx, &result1, &temp, result0.view_mut());
                }
            }
        }

        let reduced = reduce_over_cluster(ctx, &result1);

        // write back to HBM.
        let out_tile = out.view_mut().tile::<m![C / 512], 1, m![A, 1 # 32, C % 512]>(i);
        reduced
            .view()
            .slice_tile::<m![X], 1, m![A / 2048, 1 # 32]>(0)
            .to_hbm_view(&mut ctx.tdma, out_tile);
    }

    out
}
