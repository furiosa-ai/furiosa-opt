//! Matrix multiplication with cluster reduce: (1024x2) * (2x1024) -> (1024x1024)
//! This example demonstrates reduction over the cluster dimension.

use furiosa_visa_std::prelude::*;

axes![A = 1024, B = 2, C = 1024, X = 32, I = 2];

type Chip = m![1];
type Cluster = m![B]; // Cluster dimension = 2

/// Multiply matrices: [A, B] * [B, C] -> [A, C]
/// where B=2 is mapped to Cluster dimension and needs to be reduced
#[device(chip = 1)]
pub fn matmul_cluster_reduce(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, Chip, m![A, B]>,
    rhs: &HbmTensor<i8, Chip, m![B, C]>,
) -> HbmTensor<i8, Chip, m![A, C]> {
    // Load lhs: [A=1024, B=2] with B mapped to Cluster
    let lhs = lhs.to_dm::<Cluster, m![A / 4], m![A % 4]>(&mut ctx.tdma, 0);

    // Load rhs: [B=2, C=1024] with B mapped to Cluster
    let rhs = rhs.to_dm::<Cluster, m![C / 4], m![C % 4]>(&mut ctx.tdma, 128 * 1024);

    // Load rhs into VRF
    let rhs_broadcasted: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4, C % 4 # 8]> = ctx
        .main
        .begin(rhs.view())
        .fetch::<i8, m![1], m![C % 4 # 8]>()
        .switch::<m![A / 4], m![C / 4]>(SwitchConfig::Broadcast01 {
            slice1: 256,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![C / 4], m![C % 4 # 32]>()
        .commit(0);
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 4], m![C / 4, C % 4 # 8]> = ctx
        .sub
        .begin(rhs_broadcasted.view())
        .fetch::<i32, m![C / 4], m![C % 4 # 8]>()
        .collect::<m![C / 4], m![C % 4 # 8]>()
        .to_vrf(0);

    // Perform elementwise mul
    // The B dimension is in Cluster, so after contraction we still have Cluster=2
    // Result will have broadcast dimension X from GAT reduction
    let mul_result: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4, C % 4, A % 4 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<i32, m![C / 4, C % 4], m![A % 4]>()
        .collect::<m![C / 4, C % 4], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)
        .vector_final()
        .cast::<i8, m![A % 4 # 32]>()
        .commit(0);

    // Now reduce over the Cluster dimension using ReduceScatter pattern
    let reduced = reduce_over_cluster(ctx, &mul_result);

    // Write back to HBM
    reduced.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Reduce over cluster axis using ReduceScatter pattern.
fn reduce_over_cluster(
    ctx: &mut Context,
    tensor: &DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4, C % 4, A % 4 # 8]>,
) -> DmTensor<i8, Chip, m![C / 512], m![A / 4], m![C / 4 % 128, C % 4, A % 4 # 8]> {
    // ReduceScatter for Cluster=2
    // Input: Cluster dimension B=2
    // Output: C / 512 promoted to Cluster dimension
    //
    // For 2-cluster ReduceScatter:
    // Step 1: Create T0 (with slice (0, 1)) and T1 (with slice(1,0) + shuffle(1,0))
    // Step 2: Add T0 + T1 to reduce cluster axis

    // Step 1a: Use ParallelCopy (via sub context with stos) to create sliced version
    // This performs asymmetric cluster slice operation
    let sliced0: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4 % 128, C % 4, A % 4 # 8]> = ctx
        .sub
        .parallel_copy_cluster_slice::<2, m![C / 512], m![1 # 2, C / 4 % 128, C % 4, A % 4 # 8], _, _, _, _, _, _>(
            tensor.view(),
            &[0, 1],
        );

    // Step 1b: Use DmaCommand to shuffle the sliced data across clusters
    // This swaps data between Cluster 0 and Cluster 1
    let sliced1: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4 % 128, C % 4, A % 4 # 8]> = ctx
        .sub
        .parallel_copy_cluster_slice::<2, m![C / 512], m![1 # 2, C / 4 % 128, C % 4, A % 4 # 8], _, _, _, _, _, _>(
            tensor.view(),
            &[1, 0],
        );

    let shuffled1: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4 % 128, C % 4, A % 4 # 8]> =
        ctx.tdma.dm_cluster_shuffle::<_, 2, _, _, _, _>(sliced1.view(), &[1, 0]);

    // Step 2: Binary add T0 + T1 to reduce the cluster dimension
    // Use interleaved fetch with I=2 to read both tensors
    let mut reduced: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4 % 128, C % 4, A % 4 # 8]> =
        unsafe { DmTensor::from_addr(0) };

    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(sliced0.view(), shuffled1.view())
        .fetch::<i32, m![C / 4 % 128, C % 4, I], m![A % 4 # 8]>()
        .collect::<m![C / 4 % 128, C % 4, I], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C / 4 % 128, C % 4, 1 # 2], m![C / 4 % 128, C % 4]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .cast::<i8, m![A % 4 # 32]>()
        .commit_view(reduced.view_mut());

    let reshaped: DmTensor<i8, Chip, m![C / 512], m![A / 4], m![C / 4 % 128, C % 4, A % 4 # 8]> =
        unsafe { reduced.reshape() };

    reshaped
}
