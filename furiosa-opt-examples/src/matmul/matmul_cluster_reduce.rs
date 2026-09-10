//! Matrix multiplication with cluster reduce: (2048x2) * (2x2048) -> (2048x2048)
//! This example demonstrates reduction over the cluster dimension.

use furiosa_opt_std::prelude::*;

axes![A = 2048, B = 2, C = 2048, X = 32, I = 2];

type Chip = m![1];
type Cluster = m![B]; // Cluster dimension = 2

/// Multiply matrices: [A, B] * [B, C] -> [A, C]
/// where B=2 is mapped to Cluster dimension and needs to be reduced
#[device(chip = 1)]
pub fn matmul_cluster_reduce(
    device: &mut Device,
    lhs: &HbmTensor<i8, Chip, m![B, A]>,
    rhs: &HbmTensor<i8, Chip, m![B, C]>,
) -> HbmTensor<i8, Chip, m![A, C]> {
    // Keeping A contiguous lets DMA select each B matrix shard in 8-byte runs.
    let lhs = lhs.to_dm::<Cluster, m![A / 8], m![A % 8]>(&mut device.tdma);

    // Load rhs with B mapped to Cluster.
    let rhs = rhs.to_dm::<Cluster, m![C / 8], m![C % 8]>(&mut device.tdma);

    // Load rhs into VRF
    let rhs_broadcasted: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8, C % 8]> = device
        .main
        .begin(rhs.view())
        .fetch::<m![1], m![C % 8]>()
        .switch::<m![A / 8], m![C / 8]>(SwitchConfig::Broadcast01 {
            slice1: 256,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![C / 8], m![C % 8 # 32]>()
        .commit_trim::<m![C % 8]>()
        .commit();
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 8], m![C / 8, C % 8]> = device
        .sub
        .begin(rhs_broadcasted.view())
        .fetch::<m![C / 8], m![C % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C / 8], m![C % 8]>()
        .to_vrf();

    // Perform elementwise mul
    // The B dimension is in Cluster, so after contraction we still have Cluster=2
    // Result will have broadcast dimension X from GAT reduction
    let mul_result: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8, C % 8, A % 8]> = device
        .main
        .begin(lhs.view())
        .fetch::<m![C / 8, C % 8], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C / 8, C % 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)
        .vector_final()
        .cast::<i8, m![A % 8 # 32]>()
        .commit_trim::<m![A % 8]>()
        .commit();

    // Now reduce over the Cluster dimension using ReduceScatter pattern
    let reduced = reduce_over_cluster(device, &mul_result);

    // Write back to HBM
    reduced.to_hbm(&mut device.tdma)
}

/// Reduce over cluster axis using ReduceScatter pattern.
fn reduce_over_cluster(
    device: &mut Device,
    tensor: &DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8, C % 8, A % 8]>,
) -> DmTensor<i8, Chip, m![C / 1024], m![A / 8], m![C / 8 % 128, C % 8, A % 8]> {
    // ReduceScatter for Cluster=2
    // Input: Cluster dimension B=2
    // Output: C / 1024 promoted to Cluster dimension
    //
    // For 2-cluster ReduceScatter:
    // Step 1: Create T0 (with slice (0, 1)) and T1 (with slice(1,0) + shuffle(1,0))
    // Step 2: Add T0 + T1 to reduce cluster axis

    // Step 1a: Use ParallelCopy (via sub context with stos) to create sliced version
    // This performs asymmetric cluster slice operation
    let sliced0: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8 % 128, C % 8, A % 8]> =
        tensor.asymmetric_cluster_slice::<m![C / 1024], _>(&mut device.sub, &[0, 1]);

    // Step 1b: Use DmaCommand to shuffle the sliced data across clusters
    // This swaps data between Cluster 0 and Cluster 1
    let sliced1: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8 % 128, C % 8, A % 8]> =
        tensor.asymmetric_cluster_slice::<m![C / 1024], _>(&mut device.sub, &[1, 0]);

    let shuffled1: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8 % 128, C % 8, A % 8]> =
        sliced1.view().cluster_swap().to_dm(&mut device.tdma);

    // Step 2: Binary add T0 + T1 to reduce the cluster dimension
    // Use interleaved fetch with I=2 to read both tensors
    let mut reduced: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8 % 128, C % 8, A % 8]> = DmTensor::new();

    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(sliced0.view(), shuffled1.view())
        .fetch::<m![C / 8 % 128, C % 8, I], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C / 8 % 128, C % 8, I], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C / 8 % 128, C % 8]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .cast::<i8, m![A % 8 # 32]>()
        .commit_trim::<m![A % 8]>()
        .commit_view(reduced.view_mut());

    let reshaped: DmTensor<i8, Chip, m![C / 1024], m![A / 8], m![C / 8 % 128, C % 8, A % 8]> =
        unsafe { reduced.reshape() };

    reshaped
}
