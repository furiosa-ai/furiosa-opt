//! Matrix multiplication of size [1024, 2048] x [2048] -> [1024] with split accumulation pattern.
//! This demonstrates the same split/accumulate pattern as matmul_16384 but with smaller size for testing.

use furiosa_visa_std::prelude::*;

axes![A = 1024, B = 2048, X = 32, I = 2];

type Chip = m![1];
type Cluster = m![A / 512 % 2];

// TODO: Generics in a user-defined function is not supported yet. (LOW-76)
fn add_split_contractions_i32(
    ctx: &mut Context,
    lhs: &DmTensor<i32, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>,
    rhs: &DmTensor<i32, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>,
    out: DmTensorViewMut<i32, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>,
) {
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view())
        .fetch::<i32, m![A / 8 % 8, I], m![A % 8]>()
        .collect::<m![A / 8 % 8, I], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![A / 8 % 8, 1 # 2], m![A / 8 % 8]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_view(out)
}

// TODO: Generics in a user-defined function is not supported yet. (LOW-76)
fn add_split_contractions_i8(
    ctx: &mut Context,
    lhs: &DmTensor<i32, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>,
    rhs: &DmTensor<i32, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>,
    out: DmTensorViewMut<i8, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>,
) {
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(rhs.view(), lhs.view())
        .fetch::<i32, m![A / 8 % 8, I], m![A % 8]>()
        .collect::<m![A / 8 % 8, I], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![A / 8 % 8, 1 # 2], m![A / 8 % 8]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .cast::<i8, m![A % 8 # 32]>()
        .commit_view(out)
}

/// Multiply matrices of size \[1024, 2048\] x \[2048\] -> \[1024\]
/// Demonstrates split/accumulate pattern: B axis is split into 4 tiles (B=512 each)
/// and accumulated by stashing one group inside VE and later merging with the other group.
#[device(chip = 1)]
pub fn matmul_with_split_reduce(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, m![1], m![A, B]>,
    rhs: &HbmTensor<i8, m![1], m![B]>,
) -> HbmTensor<i8, m![1], m![A]> {
    type AccDmTensor = DmTensor<i32, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>;
    let mut result0 = unsafe { AccDmTensor::from_addr(0) };
    let mut result1 = unsafe { AccDmTensor::from_addr(0) };
    let mut temp = unsafe { AccDmTensor::from_addr(0) };
    let mut output = unsafe { DmTensor::<i8, Chip, Cluster, m![A / 64 % 8, X], m![A % 64]>::from_addr(0) };

    // Split B axis into 4 tiles (B=512 each) and accumulate using vector engine
    for j in 0..4 {
        // Load lhs and rhs for this B tile
        let lhs_tile = lhs
            .view()
            .tile::<m![B / 512], 1, m![A, 1 # 4, B % 512]>(j)
            .to_dm::<Cluster, m![A / 64 % 8, B / 16 % 32], m![A % 64, B % 16]>(&mut ctx.tdma, 0);
        let rhs_tile = rhs
            .view()
            .tile::<m![B / 512], 1, m![1 # 4, B % 512]>(j)
            .to_dm::<Cluster, m![A / 64 % 8, B / 16 % 32], m![B % 16]>(&mut ctx.tdma, 256 * 1024);
        let rhs_trf: TrfTensor<i8, Chip, Cluster, m![A / 64 % 8, B / 16 % 32], m![1], m![B % 16 # 32]> = ctx
            .sub
            .begin(rhs_tile.view())
            .fetch::<i8, m![1], m![B % 16]>()
            .collect::<m![1], m![B % 16 # 32]>()
            .to_trf(TrfAddress::Full);

        // Perform contraction for this tile
        if j == 0 {
            // First iteration: store result directly in result0
            ctx.main
                .begin(lhs_tile.view())
                .fetch::<i8, m![A / 2 % 32], m![A % 2, B % 16]>()
                .collect::<m![A / 2 % 32], m![A % 2, B % 16]>()
                .align::<m![A / 4 % 16], m![A % 4, B % 16], _, _>(&rhs_trf)
                .contract::<m![A % 4]>()
                .accumulate::<m![A / 4 % 16], m![A % 4 # 8]>(AccumulationKind::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<m![A / 64 % 8, X], m![A / 4 % 16]>(InterSliceReduceOpI32::Add)
                .vector_final()
                .commit_view(result0.view_mut());
        } else if j == 3 {
            // Final iteration: store in output, conversion to i8 happens
            ctx.main
                .begin(lhs_tile.view())
                .fetch::<i8, m![A / 2 % 32], m![A % 2, B % 16]>()
                .collect::<m![A / 2 % 32], m![A % 2, B % 16]>()
                .align::<m![A / 4 % 16], m![A % 4, B % 16], _, _>(&rhs_trf)
                .contract::<m![A % 4]>()
                .accumulate::<m![A / 4 % 16], m![A % 4 # 8]>(AccumulationKind::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<m![A / 64 % 8, X], m![A / 4 % 16]>(InterSliceReduceOpI32::Add)
                .vector_final()
                .commit_view(temp.view_mut());

            // Accumulate using vector engine: stash previous partial sums and zip with new ones
            add_split_contractions_i8(ctx, &result0, &temp, output.view_mut());
        } else {
            // Subsequent iterations: store in temp and accumulate
            ctx.main
                .begin(lhs_tile.view())
                .fetch::<i8, m![A / 2 % 32], m![A % 2, B % 16]>()
                .collect::<m![A / 2 % 32], m![A % 2, B % 16]>()
                .align::<m![A / 4 % 16], m![A % 4, B % 16], _, _>(&rhs_trf)
                .contract::<m![A % 4]>()
                .accumulate::<m![A / 4 % 16], m![A % 4 # 8]>(AccumulationKind::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<m![A / 64 % 8, X], m![A / 4 % 16]>(InterSliceReduceOpI32::Add)
                .vector_final()
                .commit_view(temp.view_mut());

            // Accumulate using vector engine: stash the older tile internally and zipped with the new tile
            if j % 2 == 1 {
                add_split_contractions_i32(ctx, &result0, &temp, result1.view_mut());
            } else {
                add_split_contractions_i32(ctx, &result1, &temp, result0.view_mut());
            }
        }
    }

    // Write back to HBM
    let mut out = unsafe { HbmTensor::<i8, m![1], m![A]>::from_addr(0x3000) };
    output
        .view()
        .slice_tile::<m![X], 1, m![A / 64 % 8, 1 # 32]>(0)
        .to_hbm_view(&mut ctx.tdma, out.view_mut());
    out
}
