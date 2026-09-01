use furiosa_opt_examples::vrf_operand::{
    B, N, V, W, X, Y, vrf_group_scale, vrf_slice_regroup, vrf_slice_rename, vrf_slice_reshape, vrf_slice_reshape_64,
};
use furiosa_opt_std::prelude::*;

/// The reshaped operand feeds every slice: each `W` row reads the same `B` vector, which is the
/// broadcast the operand was written as. Rejecting the un-reshaped operand is a compile error, so it
/// lives as a `compile_fail` example in `negative::vector_engine`'s module docs instead.
#[tokio::test]
async fn test_vrf_slice_reshape() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).collect::<Vec<_>>());
    let operand = HostTensor::<i32, m![B]>::from_vec((0..<m![B]>::SIZE as i32).map(|x| x * 10).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![W, B]>(&mut ctx.pdma).await;
    let operand_hbm = operand.to_hbm::<m![1], m![B]>(&mut ctx.pdma).await;

    let output = launch(vrf_slice_reshape, (&mut *ctx, &input_hbm, &operand_hbm)).await;

    let expected: Vec<i32> = (0..<m![W, B]>::SIZE as i32)
        .map(|x| x + (x % <m![B]>::SIZE as i32) * 10)
        .collect();
    assert_eq!(output.to_host::<m![W, B]>(&mut ctx.pdma).await.into_vec(), expected);
}

/// [`test_vrf_slice_reshape`] at the 64-slice topology, the one whose `compare_edf` peer runs in a
/// default config.
#[tokio::test]
async fn test_vrf_slice_reshape_64() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<i32, m![V, B]>::from_vec((0..<m![V, B]>::SIZE as i32).collect::<Vec<_>>());
    let operand = HostTensor::<i32, m![B]>::from_vec((0..<m![B]>::SIZE as i32).map(|x| x * 10).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, B]>(&mut ctx.pdma).await;
    let operand_hbm = operand.to_hbm::<m![1], m![B]>(&mut ctx.pdma).await;

    let output = launch(vrf_slice_reshape_64, (&mut *ctx, &input_hbm, &operand_hbm)).await;

    let expected: Vec<i32> = (0..<m![V, B]>::SIZE as i32)
        .map(|x| x + (x % <m![B]>::SIZE as i32) * 10)
        .collect();
    assert_eq!(output.to_host::<m![V, B]>(&mut ctx.pdma).await.into_vec(), expected);
}

/// A composite `Slice` meets the partition rule like a plain one. The regroup decomposes the axis
/// the stream itself names, so slice `w` holds row `w` either way; [`test_vrf_slice_rename`] is where
/// the reshape could move a row and the answer key would notice.
#[tokio::test]
async fn test_vrf_slice_regroup() {
    let mut ctx = Context::acquire();

    let cell = |i: i32| (i / <m![B]>::SIZE as i32) * 1000 + i % <m![B]>::SIZE as i32;
    let input = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).collect::<Vec<_>>());
    let operand = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).map(cell).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![W, B]>(&mut ctx.pdma).await;
    let operand_hbm = operand.to_hbm::<m![1], m![W, B]>(&mut ctx.pdma).await;

    let output = launch(vrf_slice_regroup, (&mut *ctx, &input_hbm, &operand_hbm)).await;

    let expected: Vec<i32> = (0..<m![W, B]>::SIZE as i32).map(|i| i + cell(i)).collect();
    assert_eq!(output.to_host::<m![W, B]>(&mut ctx.pdma).await.into_vec(), expected);
}

/// The reshape renames the 256 slices from `X` / `Y` onto the stream's `W`, so slice `x * 16 + y`
/// must still read the operand row it holds. Every slice holds a different row, so swapping the two
/// axes, or shifting the identity at all, changes the answer instead of hiding behind replication.
#[tokio::test]
async fn test_vrf_slice_rename() {
    let mut ctx = Context::acquire();

    let cell = |i: i32| (i / <m![B]>::SIZE as i32) * 1000 + i % <m![B]>::SIZE as i32;
    let input = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).collect::<Vec<_>>());
    // `X` / `Y` row `x * 16 + y` is laid out where `W` row `x * 16 + y` is, which is what the rename
    // claims, so the two tensors index the same way and the answer key is the regroup's.
    let operand =
        HostTensor::<i32, m![X, Y, B]>::from_vec((0..<m![X, Y, B]>::SIZE as i32).map(cell).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![W, B]>(&mut ctx.pdma).await;
    let operand_hbm = operand.to_hbm::<m![1], m![X, Y, B]>(&mut ctx.pdma).await;

    let output = launch(vrf_slice_rename, (&mut *ctx, &input_hbm, &operand_hbm)).await;

    let expected: Vec<i32> = (0..<m![W, B]>::SIZE as i32).map(|i| i + cell(i)).collect();
    assert_eq!(output.to_host::<m![W, B]>(&mut ctx.pdma).await.into_vec(), expected);
}

/// A coarse operand: each scale covers 16 stream elements, so the four lanes of one access read it
/// as a single address. Every group carries a different scale, so a read that stepped the operand
/// per lane, or per access, lands on the wrong one and the answer key says so.
#[tokio::test]
async fn test_vrf_group_scale() {
    let mut ctx = Context::acquire();

    // Powers of two: the division is exact, so the comparison needs no tolerance.
    let scale_of = |group: usize| (1 << (group % 4)) as f32;
    let input = HostTensor::<f32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE).map(|i| i as f32).collect::<Vec<_>>());
    let scale =
        HostTensor::<f32, m![V, N / 16]>::from_vec((0..<m![V, N / 16]>::SIZE).map(scale_of).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, N]>(&mut ctx.pdma).await;
    let scale_hbm = scale.to_hbm::<m![1], m![V, N / 16]>(&mut ctx.pdma).await;

    let output = launch(vrf_group_scale, (&mut *ctx, &input_hbm, &scale_hbm)).await;

    let expected: Vec<f32> = (0..<m![V, N]>::SIZE).map(|i| i as f32 / scale_of(i / 16)).collect();
    assert_eq!(output.to_host::<m![V, N]>(&mut ctx.pdma).await.into_vec(), expected);
}
