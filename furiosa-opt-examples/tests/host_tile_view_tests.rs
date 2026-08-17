//! See [`furiosa_opt_examples::host_tile_view`] for the defect these tests bracket.

use furiosa_opt_examples::host_tile_view::{C, Chip, H, SmallRow, SmallRowMut, tile_move};
use furiosa_opt_std::prelude::*;

/// Row `r` of the table is filled with the value `r`.
fn table() -> Vec<bf16> {
    (0..C::SIZE)
        .flat_map(|r| std::iter::repeat_n(bf16::from_f32(r as f32), H::SIZE))
        .collect()
}

/// Both ends tiled on the host. Reading row 3 and writing row 5 keeps the two failures apart: a
/// dropped read offset puts row 0 on row 5, a dropped write offset puts row 3 on row 0.
#[tokio::test]
async fn test_host_tile_moves_the_requested_row_to_the_requested_row() {
    let mut ctx = Context::acquire();

    let table = HostTensor::<bf16, m![C, H]>::from_vec(table())
        .to_hbm::<Chip, m![C, H]>(&mut ctx.pdma)
        .await;
    // -1.0 marks a row the kernel never wrote, which no source row can be mistaken for.
    let mut out = HostTensor::<bf16, m![C, H]>::from_vec(vec![bf16::from_f32(-1.0); C::SIZE * H::SIZE])
        .to_hbm::<Chip, m![C, H]>(&mut ctx.pdma)
        .await;

    let src = table.view().tile::<m![C], 1, SmallRow>(3);
    let dst = out.view_mut().tile::<m![C], 1, SmallRowMut>(5);
    launch(tile_move, (&mut *ctx, src, dst)).await;

    let actual = out.to_host::<m![C, H]>(&mut ctx.pdma).await.into_vec();
    for r in 0..C::SIZE {
        let want = if r == 5 { 3.0 } else { -1.0 };
        assert_eq!(
            &actual[r * H::SIZE..(r + 1) * H::SIZE],
            vec![bf16::from_f32(want); H::SIZE],
            "row {r} after moving row 3 to row 5"
        );
    }
}

/// `tile`'s `start` is a plain `usize`, so a host loop can select a different row each iteration,
/// which is what an embedding lookup does per token.
#[tokio::test]
async fn test_host_tile_offsets_may_be_runtime_values() {
    let mut ctx = Context::acquire();

    let table = HostTensor::<bf16, m![C, H]>::from_vec(table())
        .to_hbm::<Chip, m![C, H]>(&mut ctx.pdma)
        .await;
    let mut out = HostTensor::<bf16, m![C, H]>::from_vec(vec![bf16::from_f32(-1.0); C::SIZE * H::SIZE])
        .to_hbm::<Chip, m![C, H]>(&mut ctx.pdma)
        .await;

    // Cross the allocation's first, middle, and last rows on both the immutable source view and
    // mutable destination view. Under the NPU build, every pair goes through the view-to-buffer
    // conversion before launch.
    let moves = vec![(0, C::SIZE - 1), (C::SIZE / 2, C::SIZE / 2), (C::SIZE - 1, 0), (3, 5)];
    for &(src_row, dst_row) in &moves {
        let src = table.view().tile::<m![C], 1, SmallRow>(src_row);
        let dst = out.view_mut().tile::<m![C], 1, SmallRowMut>(dst_row);
        launch(tile_move, (&mut *ctx, src, dst)).await;
    }

    let actual = out.to_host::<m![C, H]>(&mut ctx.pdma).await.into_vec();
    for r in 0..C::SIZE {
        let want = moves
            .iter()
            .find(|(_, dst_row)| *dst_row == r)
            .map_or(-1.0, |(src_row, _)| *src_row as f32);
        assert_eq!(
            &actual[r * H::SIZE..(r + 1) * H::SIZE],
            vec![bf16::from_f32(want); H::SIZE],
            "row {r} after the loop"
        );
    }
}
