//! Chunking an axis declared with inline padding (`W # 155648`), the tutorial's `lm_head_w` shape.

use furiosa_opt_std::prelude::*;

// `Live` and `Wide` do not divide by 8, as `W = 151936` does not by 8192; `Divisible` and `Plain` do.
axes![Live = 21, Padded = 24, Wide = 21, H = 4, Divisible = 24, Plain = 24];

pub type Chip = m![1];

/// Padded to the exact chunk multiple: 3 chunks, all of them holding live rows.
pub type Chunked = m![Live # 24 / 8, Live # 24 % 8, H];

/// Padded past the chunk multiple: 4 chunks, only 3 of which hold live rows.
pub type ChunkedWide = m![Wide # 32 / 8, Wide # 32 % 8, H];

pub type Chunk = m![Live # 24 % 8, H];
pub type ChunkWide = m![Wide # 32 % 8, H];

/// The alternative report §3.4 recommends: a padded symbol of its own.
pub type ChunkedPadded = m![Padded / 8, Padded % 8, H];

/// The same copy over a padded symbol, for contrast.
#[device(chip = 1)]
pub fn padded_symbol_chunk_copy(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, ChunkedPadded>,
) -> HbmTensor<bf16, Chip, ChunkedPadded> {
    let mut output = HbmTensor::<bf16, Chip, ChunkedPadded>::new();
    for i in 0..3 {
        let src = input.view().tile::<m![Padded / 8], 1, m![1 # 3, Padded % 8, H]>(i);
        let dst = output
            .view_mut()
            .tile::<m![Padded / 8], 1, m![1 #{!} 3, Padded % 8, H]>(i);
        src.to_hbm_view(&mut ctx.tdma, dst);
    }
    output
}

/// One chunk of the tutorial's own declaration, whose live rows do not divide into its three chunks.
#[device(chip = 1)]
pub fn padded_axis_chunk_read(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, Chunked>,
) -> HbmTensor<bf16, Chip, Chunk> {
    let mut output = HbmTensor::<bf16, Chip, Chunk>::new();
    let src = input
        .view()
        .tile::<m![Live # 24 / 8], 1, m![1 # 3, Live # 24 % 8, H]>(1);
    src.to_hbm_view(&mut ctx.tdma, output.view_mut());
    output
}

/// Chunking is valid here (`24 = 3 * 8`) and leaves the padding alone: 4 chunks, 3 with live rows.
pub type ChunkedDivisible = m![Divisible # 32 / 8, Divisible # 32 % 8, H];

/// The lowering side: `IndexAccess` steps over 4 chunks where the projection's term counts 3.
#[device(chip = 1)]
pub fn divisible_chunk_read(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, ChunkedDivisible>,
) -> HbmTensor<bf16, Chip, m![Divisible # 32 % 8, H]> {
    let mut output = HbmTensor::<bf16, Chip, m![Divisible # 32 % 8, H]>::new();
    let src = input
        .view()
        .tile::<m![Divisible # 32 / 8], 1, m![1 # 4, Divisible # 32 % 8, H]>(1);
    src.to_hbm_view(&mut ctx.tdma, output.view_mut());
    output
}

/// Control case: no padding anywhere.
pub type ChunkedPlain = m![Plain / 8, Plain % 8, H];

#[device(chip = 1)]
pub fn plain_chunk_read(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, ChunkedPlain>,
) -> HbmTensor<bf16, Chip, m![Plain % 8, H]> {
    let mut output = HbmTensor::<bf16, Chip, m![Plain % 8, H]>::new();
    let src = input.view().tile::<m![Plain / 8], 1, m![1 # 3, Plain % 8, H]>(1);
    src.to_hbm_view(&mut ctx.tdma, output.view_mut());
    output
}

/// The last live chunk of [`ChunkedDivisible`], rows 16..24.
#[device(chip = 1)]
pub fn divisible_last_live_chunk_read(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, ChunkedDivisible>,
) -> HbmTensor<bf16, Chip, m![Divisible # 32 % 8, H]> {
    let mut output = HbmTensor::<bf16, Chip, m![Divisible # 32 % 8, H]>::new();
    let src = input
        .view()
        .tile::<m![Divisible # 32 / 8], 1, m![1 # 4, Divisible # 32 % 8, H]>(2);
    src.to_hbm_view(&mut ctx.tdma, output.view_mut());
    output
}
