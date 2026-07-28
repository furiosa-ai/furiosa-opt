//! Fetch-adapter table-lookup example kernels.
//!
//! Exercises the real `fetch_table_lookup` primitive on an `f4e2m1` (NVFP4 / MXFP4)
//! weight stream: the fetch chain decodes each nibble to `f8e4m3` through the
//! hardware paired-key table, then widens to `f32` with a following `fetch_cast`.
//! See the book chapter `computing-tensors/fetch-adapter.md` ("Table Lookup").

#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Decodes an `f4e2m1` weight stream to `f32` via the fetch-adapter table-lookup
/// stage (`f4e2m1 -> f8e4m3`, the paired-key 4b->8b table) followed by a widening
/// `fetch_cast`. This is the intended NVFP4 / MXFP4 decode path; the per-block scale
/// is applied downstream and is intentionally not part of this kernel.
///
/// The decode table is a block-independent 16-entry constant, so this one kernel is
/// the decode for both the NVFP4 (16-element block-scale) and MXFP4 (32-element
/// block-scale) layouts. The `fetch_table_lookup_decode_e2m1_mxfp4` twin below tiles
/// the same decode over a 32-element inner block to exercise the MXFP4 block boundary.
#[device(chip = 1)]
pub fn fetch_table_lookup_decode_e2m1(
    ctx: &mut Context,
    input: &HbmTensor<f4e2m1, m![1], m![A, B]>,
) -> HbmTensor<f32, m![1], m![B, A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma);

    let decoded: DmTensor<f32, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A / 8 % 2], m![A % 8, B]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![A / 8 % 2, A % 8], m![B]>()
        .commit_trim::<m![B]>()
        .commit();

    decoded.to_hbm(&mut ctx.tdma)
}

// MXFP4's 32-element scale block is twice the NVFP4 16-element block, so the MXFP4 twin doubles
// the fetched axis (`AM = 2 * A`) to keep the same 256-slice partitioning (`AM / 32 == A / 16`)
// while the fetch packet spans a 32-element MXFP4 block instead of a 16-element NVFP4 block.
axes![AM = 8192];

/// MXFP4-block twin of [`fetch_table_lookup_decode_e2m1`]: the same block-independent
/// `f4e2m1 -> f8e4m3 -> f32` decode, but the fetch packet spans an MXFP4 32-element scale block
/// (`AM % 16 = 16` inner over 2 time steps = 32) instead of the NVFP4 16-element block. Proves the
/// packed decode is correct at the wider block granularity; the per-block scale (E8M0 for MXFP4)
/// stays downstream, as for the NVFP4 kernel.
#[device(chip = 1)]
pub fn fetch_table_lookup_decode_e2m1_mxfp4(
    ctx: &mut Context,
    input: &HbmTensor<f4e2m1, m![1], m![AM, B]>,
) -> HbmTensor<f32, m![1], m![B, AM]> {
    let input_dm = input.to_dm::<Cluster, m![AM / 32], m![AM / 8 % 4, AM % 8, B]>(&mut ctx.tdma);

    let decoded: DmTensor<f32, Chip, Cluster, m![AM / 32], m![AM / 8 % 4, AM % 8, B]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![AM / 8 % 4], m![AM % 8, B]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![AM / 8 % 4, AM % 8], m![B]>()
        .commit_trim::<m![B]>()
        .commit();

    decoded.to_hbm(&mut ctx.tdma)
}

// The commit packet must be exactly one flit (32 bytes). The `bf16` value is 2 bytes, so the decode
// example's inner element axis is 16 wide (16 * 2 = 32 bytes), unlike the NVFP4 decode whose f32
// output makes an 8-wide axis a flit.
axes![Bf = 16];

/// Decodes an `f8e4m3` stream to `bf16` through the **non-paired** 8-bit-key baked table
/// (`f8e4m3 -> bf16`), the general single-key counterpart to the paired NVFP4 / MXFP4 4b->8b decode
/// above. An 8-bit key indexes one `bf16` value per entry (no paired byte-walk), so the fetch emits
/// `bf16` directly with no following `fetch_cast`. Proves the non-paired baked table-lookup path.
#[device(chip = 1)]
pub fn fetch_table_lookup_decode_f8_to_bf16(
    ctx: &mut Context,
    input: &HbmTensor<f8e4m3, m![1], m![A, Bf]>,
) -> HbmTensor<bf16, m![1], m![Bf, A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, Bf]>(&mut ctx.tdma);

    let decoded: DmTensor<bf16, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, Bf]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A / 8 % 2], m![A % 8, Bf]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![A / 8 % 2, A % 8], m![Bf]>()
        .commit_trim::<m![Bf]>()
        .commit();

    decoded.to_hbm(&mut ctx.tdma)
}
