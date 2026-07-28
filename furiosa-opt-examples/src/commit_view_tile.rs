//! Two `commit_view()` writes into different tile offsets of one scratch
//! `DmTensor` (the vector-engine fetch/collect/commit path). Isolates a
//! commit-into-shared-scratch regression with no rope/projection/MLP math in
//! the way: the failing model kernels (`rope`, `proj_q`/`proj_o`, `mlp`) all
//! share exactly this write pattern.
//!
//! Shape is `[A, X] = [8, 32]`. `X` cannot go below 32: the Collect engine
//! requires each output packet to be exactly one 32-byte flit (16 bf16
//! elements), so a two-way split needs at least 16 + 16.
//!
//! The kernel RETURNS the result as a fresh HBM tensor rather than writing a
//! `&mut` output, because the LIR interpreter (the test value oracle) does not
//! support a kernel whose only result is a single `&mut` output.

use furiosa_opt_std::prelude::*;

axes![A = 8, X = 32];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// Swaps the two `X`-halves of `[A, X]` via two separate `commit_view()` writes
/// into one fresh scratch `DmTensor`, then returns the scratch as HBM.
///
/// The vector engine only moves bytes here (fetch -> collect -> commit, no
/// contraction), so a healthy toolchain returns exactly `x` with its two
/// `X`-halves swapped. Write 1 sends `x[.., 0..16]` into `swapped[.., 16..32]`;
/// write 2 sends `x[.., 16..32]` into `swapped[.., 0..16]`.
///
/// Every `tile` `start` here is an ELEMENT index along `X`: the second half
/// begins at element `16`, matching what the commit (write) side and the DMA
/// path use. This is the regression guard for the fetch-path tile-offset unit:
/// the fetch offset must be an element index, not a byte offset. A byte-unit
/// fetch offset (the `index_access_axis_stride_in_bytes` `MultiShape` bug) would
/// read element `16 / 2 = 8` for bf16 and return the wrong swap.
#[device(chip = 1)]
pub fn swap_halves(ctx: &mut Context, x: &HbmTensor<bf16, Chip, m![A, X]>) -> HbmTensor<bf16, Chip, m![A, X]> {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![A, X]> = x.to_dm(&mut ctx.tdma);

    let fst_half = x.view().tile::<m![X], 16, m![A, X = 16 # 32]>(0);
    let snd_half = x.view().tile::<m![X], 16, m![A, X = 16 # 32]>(16);

    let mut swapped: DmTensor<bf16, Chip, Cluster, Slice, m![A, X]> = DmTensor::new();

    // Write 1/2: first half of `x` -> second half of `swapped`.
    ctx.main
        .begin(fst_half)
        .fetch::<m![A], m![X = 16]>()
        .collect::<m![A], m![X = 16]>()
        .commit_trim::<m![X = 16]>()
        .commit_view(swapped.view_mut().tile::<m![X], 16, m![A, X = 16 #{!} 32]>(16));

    // Write 2/2: second half of `x` -> first half of `swapped`.
    ctx.main
        .begin(snd_half)
        .fetch::<m![A], m![X = 16]>()
        .collect::<m![A], m![X = 16]>()
        .commit_trim::<m![X = 16]>()
        .commit_view(swapped.view_mut().tile::<m![X], 16, m![A, X = 16 #{!} 32]>(0));

    swapped.to_hbm(&mut ctx.tdma)
}
