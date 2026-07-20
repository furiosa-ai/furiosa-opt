use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32, D = 64, L = 768];

// A size-1 ("unit") named axis, e.g. the decode-attention single-query-token axis.
axes![One = 1];

/// Device function that transposes a tensor from shape [A, B] to [B, A].
#[device(chip = 1)]
pub fn tile_simple(ctx: &mut Context, input: HbmTensorView<'_, i8, m![1], m![A, B]>) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    for b in 0..32 {
        // TODO: replace with <m![B]>::SIZE
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}

/// Reads a DM tensor column-by-column where each column is selected by a *computed* tile offset
/// `b = g * GROUP + j` rather than a bare loop variable, returning an identity copy of the input.
/// Splits the `B = 32` axis into `GROUPS = 2` groups of `GROUP = 16`; for each `(g, j)` it reads
/// `input[.., b]` (the load-bearing computed-offset tile) into a loop-local scratch sink.
///
/// The tiled view is **DM (SRAM)**, so the `for g { for j { .. } }` nest is unrolled at
/// `visa -> lir` and the computed offset's loop-variable leaves are substituted and folded to the
/// concrete column constant. (An HBM-tiled loop is *not* unrolled — it lowers as a kept VISA `Loop`
/// whose offset must be a bare scalar variable — so a computed offset must tile an SRAM view.)
///
/// Exercises computed tile offsets through the vISA-to-LIR unroller. The matching end-to-end test
/// remains ignored because of a downstream mutable-view write-back limitation documented there.
#[device(chip = 1)]
pub fn tile_computed_offset(
    ctx: &mut Context,
    input_hbm: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![A, B]> {
    const GROUPS: usize = 2;
    const GROUP: usize = 16;
    let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut ctx.tdma);
    let mut output = unsafe { DmTensor::<i8, m![1], m![A / 256], m![A % 256], m![B]>::from_addr(0x3000) };

    // Define the whole output up front (identity copy), so the result equals the input.
    input.view().to_dm_view(&mut ctx.tdma, output.view_mut());

    // Read each column at the *computed* offset `b = g * GROUP + j` into a loop-local scratch tile.
    // This is the read-side computed-offset pattern (a Q-head select `qh = kv_head * GQA_GROUP + g`):
    // the GROUPS x GROUP iterations sweep every column, so the loop nest is SRAM-tiled and unrolled
    // and the computed offset folds. The read is the load-bearing exercise; `scratch` (allocated
    // inside the loop body, so the write target is in scope) is a per-iteration sink.
    for g in 0..GROUPS {
        for j in 0..GROUP {
            let b = g * GROUP + j;
            let mut scratch = unsafe { DmTensor::<i8, m![1], m![A / 256], m![A % 256], m![1 # 32]>::from_addr(0x6000) };
            let input_col = input.view().tile::<m![B], 1, m![B = 1 # 32]>(b);
            input_col.to_dm_view(&mut ctx.tdma, scratch.view_mut());
        }
    }

    output.to_hbm(&mut ctx.tdma)
}

type Chip = m![1];
type Cluster = m![1];
type Slice = m![1 # 256];

/// Commits a fetched 32-wide window into the *upper* half of a 64-wide DM via `commit_view` into a
/// `view_mut().tile()`. The tile's out-of-tile cells are down (Bottom) padding, so the commit
/// sequencer places the `/ 8` strided time within the live prefix and leaves them unwritten rather
/// than rejecting the write. Copies `input[0..32]` into `result[32..64]`. Regression for a
/// `view_mut().tile()` commit into a windowed-and-down-padded DM that surfaced as
/// `StreamUnmatchedSegment`.
#[device(chip = 1)]
pub fn tile_window_commit(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![D]>) -> HbmTensor<f32, Chip, m![D]> {
    let tensor: DmTensor<f32, Chip, Cluster, Slice, m![D]> = input.to_dm(&mut ctx.tdma);
    let tile_one = tensor.view().tile::<m![D], 32, m![D = 32 # 64]>(0);

    let mut result: DmTensor<f32, Chip, Cluster, Slice, m![D]> = unsafe { DmTensor::from_addr(1 << 12) };

    ctx.main
        .begin(tile_one)
        .fetch::<m![1], m![D = 32]>()
        .fetch_cast::<f32>()
        .collect::<m![D = 32 / 8], m![D = 32 % 8]>()
        .commit_trim::<m![D = 32 % 8]>()
        .commit_view(result.view_mut().tile::<m![D], 32, m![D = 32 #{!} 64]>(32));

    result.to_hbm(&mut ctx.tdma)
}

/// Same transpose as [`tile_simple`], but the input is aliased with a bare `view()`
/// inside the loop before tiling. The bare view resolves directly to the outer
/// param tensor, which the loop-body builder must find among the parent's tensors;
/// without that it fails VISA -> LIR with "no tensor exists for T{n}".
#[device(chip = 1)]
pub fn tile_view_in_loop(ctx: &mut Context, input: &HbmTensor<i8, m![1], m![A, B]>) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    for b in 0..32 {
        let input_slice = input.view().tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}

/// Identity copy through a two-level HBM-tiled loop nest whose *inner* body tiles a view
/// produced by the *outer* loop variable -- the loop-body capture shape of the attention-decode
/// kernels. The outer `for g` slices `input` into a 16-wide `B`-group (`in_group`, tiled by the
/// outer index `g`); the inner `for h` tiles *that group* by the inner index, so the inner loop
/// body reads a tensor whose `IndexAccess` `table` traces back through `in_group` to the OUTER
/// loop's index tensor -- an index allocated in the enclosing scope, absent from the inner body's
/// own tensors.
///
/// An HBM-tiled loop is not unrolled (contrast the SRAM nest in [`tile_computed_offset`]), so both
/// `for`s survive as kept VISA `Loop`s. `convert_loop` must therefore (1) thread the outer index
/// into the inner loop's `captured_inputs` -- else VISA -> LIR hits the loop-body-graph verify wall
/// (`Loop::build_body_graph -> verify_structural_validity`, "no tensor exists for T{n}") -- and
/// (2) keep the inner nest's own locals OUT of the outer loop's `local_tensors` -- else the LIR
/// loop-body completeness check rejects them ("contained in local_tensors but not referenced by any
/// local instructions"). It is the in-repo regression for both facets of that fix; the matching
/// `test_tile_view_in_nested_loop` is a lowering-only check (`build_lir`, not `compare_lir!`) because
/// the host backend's tile-shape validator does not accept this two-level HBM tile.
#[device(chip = 1)]
pub fn tile_view_in_nested_loop(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![A, B]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x3000) };
    for g in 0..2 {
        // Outer-index tile: a 16-wide `B`-group selected by the enclosing loop var `g`. Its
        // `tile` takes `&self`, so the inner loop can re-slice it every iteration; the index
        // operand it carries (`g`) is allocated in the enclosing loop scope.
        let in_group = input.view().tile::<m![B / 16], 1, m![A, 1 # 2, B % 16]>(g);
        for h in 0..16 {
            // Inner-index tile of the outer-index view: the read tensor's index operand (`h`)
            // is inner-local, but its `table` traces back through `in_group` to the outer `g`.
            let in_col = in_group.tile::<m![B % 16], 1, m![A, 1 # 32]>(h);
            // The write side re-slices `output` by the same outer group index `g` and inner
            // column index `h`; the outer `g` view is rebuilt here (the mutable `tile` consumes
            // its receiver) but still references the enclosing `g` index tensor.
            let out_col = output
                .view_mut()
                .tile::<m![B / 16], 1, m![A, 1 # 2, B % 16]>(g)
                .tile::<m![B % 16], 1, m![A, 1 #{!} 32]>(h);
            in_col.to_hbm_view(&mut ctx.tdma, out_col);
        }
    }
    output
}

/// Chunked-output write created *inside* a loop and returned *after* it: the exact
/// T1 pattern that VISA -> LIR lowering must keep visible past the loop op.
///
/// Copies `input[B, D]` to `output[B, D]` one `D`-chunk per iteration. Each
/// `output.view_mut().tile::<m![D / 16], 1, m![B, 1 #{!} 4, D % 16]>(i)` is a
/// chunked write tile whose strided `D_16` chunk-index axis is absent from the
/// contiguous output shape, so lowering must materialize it (`split_chunk_index_axis`).
/// The write tile is a loop-body-local tensor whose writeback resolves against the
/// returned `output` *after* the loop closes; dropping loop-body locals from the parent
/// regresses this to a "no tensor T{n}" panic. Together this is a single runnable value
/// oracle (`compare_lir!`) for both the loop-tensor re-merge and the chunk-axis fix.
#[device(chip = 1)]
pub fn tile_chunked_output(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![B, D]>,
) -> HbmTensor<i8, m![1], m![B, D]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, D]>::from_addr(0x3000) };
    for i in 0..4 {
        let input_slice = input.view().tile::<m![D / 16], 1, m![B, 1 # 4, D % 16]>(i);
        let output_slice = output.view_mut().tile::<m![D / 16], 1, m![B, 1 #{!} 4, D % 16]>(i);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}

/// Identity-copy a tensor tiled along a size-1 named axis (the decode-attention `One = 1`
/// single-query-token pattern). The `.tile()` references the `One_1` tag that trivial-axis stripping
/// removed from the shape, exercising the absent-unit-axis no-op lowering (`is_absent_unit_axis`).
#[device(chip = 1)]
pub fn tile_unit_axis(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![One, B, D]>,
) -> HbmTensor<i8, m![1], m![One, B, D]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![One, B, D]>::from_addr(0x3000) };
    let input_slice = input.view().tile::<m![One], 1, m![1, B, D]>(0);
    let output_slice = output.view_mut().tile::<m![One], 1, m![1, B, D]>(0);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    output
}

#[device(chip = 1)]
pub fn tile_with_larger_than_one_1(
    ctx: &mut Context,
    up_weight: &HbmTensor<bf16, Chip, m![L, A]>,
) -> HbmTensor<bf16, Chip, m![L, A]> {
    let mut output = unsafe { HbmTensor::<bf16, Chip, m![L, A]>::from_addr(0x3000) };

    let input_slice = up_weight.view().tile::<m![L], 256, m![L = 256 # 768, A]>(0);
    let output_slice = output.view_mut().tile::<m![L], 256, m![L = 256 #{!} 768, A]>(2);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);

    let input_slice = up_weight.view().tile::<m![L], 256, m![L = 256 # 768, A]>(1);
    let output_slice = output.view_mut().tile::<m![L], 256, m![L = 256 #{!} 768, A]>(1);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);

    let input_slice = up_weight.view().tile::<m![L], 256, m![L = 256 # 768, A]>(2);
    let output_slice = output.view_mut().tile::<m![L], 256, m![L = 256 #{!} 768, A]>(0);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);

    output
}

#[device(chip = 1)]
pub fn tile_with_larger_than_one_2(ctx: &mut Context, up_weight: &HbmTensor<bf16, Chip, m![L, A]>) {
    let up_weight = up_weight
        .view()
        .tile::<m![L / 256], 2, m![L / 256 = 2 #{!} 3, L %  256, A]>(1);
}

/// Swaps the disjoint 256-wide `L`-chunks `[0, 256)` and `[256, 512)` via a tile window > 1,
/// copying the trailing chunk `[512, 768)` through unchanged.
///
/// `start` for a tile window > 1 is the raw `L`-axis offset of the window, not a chunk index scaled
/// by the window length -- confirmed by the working `tile_window_commit` example, where window `32`
/// at `start = 32` lands on `result[32..64]`, not some multiple of `32`. Every `start` here (`0`,
/// `256`, `512`) is already the exact axis offset of a chunk boundary, so the three windows are
/// disjoint and cover the whole `L` axis exactly once (unlike `tile_with_larger_than_one_1`, whose
/// `start` values `0`/`1`/`2` are small overlapping raw offsets that reproduce PROG-539 and leave
/// most of the output tensor unwritten).
#[device(chip = 1)]
pub fn tile_size_gt1_chunk_swap(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![L, A]>,
) -> HbmTensor<bf16, Chip, m![L, A]> {
    let mut output = unsafe { HbmTensor::<bf16, Chip, m![L, A]>::from_addr(0x3000) };

    let input_slice = input.view().tile::<m![L], 256, m![L = 256 # 768, A]>(0);
    let output_slice = output.view_mut().tile::<m![L], 256, m![L = 256 #{!} 768, A]>(256);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);

    let input_slice = input.view().tile::<m![L], 256, m![L = 256 # 768, A]>(256);
    let output_slice = output.view_mut().tile::<m![L], 256, m![L = 256 #{!} 768, A]>(0);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);

    let input_slice = input.view().tile::<m![L], 256, m![L = 256 # 768, A]>(512);
    let output_slice = output.view_mut().tile::<m![L], 256, m![L = 256 #{!} 768, A]>(512);
    input_slice.to_hbm_view(&mut ctx.tdma, output_slice);

    output
}
