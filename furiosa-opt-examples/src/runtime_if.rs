//! Runtime branches over a non-unrolled loop index.
//!
//! The branch condition is an arbitrary runtime scalar: the example derives a `bool` variable from the
//! loop index (`let take_then = i == 0;`) and branches on it, exercising the condition path end to end.
//!
//! Only the statement form is here: each arm writes a distinct output, so no value phi is needed and
//! the kernel compiles end to end to EDF. The value form (`result = if ..`, a tensor phi merge) lowers
//! through vISA and runs on the simulator, but the scheduler cannot lower the merge to EDF yet (it
//! fails `BeamSearch`), so those four kernels live in `npu-opt-examples::unsupported::runtime_if`.

use furiosa_opt_std::prelude::*;

// A 4-PE device is 1 cluster x 256 slices, so `Cluster = m![1]` matches. `W / 4` x `W % 4` gives
// `Row` an 8-byte-aligned in-slice extent.
axes![W = 1024];

type Chip = m![1];
type Cluster = m![1];
type Row = DmTensor<i32, Chip, Cluster, m![W / 4], m![W % 4]>;

/// Statement-form runtime `if`: each arm writes into a *different* externally-defined output tensor's
/// view, so both arms' results are used (as the two kernel outputs) rather than a reassigned carry.
/// The condition is a `bool` from the index (`i == 0`).
///
/// `for i in 0..2`: `i == 0` takes the then-arm (`input + 1` into `out_then`), else `input + 2` into
/// `out_else`. Oracle: `out_then = input + 1`, `out_else = input + 2`.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_two_outputs(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![W]>,
) -> (HbmTensor<i32, Chip, m![W]>, HbmTensor<i32, Chip, m![W]>) {
    let input_dm: Row = input.to_dm(&mut ctx.tdma);
    let mut out_then: Row = DmTensor::new();
    let mut out_else: Row = DmTensor::new();

    for i in 0..2 {
        if i == 0 {
            add_const_view_mut(ctx, &input_dm, 1, out_then.view_mut());
        } else {
            add_const_view_mut(ctx, &input_dm, 2, out_else.view_mut());
        }
    }

    (out_then.to_hbm(&mut ctx.tdma), out_else.to_hbm(&mut ctx.tdma))
}

fn add_const_view_mut(
    ctx: &mut Context,
    input_dm: &Row,
    c: i32,
    output_view_mut: DmTensorViewMut<'_, i32, Chip, Cluster, m![W / 4], m![W % 4]>,
) {
    ctx.main
        .begin(input_dm.view())
        .fetch::<m![1], m![W % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![W % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, c)
        .vector_final()
        .commit_trim::<m![W % 4]>()
        .commit_view(output_view_mut)
}
