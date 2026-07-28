//! Runtime branches over a non-unrolled loop index.
//!
//! The branch condition is an arbitrary runtime scalar: each example derives a `bool` variable from
//! the loop index (`let take_then = i == 1;`) and branches on it, exercising the condition path end to
//! end (comparison, `>`, a constant, a named `bool`).
//!
//! All but one example *use* the branch result: a live loop-carried accumulator read each iteration,
//! or an intermediate whose value a later op consumes. Only [`runtime_if_value`] reassigns a result it
//! never reads (a dead carry), kept as the single example of that shape.
//!
//! EDF status: a value-`if` (`result = if ..`, a tensor phi merge) lowers through VISA and runs on the
//! simulator, but the scheduler cannot yet lower the merge to EDF (it fails `BeamSearch`), so
//! [`runtime_if_value`], [`runtime_if_accumulate`], [`runtime_if_chain`], and [`runtime_if_const`] stop
//! at the `visa` stage in the regression snapshot and their `compare_edf!` tests are `#[ignore]`d. Only
//! the statement form [`runtime_if_two_outputs`] (each arm writes a distinct output, no value phi)
//! compiles end to end to EDF.

use furiosa_opt_std::prelude::*;

// A 4-PE device is 1 cluster x 256 slices, so `Cluster = m![1]` matches. `W / 4` x `W % 4` gives
// `Row` an 8-byte-aligned in-slice extent.
axes![W = 1024];

type Chip = m![1];
type Cluster = m![1];
type Row = DmTensor<i32, Chip, Cluster, m![W / 4], m![W % 4]>;

/// The single example whose branch result is reassigned each iteration but never read (a dead carry).
///
/// `result = if take_then { .. } else { .. }` overwrites `result` without consuming its prior value,
/// so the loop-carried initial arriving at each iteration is dead on arrival. The condition is a
/// `bool` derived from the loop index. `for i in 0..2`, `i == 1` at the last iteration, so the
/// surviving value is the then-arm.
///
/// Oracle: `output = input + 1`.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_value(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![W]>) -> HbmTensor<i32, Chip, m![W]> {
    let input_dm: Row = input.to_dm(&mut ctx.tdma);
    let mut result: Row = DmTensor::new();

    for i in 0..2 {
        let take_then = i == 1;
        result = if take_then {
            add_const(ctx, &input_dm, 1)
        } else {
            add_const(ctx, &input_dm, 2)
        };
    }

    result.to_hbm(&mut ctx.tdma)
}

/// Branch result USED as a live loop-carried accumulator: each arm reads `result` and adds onto it, so
/// the carry is genuinely consumed (unlike [`runtime_if_value`]). The condition is a `bool` from the
/// index (`i > 0`).
///
/// `result` starts as `input`; `i == 0` (else) adds 2, `i == 1` (then, `i > 0`) adds 1.
/// Oracle: `output = input + 3`.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_accumulate(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![W]>) -> HbmTensor<i32, Chip, m![W]> {
    let mut result: Row = input.to_dm(&mut ctx.tdma);

    for i in 0..2 {
        let take_then = i > 0;
        result = if take_then {
            add_const(ctx, &result, 1)
        } else {
            add_const(ctx, &result, 2)
        };
    }

    result.to_hbm(&mut ctx.tdma)
}

/// Branch result USED by a further op: the `if` selects an intermediate `mid` (reading the live carry
/// `result`), then a second op consumes `mid` to update `result`. The condition is a `bool` from the
/// index (`i == 1`).
///
/// `result` starts as `input`; each iteration `mid = result + (1 if take_then else 2)` then
/// `result = mid + 10`. `i == 0` (else): `mid = input + 2`, `result = input + 12`. `i == 1` (then):
/// `mid = input + 13`, `result = input + 23`.
/// Oracle: `output = input + 23`.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_chain(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![W]>) -> HbmTensor<i32, Chip, m![W]> {
    let mut result: Row = input.to_dm(&mut ctx.tdma);

    for i in 0..2 {
        let take_then = i == 1;
        let mid = if take_then {
            add_const(ctx, &result, 1)
        } else {
            add_const(ctx, &result, 2)
        };
        add_const_view_mut(ctx, &mid, 10, result.view_mut());
    }

    result.to_hbm(&mut ctx.tdma)
}

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

/// Constant branch condition (`if true`): the branch limit folds to the constant `1`, so the then-arm
/// is always taken. The result is used (returned) rather than reassigned across a loop.
///
/// Oracle: `output = input + 1`.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_const(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![W]>) -> HbmTensor<i32, Chip, m![W]> {
    let mut result: Row = input.to_dm(&mut ctx.tdma);

    result = if true {
        add_const(ctx, &result, 1)
    } else {
        add_const(ctx, &result, 2)
    };

    result.to_hbm(&mut ctx.tdma)
}

/// Fetch `input`, add `c` on the vector engine, and commit a fresh `[W]` result.
fn add_const(ctx: &mut Context, input_dm: &Row, c: i32) -> Row {
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
        .commit()
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
