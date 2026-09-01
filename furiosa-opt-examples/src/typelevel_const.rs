//! Regression fixture for the `Const::Ty` arm of `translate_const`
//! (npu-visa-translate). A `#[device]` entrypoint calls a helper generic over a
//! const loop bound `N`, and the helper's body reads `N` as a *value* in
//! `for b in 0..N`. That loop bound reaches MIR->vISA as a type-level constant
//! `Const::Ty(usize, N)` (a `ConstKind::Param`), not as a concrete literal.
//!
//! Because the callee is resolved with the entrypoint's concrete generic args,
//! `N` has a concrete value at translation, so MIR->vISA must const-eval it
//! rather than reject the kernel. This is what lets a macro-per-width kernel
//! family collapse into a single fn generic over a `const N: usize` loop bound.

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32];

/// Transpose the first `N` rows of `[A, B]` into `[B, A]`, driving the per-row
/// loop bound from the const-generic parameter `N` (a type-level constant, read
/// as a value in `0..N`). `N` is monomorphized to a literal at translation.
fn transpose_rows<const N: usize>(
    ctx: &mut Context,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
) {
    for b in 0..N {
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
}

/// Concrete (monomorphized) entrypoint that calls the const-generic helper with
/// `N = <m![B]>::SIZE`, so the helper's `for b in 0..N` lowers with `N` as a
/// literal.
#[device(chip = 1)]
pub fn typelevel_const(
    ctx: &mut Context,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    transpose_rows::<32>(ctx, input, &mut output);
    output
}

/// Stage `[A]` into `K`-wide slice groups, driving the partition from the
/// const-generic parameter through the mapping escape: `m![A / {K}]` splices
/// `K` into a size position the way a literal would be.
fn stage<const K: usize>(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A]>,
) -> DmTensor<i8, m![1], m![1], m![A / { K }], m![A % { K }]> {
    input.to_dm(&mut ctx.tdma)
}

/// Concrete entrypoint instantiating the mapping-escape helper with `K = 8`,
/// the companion of [`typelevel_const`]: one covers a const read as a loop
/// bound, this one a const spliced into the mapping itself.
#[device(chip = 1, pe = 1)]
pub fn typelevel_mapping(ctx: &mut Context, input: &HbmTensor<i8, m![1], m![A]>) -> HbmTensor<i8, m![1], m![A]> {
    let staged = stage::<8>(ctx, input);
    staged.to_hbm(&mut ctx.tdma)
}
