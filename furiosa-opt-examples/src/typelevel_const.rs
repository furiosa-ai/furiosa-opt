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
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    transpose_rows::<32>(ctx, input, &mut output);
    output
}
