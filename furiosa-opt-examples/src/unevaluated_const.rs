//! Regression fixture for the `Const::Unevaluated` arm of `translate_const`
//! (npu-visa-translate). The kernel body reads `<m![B]>::SIZE` as a *value*
//! and guards it with a `const { assert!(..) }`. Both reach MIR as a
//! monomorphizable anonymous/associated const, which MIR->vISA translation must
//! const-eval rather than panic on. See furiosa-ai#18502.

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32];

/// Transpose `[A, B]` to `[B, A]`, driving the per-row loop bound from
/// `<m![B]>::SIZE` (read as a value) and asserting it in a `const { .. }` block.
///
/// `clippy::assertions_on_constants` fires because a `const { .. }` shape guard's
/// condition is, by design, a compile-time constant; that is the exact idiom this
/// fixture pins, so the lint is allowed locally.
#[device(chip = 1)]
#[allow(clippy::assertions_on_constants)]
pub fn unevaluated_const(
    ctx: &mut Context,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    const { assert!(<m![B]>::SIZE % 32 == 0, "B must be a multiple of 32") };
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    for b in 0..<m![B]>::SIZE {
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}
