//! Rejection fixture for the `Call { target: None }` (diverging panic) arm of
//! MIR->vISA translation (npu-visa-translate). The body contains a *runtime*
//! `assert!` over a value-dependent condition, which lowers to a diverging
//! `core::panicking::panic*` call. MIR->vISA must reject this with a clean
//! diagnostic ("runtime `panic!`/`assert!` is not supported in a `#[device]`
//! body ...") rather than ICE with `not yet implemented`. Device bodies allow
//! only compile-time shape guards (`const { assert!(..) }`). See furiosa-ai#18502.

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32];

/// Transpose `[A, B]` to `[B, A]`, guarded by a *runtime* `assert!` on a
/// value-dependent condition (the loop index), which MIR->vISA must reject.
#[device(chip = 1)]
pub fn runtime_panic(
    ctx: &mut Context,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    for b in 0..<m![B]>::SIZE {
        assert!(b < 1000, "runtime guard that must be rejected");
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}
