//! Runtime `if` producing a *scalar* result: the merged value is a `ScalarExpr::Ternary` (the scalar
//! analogue of a tensor phi at a branch merge), which lowers to `SymExpr::Ternary`.
//!
//! A scalar `if` result is usable exactly where a *runtime* scalar is:
//!
//! - As a view/tile index it is fine ([`runtime_if_scalar_index`]): the index becomes an SPM value.
//! - As a vector-engine immediate it is not: an immediate is encoded in the instruction, so it must
//!   const-fold to a compile-time constant. That case is the rejection fixture
//!   [`negative::runtime_if_scalar`](crate::negative::runtime_if_scalar).

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32, N = 1024];
type Chip = m![1];

/// Runtime `if` selecting a tile INDEX. The tile offset `if i == 0 { 0 } else { 16 }` is a runtime
/// scalar (a `Ternary` on the loop index), and a runtime scalar is a valid view index, so this
/// compiles and runs. Iterations `0`/`1` copy the two 16-wide halves of `B`, together covering the
/// whole row.
///
/// Oracle: `output == input`.
#[device(chip = 1)]
pub fn runtime_if_scalar_index(
    ctx: &mut Context,
    input_hbm: &HbmTensor<i8, Chip, m![A, B]>,
) -> HbmTensor<i8, Chip, m![A, B]> {
    let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut ctx.tdma);
    let mut output = DmTensor::<i8, Chip, m![A / 256], m![A % 256], m![B]>::new();
    for i in 0..2 {
        let off: usize = if i == 0 { 0 } else { 16 };
        let src = input.view().tile::<m![B], 16, m![B = 16 # 32]>(off);
        let dst = output.view_mut().tile::<m![B], 16, m![B = 16 #{!} 32]>(off);
        src.to_dm_view(&mut ctx.tdma, dst);
    }
    output.to_hbm(&mut ctx.tdma)
}
