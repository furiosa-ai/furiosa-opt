//! Runtime `if` producing a *scalar* result: the merged value is a `ScalarExpr::Ternary` (the scalar
//! analogue of a tensor phi at a branch merge), which lowers to `SymExpr::Ternary`.
//!
//! A scalar `if` result is usable exactly where a *runtime* scalar is:
//!
//! - As a view/tile index it is fine ([`runtime_if_scalar_index`]): the index becomes an SPM value.
//! - As a vector-engine immediate it is not ([`runtime_if_scalar_immediate`]): an immediate is
//!   encoded in the instruction, so it must const-fold to a compile-time constant. That function
//!   fails to compile with a diagnostic saying so; it is a negative example.

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

type Row = DmTensor<i32, Chip, m![1], m![N / 4], m![N % 4]>;

/// Runtime `if` selecting a vector-engine IMMEDIATE: `vector_fxp(AddFxp, if i == 1 { 1 } else { 2 })`.
/// The immediate is encoded in the instruction, so it must const-fold to a compile-time constant; the
/// `if` result is a runtime scalar that does not, so this fails to compile (a negative example).
///
/// Expected: `a vector-engine immediate must be a compile-time constant, but `...` does not const-fold`.
///
/// Runs at a 4-PE device (1 cluster) so the `Cluster = m![1]` `Row` is a valid allocation and the
/// compile reaches the vector-engine-immediate check that is the point of this negative example.
#[device(chip = 1, pe = 4)]
pub fn runtime_if_scalar_immediate(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![N]>,
) -> HbmTensor<i32, Chip, m![N]> {
    let input_dm: Row = input.to_dm(&mut ctx.tdma);
    let mut result: Row = DmTensor::new();
    for i in 0..2 {
        let c: i32 = if i == 1 { 1 } else { 2 };
        result = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![1], m![N % 4]>()
            .fetch_cast::<i32>()
            .collect::<m![1], m![N % 4 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, c)
            .vector_final()
            .commit_trim::<m![N % 4]>()
            .commit();
    }
    result.to_hbm(&mut ctx.tdma)
}
