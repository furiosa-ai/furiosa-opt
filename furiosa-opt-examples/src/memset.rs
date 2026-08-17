//! Kernels for the `DmTensorViewMut::memset` (ParallelMemSet / `itos`) fill op.
//!
//! Each kernel loads an HBM input into a 2-cluster DM region, overwrites the whole region with a
//! typed fill `value` via `dm.view_mut().memset(..)`, relayouts it (a DM->DM copy that reads the
//! filled region), then writes it back. The input is fully overwritten, so the readback must be the
//! fill value regardless of input: the signal that ParallelMemSet fired. The relayout reading the
//! just-filled region also pins the in-place fill aliasing (the consumer sees the fill, not the input).
//!
//! Covers every `Scalar` this branch translates to vISA: the byte+ RNGD scalars (`i8`, `i16`, `i32`,
//! `f32`, `bf16`, `f8e4m3`) and sub-byte `i4`. A plain literal folds `i8`/`i16`/`i32`/`f32`; a
//! `const { .. }` folds a computed value (`bf16`/`f8e4m3`/`i4`). `f4e2m1` is a follow-up (a valid DSL
//! scalar, but its vISA `mir_ast::Scalar` variant lands with the separate fetch/table-lookup work), so
//! a `memset(f4e2m1_value)` is rejected with a diagnostic rather than mis-lowered.
//!
//! The DM region spans `num_clusters_per_execution` clusters (`m![1 # 2]`) and uses the
//! `memory_op::dm_relayout` shape (the 2-cluster round-trip `m![A, B]` -> `m![A / 2, A % 2, B]`); `B`
//! is even, so sub-byte in-slice rows are byte-aligned.

use furiosa_opt_std::prelude::*;

axes![
    A = 256, // partitioned rows (2 clusters * 128 slices)
    B = 128  // in-slice payload per row
];

/// HBM input: a plain `m![A, B]` region.
type In<D> = HbmTensor<D, m![1], m![A, B]>;
/// HBM output: the 2-cluster relayout shape `m![A / 2, A % 2, B]` (`dm_relayout`'s output shape).
type Out<D> = HbmTensor<D, m![1], m![A / 2, A % 2, B]>;
/// 2-cluster DM region loaded straight from `In`.
type Dm<D> = DmTensor<D, m![1], m![1 # 2], m![A], m![B]>;
/// The relaid 2-cluster DM whose layout maps cleanly onto `Out`.
type Relaid<D> = DmTensor<D, m![1], m![1 # 2], m![1 # 2, A / 2], m![A % 2, B]>;

/// Fills `dm`, relayouts it (the relayout reads the fill), and writes it back. Shared by every
/// per-type kernel below via its concrete `value`.
macro_rules! memset_kernel {
    ($name:ident, $d:ty, $value:expr) => {
        #[device(chip = 1)]
        pub fn $name(ctx: &mut Context, input: &In<$d>) -> Out<$d> {
            let mut dm: Dm<$d> = input.to_dm(&mut ctx.tdma);
            dm.view_mut().memset($value, &mut ctx.sub);
            let relaid: Relaid<$d> = dm.to_dm(&mut ctx.tdma);
            relaid.to_hbm(&mut ctx.tdma)
        }
    };
}

// `PadValue::Zero` path (a zero fill folds to all-zero element bits).
memset_kernel!(memset_i32_zero, i32, 0);
// Signed `PadValue::Custom` (low byte `0xff`).
memset_kernel!(memset_i8_neg_one, i8, -1);
// 2-byte `PadValue::Custom`.
memset_kernel!(memset_i16_300, i16, 300);
// 4-byte float `PadValue::Custom`.
memset_kernel!(memset_f32_one_half, f32, 1.5);
// `const { .. }`-folded `bf16` (a computed value needs the const block to fold).
memset_kernel!(memset_bf16_one, bf16, const { bf16::from_f32(1.0) });
// `const { .. }`-folded `f8e4m3`.
memset_kernel!(memset_f8e4m3_one, f8e4m3, const { f8e4m3::from_f32(1.0) });
// Sub-byte `i4`: the fill value `-1` is the nibble `0xf`, replicated to every packed code.
memset_kernel!(memset_i4_neg_one, i4, const { i4::from_i32(-1) });
// Explicit in-place aliasing pin (structurally identical: the relayout consumes the fill).
memset_kernel!(memset_alias_bf16, bf16, const { bf16::from_f32(3.0) });
