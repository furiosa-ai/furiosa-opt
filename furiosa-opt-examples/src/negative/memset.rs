//! Rejection fixture for a `memset` fill value that does not const-fold. The shapes mirror the
//! public [`memset`](crate::memset) kernels.

use furiosa_opt_std::prelude::*;

pub use crate::memset::{A, B};

/// HBM input: a plain `m![A, B]` region.
type In<D> = HbmTensor<D, m![1], m![A, B]>;
/// HBM output: the 2-cluster relayout shape `m![A / 2, A % 2, B]`.
type Out<D> = HbmTensor<D, m![1], m![A / 2, A % 2, B]>;
/// 2-cluster DM region loaded straight from `In`.
type Dm<D> = DmTensor<D, m![1], m![1 # 2], m![A], m![B]>;
/// The relaid 2-cluster DM whose layout maps cleanly onto `Out`.
type Relaid<D> = DmTensor<D, m![1], m![1 # 2], m![1 # 2, A / 2], m![A % 2, B]>;

/// A computed fill passed WITHOUT `const { .. }`. Without the const block, `bf16::from_f32(1.0)` is a
/// runtime call in this position; it has no device-translatable body, so the inline pass reports it
/// (guiding toward `const { .. }`) instead of the compiler panicking on the unlowered call.
#[device(chip = 1)]
pub fn memset_missing_const_bf16(ctx: &mut Context, input: &In<bf16>) -> Out<bf16> {
    let mut dm: Dm<bf16> = input.to_dm(&mut ctx.tdma);
    dm.view_mut().memset(bf16::from_f32(1.0), &mut ctx.sub);
    let relaid: Relaid<bf16> = dm.to_dm(&mut ctx.tdma);
    relaid.to_hbm(&mut ctx.tdma)
}
