//! Branch-unit example: per-element branches driven by `TagMode::Comparison` tags and
//! [`Branched`] operands whose [`TagGuard`] is a tag bit pattern.
//!
//! The tag mode splits elements into zero / positive / negative, and each op below drives a
//! slot per arm: mask negatives to zero, FMA the non-zeros, then clamp against a scale.
use super::*;
// Bit requirements read best unqualified inside a pattern literal.
use furiosa_opt_std::prelude::BitReq::{Ignore, One, Zero};

// Prefixed rather than reusing `P`/`A`/`T`: the parent module declares those names at other sizes,
// and a same-named axis here would lose to it in `pub use branch::*`, leaving this kernel's
// parameter types unnameable from an integration test.
axes![BP = 64, BA = 4, BT = 8];

type Chip = m![1];
type Cluster = m![1];

#[device(chip = 1, pe = 1)]
pub fn ve_elementwise_branched(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![BP, BA, BT]>,
    scale: &HbmTensor<f32, Chip, m![BP, BT]>,
) -> HbmTensor<f32, Chip, m![BP, BA, BT]> {
    let input_dm = input.to_dm::<Cluster, m![BP], m![BA, BT]>(&mut ctx.tdma);
    let scale_dm = scale.to_dm::<Cluster, m![BP], m![BT]>(&mut ctx.tdma);

    let scale_vrf: VrfTensor<f32, Chip, Cluster, m![BP], m![BT]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![BT]>()
        .collect::<m![1], m![BT]>()
        .to_vrf();

    let nonzero = TagGuard::not_matches([One, Ignore, Ignore, Ignore]);
    let negative = TagGuard::matches([Zero, Zero, One, Ignore]);

    let result: DmTensor<f32, Chip, Cluster, m![BP], m![BA, BT]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![BA], m![BT]>()
        .collect::<m![BA], m![BT]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0.0),
            Cmp::Greater(0.0),
            Cmp::Less(0.0),
            Cmp::LessUnsigned(-0.0),
        ]))
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(negative, 0.0f32))
        .vector_narrow_split::<m![BA, BT / 4], m![BT % 4]>()
        .vector_fp_ternary(FpTernaryOp::FmaF, Branched::imm(nonzero, (-2.0f32, 10.0f32)))
        .vector_widen_concat::<m![BA], m![BT]>()
        .vector_clip(
            ClipBinaryOpF32::Max,
            Branched::imm(negative, 1.0f32).rf(TagGuard::all(), &scale_vrf),
        )
        .vector_final()
        .commit_trim::<m![BT]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// Anchors the bit-order convention end to end: `cmp[i]` sets tag bit `i`, a guard's [`BitReq`] `i`
/// reads that bit, and the pass the compiler builds agrees with both.
///
/// Each guard constrains exactly one bit and leaves a distinct mark, so the output says which bit
/// fired. Reversing the bit order anywhere along the chain moves a mark to a different element, which
/// the test's hand-computed expectation catches. A host-versus-compiled comparison could not: both
/// sides read the pattern through the same accessor, so a change there moves them together.
///
/// The four comparisons do not overlap, so no mark hides another, and `BitOr`/`Min` are exact, which
/// keeps the expectation a plain integer.
///
/// Bit 1 is marked by a guarded *stash* read rather than an immediate, so the element lands back on its
/// original value -- which also pins that the slot read the stash and not the running stream.
#[device(chip = 1, pe = 1)]
pub fn ve_branch_bit_order(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![BP, BA, BT]>,
) -> HbmTensor<i32, Chip, m![BP, BA, BT]> {
    let input_dm = input.to_dm::<Cluster, m![BP], m![BA, BT]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![BP], m![BA, BT]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![BA], m![BT]>()
        .collect::<m![BA], m![BT]>()
        .vector_init()
        // One comparison per bit, none implying another, so every combination below is reachable.
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0),
            Cmp::Equal(5),
            Cmp::Less(10),
            Cmp::Greater(1000),
        ]))
        // Read back by the clip below, so this is the value the guarded slot restores.
        .vector_stash()
        // Three immediates, one per bit, in first-match order: bit0, then bit2, then bit3.
        .vector_logic(
            LogicBinaryOpI32::BitOr,
            Branched::imm(TagGuard::matches([One, Ignore, Ignore, Ignore]), 0x0100)
                .imm(TagGuard::matches([Ignore, Ignore, One, Ignore]), 0x0200)
                .imm(TagGuard::matches([Ignore, Ignore, Ignore, One]), 0x0400),
        )
        // Bit1, on the clip ALU: a pass has one logic unit and three immediates, both spent above.
        .vector_clip(
            ClipBinaryOpI32::Min,
            Branched::rf(TagGuard::matches([Ignore, One, Ignore, Ignore]), Stash),
        )
        .vector_final()
        .commit_trim::<m![BT]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
