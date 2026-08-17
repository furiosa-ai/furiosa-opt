use super::*;

#[device(chip = 1)]
pub fn ve_group_pair_add(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// `2a + 4b` on interleaved `f32` inputs, with each doubling done as `exponent += 1` on the bit
/// pattern instead of a float multiply.
///
/// Pair mode reads the two groups as one stream, so each `vector_reinterpret` covers both: the
/// per-group integer adds land while the stream is read as `i32`, and the `f32` view is back in place
/// for the float zip.
#[device(chip = 1)]
pub fn ve_group_pair_reinterpret_scale_f32(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_reinterpret::<i32>()
        .vector_fxp(FxpBinaryOp::AddFxp, 1 << 23, 2 << 23)
        .vector_reinterpret::<f32>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// `|a| + |b|` on `f32`, with the sign bit cleared per group by the Logic cluster.
///
/// The per-group `vector_logic` masks each group with its own operand while the stream is read as
/// `i32`, then the `f32` view returns for the clip zip that adds the two groups.
#[device(chip = 1)]
pub fn ve_group_pair_logic_abs_add_f32(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_preprocess_both(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, 2, 3)
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_preprocess_g0(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, 10, ())
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_preprocess_g1(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, (), 10)
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// An op *after* the zip, with a bare operand. The zip merges the two groups, so from there on the
/// tag carries the group bit and nothing else, and a bare operand is unconditional -- it is not
/// narrowed to the group the zip wrote. This is the only shape that pins that, so it is the one
/// `compare_edf` reads to check host and device agree on it.
#[device(chip = 1)]
pub fn ve_group_pair_post_zip(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_clip(ClipBinaryOpI32::Max, 0)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_chain(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::AddFxp, 10, 20)
        .vector_fxp(FxpBinaryOp::MulInt, 2, 3)
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
#[device(chip = 1)]
pub fn ve_group_pair_fxp(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_zip(FxpBinaryOp::MulInt)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_logic(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_logic_zip(LogicBinaryOpI32::BitXor)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_fp(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// Same pipeline as `ve_group_pair_fp`, but with an extra `Q` axis threaded
/// through the stream time so that there are multiple packets per partition.
/// Used to verify that the buffered `split` / `concat` pipeline stays correct
/// when the per-partition slice spans more than one Way8 flit.
#[device(chip = 1)]
pub fn ve_group_pair_fp_multi_packet(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![Q, A]>,
    rhs: &HbmTensor<i32, Chip, m![Q, A]>,
) -> HbmTensor<f32, Chip, m![Q, A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![Q, A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![Q, A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![Q, A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![Q, I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![Q, I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![Q, 1 # 2], m![Q]>()
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![Q, 1 # 2], m![A % 2 # 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Q], m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_unary(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt, true, true)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn ve_group_pair_unary_selective(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Exp, true, false)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

// =============================================================================
// Ternary operations (VectorTensor and VectorTensorPair)
// =============================================================================

/// Ternary operation example using VectorTensor with tuple syntax.
/// output = input * 2.0 + 3.0  (using MulAdd: a*b+c where a=input, b=2.0, c=3.0)
#[device(chip = 1)]
pub fn ve_elementwise_ternary(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![A]>) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![A % 2 # 4]>()
        // Using tuple syntax: (operand0, operand1) where operand0 is f32 constant
        // FmaF: data * operand0 + operand1 = input * 2.0 + 3.0
        .vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 3.0f32))
        .vector_widen_pad::<m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// Ternary operation with stash as operand0.
/// First stash input, then compute: input * stash + 1.0
/// Since stash = input, this computes input * input + 1.0 = input^2 + 1.0
#[device(chip = 1)]
pub fn ve_elementwise_ternary_stash(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // The stash is written inside the 4-way region, the way the ternary reads it at.
        .vector_narrow_trim::<m![A % 2 # 4]>()
        .vector_stash()
        // Using Stash as operand0: data * stash + 1.0 = input * input + 1.0
        .vector_fp_ternary(FpTernaryOp::FmaF, (Stash, 1.0f32))
        .vector_widen_pad::<m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// VectorTensorPair ternary operation example.
/// group0: lhs * 2.0 + 1.0
/// group1: rhs * 3.0 + 2.0
/// Then combine with MulF.
#[device(chip = 1)]
pub fn ve_group_pair_ternary(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 1.0f32), (3.0f32, 2.0f32))
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// VectorTensorPair ternary operation with selective groups.
/// group0: lhs * 2.0 + 1.0
/// group1: no ternary operation (pass through)
/// Then combine with AddF.
#[device(chip = 1)]
pub fn ve_group_pair_ternary_selective(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![1 # 2], m![A % 2 # 4]>()
        // Using () to skip group1
        .vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 1.0f32), ())
        .vector_fp_zip(FpBinaryOp::DivF)
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

// =============================================================================
// Intra-slice reduce operations (ve_intra_slice_reduce_*)
// =============================================================================
