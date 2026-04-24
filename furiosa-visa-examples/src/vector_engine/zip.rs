use super::*;

#[device(chip = 1)]
pub fn ve_group_pair_add(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_preprocess_both(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, 2, 3)
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_preprocess_g0(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, 10, ())
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_preprocess_g1(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, (), 10)
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_chain(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::AddFxp, 10, 20)
        .vector_fxp(FxpBinaryOp::MulInt, 2, 3)
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}
#[device(chip = 1)]
pub fn ve_group_pair_fxp(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_zip(FxpBinaryOp::MulInt)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_logic(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_logic_zip(LogicBinaryOpI32::BitXor)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_fp(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_to_fp(31)
        .vector_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
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
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![Q, A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![Q, A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![Q, A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![Q, I], m![A % 2]>()
        .collect::<m![Q, I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![Q, 1 # 2], m![Q]>()
        .vector_fxp_to_fp(31)
        .vector_split::<m![Q, 1 # 2], m![A % 2 # 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_concat::<m![Q], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_unary(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_to_fp(31)
        .vector_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt, true, true)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

#[device(chip = 1)]
pub fn ve_group_pair_unary_selective(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<i32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp_to_fp(31)
        .vector_split::<m![1 # 2], m![A % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Exp, true, false)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

// =============================================================================
// Ternary operations (VectorTensor and VectorTensorPair)
// =============================================================================

/// Ternary operation example using VectorTensor with tuple syntax.
/// output = input * 2.0 + 3.0  (using MulAdd: a*b+c where a=input, b=2.0, c=3.0)
#[device(chip = 1)]
pub fn ve_elementwise_ternary(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![A]>) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        // Using tuple syntax: (operand0, operand1) where operand0 is f32 constant
        // FmaF: data * operand0 + operand1 = input * 2.0 + 3.0
        .vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 3.0f32))
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Ternary operation with stash as operand0.
/// First stash input, then compute: input * stash + 1.0
/// Since stash = input, this computes input * input + 1.0 = input^2 + 1.0
#[device(chip = 1)]
pub fn ve_elementwise_ternary_stash(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_stash()
        .vector_trim_way4::<m![A % 2 # 4]>()
        // Using Stash as operand0: data * stash + 1.0 = input * input + 1.0
        .vector_fp_ternary(FpTernaryOp::FmaF, (Stash, 1.0f32))
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
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
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<f32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_split::<m![1 # 2], m![A % 2 # 4]>()
        // Using IntoGroupTernaryOperand: (f32, f32) tuple for each group
        .vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 1.0f32), (3.0f32, 2.0f32))
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
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
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x2000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<f32, m![I], m![A % 2]>()
        .collect::<m![I], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_split::<m![1 # 2], m![A % 2 # 4]>()
        // Using () to skip group1
        .vector_fp_ternary(FpTernaryOp::FmaF, (2.0f32, 1.0f32), ())
        .vector_fp_zip(FpBinaryOp::DivF)
        .vector_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

// =============================================================================
// Intra-slice reduce operations (ve_intra_slice_reduce_*)
// =============================================================================
