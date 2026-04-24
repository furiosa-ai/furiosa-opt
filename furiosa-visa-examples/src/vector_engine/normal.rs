use super::*;

#[device(chip = 1)]
pub fn ve_elementwise_fxp_const(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// **NOTE**: This example demonstrates an ALU conflict and should PANIC:
/// - AddFxp uses FxpAdd ALU
/// - MulInt uses FxpMul ALU
/// - SubFxp uses FxpAdd ALU (conflict with first AddFxp!)
///
/// Expected panic: "FxpAdd is already in use"
#[device(chip = 1)]
pub fn ve_elementwise_fxp_chain(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, 10)
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_fxp(FxpBinaryOp::SubFxp, 5)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_full_pipeline(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_fxp_to_fp(31)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.5f32)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max, 0)
        .vector_clip(ClipBinaryOpI32::Min, 1000)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_stash_f32(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![A]>) -> HbmTensor<f32, Chip, m![A]> {
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
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.0f32)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, Stash)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// i32 stash example: stash input, multiply by 2, then max with stashed value
#[device(chip = 1)]
pub fn ve_elementwise_stash_i32(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_stash()
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_clip(ClipBinaryOpI32::Max, Stash)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_fp_unary_with_mode(
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
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_unary_with_mode(FpUnaryOp::Exp, UnaryArgMode::Mode0)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_fp_binary_with_mode(
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
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, 2.0f32)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_fp_ternary_with_mode(
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
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_ternary_with_mode(FpTernaryOp::FmaF, TernaryArgMode::Mode120, (2.0f32, 1.0f32))
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_logic_with_mode(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_logic_with_mode(LogicBinaryOpI32::BitXor, BinaryArgMode::Mode11, 0xff)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_fxp_with_mode(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_fp_div_with_mode(
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
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_div_with_mode(FpDivBinaryOp::DivF, BinaryArgMode::Mode10, 2.0f32)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_clip_with_mode(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_clip_with_mode(ClipBinaryOpI32::AddFxp, BinaryArgMode::Mode11, 3)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_group_pair_fp_zip_with_mode(
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
        .vector_fp_zip_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10)
        .vector_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x4000)
}

// Stash at fxp stage, read at clip stage (i32 -> i32)
// input * 2, then max(result, stashed_input)
#[device(chip = 1)]
pub fn ve_stash_fxp_fxp(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_stash()
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_clip(ClipBinaryOpI32::Max, Stash)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// Stash at fxp stage, read at fp stage (i32 stash -> f32 read via fxp_to_fp)
// input * 2 (fxp), convert to fp, then add stashed_input (reinterpreted as f32)
#[device(chip = 1)]
pub fn ve_stash_fxp_fp(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_stash()
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_fxp_to_fp(8)
        .vector_clip(ClipBinaryOpF32::Max, Stash)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// Stash at fp stage, read at clip stage (f32 -> f32)
// input * 2.0, stash, then * 3.0, then max(result, stashed)
#[device(chip = 1)]
pub fn ve_stash_fp_fp(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![A]>) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.0f32)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 3.0f32)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, Stash)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// Stash at fp stage, read at clip stage after fp_to_fxp (f32 stash -> i32 read)
// input * 2.0 (fp), stash, convert to fxp, then max(result, stashed reinterpreted as i32)
#[device(chip = 1)]
pub fn ve_stash_fp_fxp(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.0f32)
        .vector_stash()
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Max, Stash)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_logic(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0xFF)
        .vector_logic(LogicBinaryOpI32::BitOr, 0x100)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn ve_elementwise_vrf(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, B]>,
    vrf_data: &HbmTensor<i32, Chip, m![B]>,
) -> HbmTensor<i32, Chip, m![A, B]> {
    let input_intermediate: HbmTensor<i32, Chip, m![B, A]> = input.to_hbm(&mut ctx.tdma, 0x8000);
    let input_dm = input_intermediate.to_dm::<Cluster, m![A / 2], m![B, A % 2]>(&mut ctx.tdma, 0x1000);
    let vrf_dm = vrf_data.to_dm::<Cluster, m![A / 2], m![B]>(&mut ctx.tdma, 0x2000);

    let vrf: VrfTensor<i32, Chip, Cluster, m![A / 2], m![B]> = ctx
        .sub
        .begin(vrf_dm.view())
        .fetch::<i32, m![1], m![B]>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf(0);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![B, A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![B], m![A % 2]>()
        .collect::<m![B], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, &vrf)
        .vector_final()
        .commit(0x3000);

    result.to_hbm(&mut ctx.tdma, 0x5000)
}

#[device(chip = 1)]
pub fn ve_elementwise_multi_vrf(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, B]>,
    vrf_data1: &HbmTensor<i32, Chip, m![B]>,
    vrf_data2: &HbmTensor<i32, Chip, m![B]>,
) -> HbmTensor<i32, Chip, m![A, B]> {
    let input_intermediate: HbmTensor<i32, Chip, m![B, A]> = input.to_hbm(&mut ctx.tdma, 0x0);
    let input_dm = input_intermediate.to_dm::<Cluster, m![A / 2], m![B, A % 2]>(&mut ctx.tdma, 0x1000);
    let vrf_dm1 = vrf_data1.to_dm::<Cluster, m![A / 2], m![B]>(&mut ctx.tdma, 0x2000);
    let vrf_dm2 = vrf_data2.to_dm::<Cluster, m![A / 2], m![B]>(&mut ctx.tdma, 0x3000);

    let vrf1: VrfTensor<i32, Chip, Cluster, m![A / 2], m![B]> = ctx
        .sub
        .begin(vrf_dm1.view())
        .fetch::<i32, m![1], m![B]>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf(0);

    let vrf2: VrfTensor<i32, Chip, Cluster, m![A / 2], m![B]> = ctx
        .sub
        .begin(vrf_dm2.view())
        .fetch::<i32, m![1], m![B]>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf(1024);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![B, A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![B], m![A % 2]>()
        .collect::<m![B], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, &vrf1)
        .vector_fxp(FxpBinaryOp::MulInt, &vrf2)
        .vector_clip(ClipBinaryOpI32::AddFxp, &vrf1)
        .vector_final()
        .commit(0x4000);

    result.to_hbm(&mut ctx.tdma, 0x5000)
}

// =============================================================================
// Group pair operations (ve_group_pair_*)
// =============================================================================
