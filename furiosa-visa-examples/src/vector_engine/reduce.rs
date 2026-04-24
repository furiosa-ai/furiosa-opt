use super::*;

#[device(chip = 1)]
pub fn ve_intra_slice_reduce_add_fxp_sat(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, R]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_transposed: HbmTensor<i32, Chip, m![R, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![R, A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![R], m![A % 2]>()
        .collect::<m![R], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Intra-slice reduce: max (i32)
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_max_i32(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, R]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_transposed: HbmTensor<i32, Chip, m![R, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![R, A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![R], m![A % 2]>()
        .collect::<m![R], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(IntraSliceReduceOpI32::Max)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Intra-slice reduce: min (i32)
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_min_i32(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, R]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_transposed: HbmTensor<i32, Chip, m![R, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![R, A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![R], m![A % 2]>()
        .collect::<m![R], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(IntraSliceReduceOpI32::Min)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Intra-slice reduce: add (f32)
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_add_f32(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A, R]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_transposed: HbmTensor<f32, Chip, m![R, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![R, A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![R], m![A % 2]>()
        .collect::<m![R], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Intra-slice reduce: max (f32)
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_max_f32(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A, R]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_transposed: HbmTensor<f32, Chip, m![R, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![R, A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![R], m![A % 2]>()
        .collect::<m![R], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Intra-slice reduce: min (f32)
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_min_f32(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A, R]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_transposed: HbmTensor<f32, Chip, m![R, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![R, A % 2]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![R], m![A % 2]>()
        .collect::<m![R], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(IntraSliceReduceOpF32::Min)
        .vector_pad_way8::<m![A % 2 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// =============================================================================
// Intra-slice reduce: R split across Slice and Time (ve_intra_slice_reduce_split_*)
// =============================================================================

/// Intra-slice reduce with reduce axis S split across Slice and Time.
/// S = 15, padded to 16: Slice gets S # 16 / 4 (=4), Time gets S # 16 % 4 (=4).
/// Note: only the Time portion of S is reduced; Slice portion (S # 16 / 4) remains.
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_split_slice_time(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![S, A]>,
) -> HbmTensor<i32, Chip, m![S # 16 / 4, A]> {
    let input_transposed: HbmTensor<i32, Chip, m![S, A]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm =
        input_transposed.to_dm::<Cluster, m![S # 16 / 4, A / 8], m![S # 16 % 4, A % 8]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![S # 16 / 4, A / 8], m![A % 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![S # 16 % 4], m![A % 8]>()
        .collect::<m![S # 16 % 4], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S # 16 % 4, A / 4 % 2], m![A % 4]>()
        .vector_intra_slice_reduce::<S, m![A / 4 % 2], m![A % 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Intra-slice reduce with R split across Time and Packet.
/// R = 4: Time = m![R / 2], Packet = m![R % 2, A % 2 # 8].
/// C2 satisfied: Packet has no padding on reduce axis (R % 2 = 2, no padding).
#[device(chip = 1)]
pub fn ve_intra_slice_reduce_split_time_packet(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![R16, A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_transposed: HbmTensor<i32, Chip, m![A, R16]> = input.to_hbm(&mut ctx.tdma, 0x4000);
    let input_dm = input_transposed.to_dm::<Cluster, m![A / 2], m![A % 2, R16]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2, 1 # 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![A % 2, R16 / 8], m![R16 % 8]>()
        .collect::<m![A % 2, R16 / 8], m![R16 % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![A % 2, R16 / 4], m![R16 % 4]>()
        .vector_intra_slice_reduce::<R16, m![A % 2], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_pad_way8::<m![1 # 8]>()
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// =============================================================================
// Inter-slice reduce operations (ve_inter_slice_reduce_*)
// Uses VRU (Vector Reduce Unit) — reduces across slices.
// Path 3: VRU only (vector_init → vector_inter_slice_reduce → vector_final → commit)
// =============================================================================

/// Inter-slice reduce: saturating add (i32), VRU-only path.
/// Input [R, A] with R in Slice → VRU reduces R across slices → Output [A]
#[device(chip = 1)]
pub fn ve_inter_slice_reduce_add_sat_i32(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![R, A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 8, R], m![A % 8]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 8, 1 # 4], m![A % 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Inter-slice reduce: max (i32), VRU-only path.
#[device(chip = 1)]
pub fn ve_inter_slice_reduce_max_i32(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![R, A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 8, R], m![A % 8]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 8, 1 # 4], m![A % 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::Max)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Inter-slice reduce: add (f32), VRU-only path.
#[device(chip = 1)]
pub fn ve_inter_slice_reduce_add_f32(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![R, A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 8, R], m![A % 8]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![A / 8, 1 # 4], m![A % 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// =============================================================================
// VRU → VAU (vector_inter_slice_reduce → vector_intra_slice_branch → VAU ops)
// =============================================================================

/// VRU first (inter-slice reduce), then VAU (add constant).
/// Input [R, A] → VRU reduces R → VAU adds 100 → Output [A]
#[device(chip = 1)]
pub fn ve_vru_then_vau_i32(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![R, A]>) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 8, R], m![A % 8]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<i32, Chip, Cluster, m![A / 8, 1 # 4], m![A % 8]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<i32, m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

// =============================================================================
// Axis promotion: InSlice axis moves to Partitioning after VRU reduce
// =============================================================================

/// VRU with axis promotion: T moves from InSlice to Partitioning.
///
/// Input:  Slice=[W, R]=64*4=256, Element=[T, P]=4*8=32
/// InSlice = [T_1=4, P_1=8]
/// After VRU: Slice2=[W, T]=64*4=256, Time2=[1]
/// T_1=4 was in InSlice, now in Partitioning → promotion
#[device(chip = 1)]
pub fn ve_inter_slice_reduce_promote_f32(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![W, R, T, P]>,
) -> HbmTensor<f32, Chip, m![W, T, P]> {
    let input_dm = input.to_dm::<Cluster, m![W, R], m![T, P]>(&mut ctx.tdma, 0x1000);

    let result: DmTensor<f32, Chip, Cluster, m![W, T], m![P]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<f32, m![T], m![P]>()
        .collect::<m![T], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![W, T], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit(0x2000);

    result.to_hbm(&mut ctx.tdma, 0x3000)
}
