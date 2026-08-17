use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::{D, G, N, T};

type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// Attention softmax scale, `sqrt(D::SIZE)` (applied as a division of the logits).
const SQRT_HEAD_DIM: f32 = 11.313_708;

/// Small epsilon added to reduction denominators to avoid division by zero.
const EPS: f32 = 6.25e-8;

/// Load a per-head `[N, G]` DM tensor into the VRF. This layout conversion is
/// needed every time a running max/sum/alpha value is consumed by a vector op.
fn ng_to_vrf<Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<f32, Chip, Cluster, Slice, m![N, G]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![N, G]> {
    ctx.sub
        .begin(x.view())
        .fetch::<m![N / 4], m![N % 4, G]>()
        .collect::<m![N / 4], m![N % 4, G]>()
        .to_vrf()
}

/// `q @ K^T` scaled by `1 / sqrt(D)`, producing per-head logits `[N, G, T]`.
///
/// The single query is broadcast across 16 slices so the `T` reduction can be
/// parallelized.
fn qk_scaled(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![N, G, D]>,
    k: &HbmTensor<bf16, Chip, m![T, N, D]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![N, G, T]> {
    type SliceQK = m![T / 8, 1 # 4];

    let q: DmTensor<bf16, Chip, Cluster, SliceQK, m![N, G, D]> = q.to_dm(&mut ctx.tdma);
    let k: DmTensor<bf16, Chip, Cluster, SliceQK, m![T % 8, N, D]> = k.to_dm(&mut ctx.tdma);
    let k_trf: TrfTensor<bf16, Chip, Cluster, SliceQK, m![T % 8], m![N, D]> = ctx
        .sub
        .begin(k.view())
        .fetch::<m![T % 8, N, D / 16], m![D % 16]>()
        .collect::<m![T % 8, N, D / 16], m![D % 16]>()
        .to_trf();

    let qk: DmTensor<bf16, Chip, Cluster, SliceQK, m![N, G, T % 8]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![N, G, D / 16], m![D % 16]>()
        .collect::<m![N, G, D / 16], m![D % 16]>()
        .contract_outer::<m![N, G, D / 32], m![D % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![N, G]>()
        .contract_lane::<m![N, G], m![T % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N, G, T / 4 % 2], m![T % 4]>()
        .vector_fp_div(SQRT_HEAD_DIM)
        .vector_widen_concat::<m![N, G], m![T % 8]>()
        .vector_final()
        .cast::<bf16, m![T % 8 # 16]>()
        .commit_trim::<m![T % 8]>()
        .commit();

    ctx.main
        .begin(qk.view())
        .fetch::<m![N, G], m![T % 8 # 16]>()
        .switch::<Slice, m![N, G, T / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![N, G, T / 8], m![T % 8 # 16]>()
        .commit_trim::<m![T % 8]>()
        .commit()
}

/// Per-head maximum of the logits, `[N, G]`.
fn row_max(
    ctx: &mut Context,
    qk: &DmTensor<bf16, Chip, Cluster, Slice, m![N, G, T]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![N, G]> {
    ctx.main
        .begin(qk.view())
        .fetch::<m![N, G, T / 16], m![T % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![N, G, T / 8], m![T % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N, G, T / 4], m![T % 4]>()
        .vector_intra_slice_reduce::<T, m![N, G], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![N], m![G # 8]>()
        .commit_trim::<m![G]>()
        .commit()
}

/// `exp(qk - max)` with the causal mask applied, `[N, G, T]`.
fn masked_exp(
    ctx: &mut Context,
    qk: &DmTensor<bf16, Chip, Cluster, Slice, m![N, G, T]>,
    max_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![N, G]>,
    mask: &HbmTensor<f32, Chip, m![T]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![N, G, T]> {
    let exp: DmTensor<f32, Chip, Cluster, Slice, m![N, G, T]> = ctx
        .main
        .begin(qk.view())
        .fetch::<m![N, G, T / 16], m![T % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![N, G, T / 8], m![T % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N, G, T / 4], m![T % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![N, G, T / 8], m![T % 8]>()
        .vector_final()
        .commit_trim::<m![T % 8]>()
        .commit();

    let mask: DmTensor<f32, Chip, Cluster, Slice, m![T]> = mask.to_dm(&mut ctx.tdma);
    let mask_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![T]> = ctx
        .sub
        .begin(mask.view())
        .fetch::<m![T / 8], m![T % 8]>()
        .collect::<m![T / 8], m![T % 8]>()
        .to_vrf();

    ctx.main
        .begin(exp.view())
        .fetch::<m![N, G, T / 8], m![T % 8]>()
        .collect::<m![N, G, T / 8], m![T % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
        .vector_final()
        .commit_trim::<m![T % 8]>()
        .commit()
}

/// Per-head sum of the masked exponentials, `[N, G]` (softmax denominator).
fn row_sum(
    ctx: &mut Context,
    masked_exp: &DmTensor<f32, Chip, Cluster, Slice, m![N, G, T]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![N, G]> {
    ctx.main
        .begin(masked_exp.view())
        .fetch::<m![N, G, T / 8], m![T % 8]>()
        .collect::<m![N, G, T / 8], m![T % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N, G, T / 4], m![T % 4]>()
        .vector_intra_slice_reduce::<T, m![N, G], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![N], m![G # 8]>()
        .commit_trim::<m![G]>()
        .commit()
}

/// `softmax_weights @ V`, producing the (unnormalized) attention output `[N, G, D]`.
fn weighted_sum(
    ctx: &mut Context,
    masked_exp: &DmTensor<f32, Chip, Cluster, Slice, m![N, G, T]>,
    v: &HbmTensor<bf16, Chip, m![T, N, D]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![N, G, D]> {
    type SliceQV = m![N, T / 16];

    let weight: DmTensor<bf16, Chip, Cluster, Slice, m![N, G, T]> = ctx
        .main
        .begin(masked_exp.view())
        .fetch::<m![N, G, T / 8], m![T % 8]>()
        .collect::<m![N, G, T / 8], m![T % 8]>()
        .cast::<bf16, m![T % 8 # 16]>()
        .commit_trim::<m![T % 8]>()
        .commit();
    let weight: DmTensor<bf16, Chip, Cluster, SliceQV, m![G, T % 16]> = weight.to_dm(&mut ctx.tdma);

    let v: DmTensor<bf16, Chip, Cluster, SliceQV, m![T % 16, D]> = v.to_dm(&mut ctx.tdma);
    let v: DmTensor<bf16, Chip, Cluster, SliceQV, m![D, T % 16]> = ctx
        .main
        .begin(v.view())
        .fetch::<m![D / 8, T % 16], m![D % 8 # 16]>()
        .collect::<m![D / 8, T % 16], m![D % 8 # 16]>()
        .transpose::<m![D / 8, T / 4 % 4, D % 8], m![T % 4 # 16]>()
        .commit_trim::<m![T % 4]>()
        .commit();
    let v_trf: TrfTensor<bf16, Chip, Cluster, SliceQV, m![D % 8], m![D / 8, T % 16]> = ctx
        .sub
        .begin(v.view())
        .fetch::<m![D % 8, D / 8], m![T % 16]>()
        .collect::<m![D % 8, D / 8], m![T % 16]>()
        .to_trf();

    let weighted_sum: DmTensor<bf16, Chip, Cluster, SliceQV, m![G, D]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![G], m![T % 16]>()
        .collect::<m![G], m![T % 16]>()
        .contract_outer::<m![G, D / 8], m![T % 16], _, _, _>(&v_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![G, D / 8]>()
        .contract_lane::<m![G, D / 8], m![D % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();
    let weighted_sum: DmTensor<bf16, Chip, Cluster, m![N, 1 # 32], m![G, D]> = ctx
        .main
        .begin(weighted_sum.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![N, 1 # 32], m![G, D / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();

    weighted_sum.to_dm(&mut ctx.tdma)
}

/// Attention over the first key/value block: initializes the running max, sum
/// and output that later blocks accumulate into.
#[expect(clippy::too_many_arguments)]
pub(crate) fn forward_first(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![N, G, D]>,
    k: &HbmTensor<bf16, Chip, m![T, N, D]>,
    v: &HbmTensor<bf16, Chip, m![T, N, D]>,
    mask: &HbmTensor<f32, Chip, m![T]>,
    max_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    sum_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
) {
    let qk = qk_scaled(ctx, q, k);
    let max = row_max(ctx, &qk);
    let max_vrf = ng_to_vrf(ctx, &max);

    let exp = masked_exp(ctx, &qk, &max_vrf, mask);
    let sum = row_sum(ctx, &exp);
    let result = weighted_sum(ctx, &exp, v);

    max.view().to_hbm_view(&mut ctx.tdma, max_hbm.view_mut());
    sum.view().to_hbm_view(&mut ctx.tdma, sum_hbm.view_mut());
    result.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());
}

/// Attention over a subsequent key/value block, folding its contribution into
/// the running (online-softmax) max, sum and output stored in HBM.
#[expect(clippy::too_many_arguments)]
pub(crate) fn forward(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![N, G, D]>,
    k: &HbmTensor<bf16, Chip, m![T, N, D]>,
    v: &HbmTensor<bf16, Chip, m![T, N, D]>,
    mask: &HbmTensor<f32, Chip, m![T]>,
    max_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    sum_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
) {
    let qk = qk_scaled(ctx, q, k);
    let max = row_max(ctx, &qk);

    // Combine this block's max with the running max, and derive the rescale
    // factor `alpha = exp(old_max - new_max)` for the accumulated sum/output.
    let old_max: DmTensor<f32, Chip, Cluster, Slice, m![N, G]> = max_hbm.to_dm(&mut ctx.tdma);
    let old_max_vrf = ng_to_vrf(ctx, &old_max);

    let max: DmTensor<f32, Chip, Cluster, Slice, m![N, G]> = ctx
        .main
        .begin(max.view())
        .fetch::<m![N / 4], m![N % 4, G]>()
        .collect::<m![N / 4], m![N % 4, G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &old_max_vrf)
        .vector_final()
        .commit_trim::<m![N % 4, G]>()
        .commit();

    max.view().to_hbm_view(&mut ctx.tdma, max_hbm.view_mut());

    let alpha: DmTensor<f32, Chip, Cluster, Slice, m![N, G]> = ctx
        .main
        .begin(max.view())
        .fetch::<m![N / 4], m![N % 4, G]>()
        .collect::<m![N / 4], m![N % 4, G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N / 2], m![N % 2, G]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, &old_max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![N / 4], m![N % 4, G]>()
        .vector_final()
        .commit_trim::<m![N % 4, G]>()
        .commit();
    let alpha_vrf = ng_to_vrf(ctx, &alpha);
    let max_vrf = ng_to_vrf(ctx, &max);

    let exp = masked_exp(ctx, &qk, &max_vrf, mask);
    let expsum = row_sum(ctx, &exp);
    let expsum_vrf = ng_to_vrf(ctx, &expsum);

    // sum = old_sum * alpha + expsum
    let sum: DmTensor<f32, Chip, Cluster, Slice, m![N, G]> = sum_hbm.to_dm(&mut ctx.tdma);
    let sum: DmTensor<f32, Chip, Cluster, Slice, m![N, G]> = ctx
        .main
        .begin(sum.view())
        .fetch::<m![N / 4], m![N % 4, G]>()
        .collect::<m![N / 4], m![N % 4, G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N / 2], m![N % 2, G]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &alpha_vrf)
        .vector_widen_concat::<m![N / 4], m![N % 4, G]>()
        .vector_clip(ClipBinaryOpF32::Add, &expsum_vrf)
        .vector_final()
        .commit_trim::<m![N % 4, G]>()
        .commit();

    sum.view().to_hbm_view(&mut ctx.tdma, sum_hbm.view_mut());

    let result = weighted_sum(ctx, &exp, v);
    let result_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![N, G, D]> = ctx
        .sub
        .begin(result.view())
        .fetch::<m![N, G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![N, G, D / 8], m![D % 8]>()
        .to_vrf();

    // out = old_out * alpha + result
    let out: DmTensor<bf16, Chip, Cluster, Slice, m![N, G, D]> = out_hbm.to_dm(&mut ctx.tdma);
    let out: DmTensor<bf16, Chip, Cluster, Slice, m![N, G, D]> = ctx
        .main
        .begin(out.view())
        .fetch::<m![N, G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![N, G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N, G, D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &alpha_vrf)
        .vector_widen_concat::<m![N, G, D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();
    let out: DmTensor<bf16, Chip, Cluster, Slice, m![N, G, D]> = ctx
        .main
        .begin(out.view())
        .fetch::<m![N, G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![N, G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &result_vrf)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();

    out.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());
}

/// Normalize the accumulated attention output by the running softmax sum. Called from the
/// decoder kernel rather than `forward`/`forward_first`, since `sum` is only final once every
/// KV chunk has been folded in.
pub(crate) fn norm<Slice: M>(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![N, G, D]>,
    sum_hbm: &HbmTensor<f32, Chip, m![N, G]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![N, G, D]> {
    let sum: DmTensor<f32, Chip, Cluster, Slice, m![N, G]> = sum_hbm.to_dm(&mut ctx.tdma);
    let sum_vrf = ng_to_vrf(ctx, &sum);

    ctx.main
        .begin(input.view())
        .fetch::<m![N, G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![N, G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N, G, D / 4], m![D % 4]>()
        .vector_fp_div(&sum_vrf)
        .vector_widen_concat::<m![N, G, D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}
