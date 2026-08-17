use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::{D, G, H};
use crate::transformer::ops::SliceN32;

type Cluster = m![1 # 2];

pub(crate) fn forward<Cluster: M, Slice: M>(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let ms: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(input.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(1024.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 6.25e-8f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let weight: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![H]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(input.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn forward_q(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]>,
    rms_weight: &HbmTensor<bf16, Chip, m![D]>,
) -> DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> {
    let ms: DmTensor<f32, Chip, Cluster, SliceN32, m![G]> = ctx
        .main
        .begin(input.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G, D / 4], m![D % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<D, m![G], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(128f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 6.25e-8f32)
        .vector_final()
        .transpose::<m![1], m![G # 8]>()
        .commit_trim::<m![G]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, SliceN32, m![G]> = ctx
        .main
        .begin(ms.view())
        .fetch::<m![1], m![G # 8]>()
        .collect::<m![1], m![G # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![G # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![G # 8]>()
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();

    let weight: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![G # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![G # 8]>()
        .collect::<m![1], m![G # 8]>()
        .to_vrf();

    ctx.main
        .begin(input.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G, D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![G, D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

pub(crate) fn forward_k(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, SliceN32, m![D]>,
    rms_weight: &HbmTensor<bf16, Chip, m![D]>,
) -> DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> {
    let ms: DmTensor<f32, Chip, Cluster, SliceN32, m![1 # 8]> = ctx
        .main
        .begin(input.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<D, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(128f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 6.25e-8f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, SliceN32, m![1 # 8]> = ctx
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let weight: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(input.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}
