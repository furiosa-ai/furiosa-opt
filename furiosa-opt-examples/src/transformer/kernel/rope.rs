use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::{D, G, N, T};
use crate::transformer::ops::SliceN32;

type Cluster = m![1 # 2];

pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]>,
    k: &DmTensor<bf16, Chip, Cluster, SliceN32, m![D]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![T, N, D]>,
) -> DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> {
    let cos: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = cos.to_dm(&mut ctx.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = sin.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .sub
        .begin(cos.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .sub
        .begin(sin.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();

    let fst_half_q = q.view().tile::<m![D], 64, m![G, D = 64 # 128]>(0);
    let snd_half_q = q.view().tile::<m![D], 64, m![G, D = 64 # 128]>(64);

    let mut rotate_half_q: DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> = DmTensor::new();

    ctx.main
        .begin(fst_half_q)
        .fetch::<m![G], m![D = 64]>()
        .collect::<m![G, D = 64 / 16], m![D = 64 % 16]>()
        .commit_trim::<m![D = 64 % 16]>()
        .commit_view(rotate_half_q.view_mut().tile::<m![D], 64, m![G, D = 64 #{!} 128]>(64));

    ctx.main
        .begin(snd_half_q)
        .fetch::<m![G], m![D = 64]>()
        .collect::<m![G, D = 64 / 16], m![D = 64 % 16]>()
        .commit_trim::<m![D = 64 % 16]>()
        .commit_view(rotate_half_q.view_mut().tile::<m![D], 64, m![G, D = 64 #{!} 128]>(0));

    let fst_half_k = k.view().tile::<m![D], 64, m![D = 64 # 128]>(0);
    let snd_half_k = k.view().tile::<m![D], 64, m![D = 64 # 128]>(64);

    let mut rotate_half_k: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = DmTensor::new();

    ctx.main
        .begin(fst_half_k)
        .fetch::<m![1], m![D = 64]>()
        .collect::<m![D = 64 / 16], m![D = 64 % 16]>()
        .commit_trim::<m![D = 64 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![D], 64, m![D = 64 #{!} 128]>(64));

    ctx.main
        .begin(snd_half_k)
        .fetch::<m![1], m![D = 64]>()
        .collect::<m![D = 64 / 16], m![D = 64 % 16]>()
        .commit_trim::<m![D = 64 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![D], 64, m![D = 64 #{!} 128]>(0));

    let tcq: DmTensor<f32, Chip, Cluster, SliceN32, m![G, D]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G, D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![G, D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();

    let tsq: DmTensor<f32, Chip, Cluster, SliceN32, m![G, D]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G, D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![G, D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();

    let tsq_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![G, D]> = ctx
        .sub
        .begin(tsq.view())
        .fetch::<m![G, D / 8], m![D % 8]>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .to_vrf();

    let result_q: DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> = ctx
        .main
        .begin(tcq.view())
        .fetch::<m![G, D / 8], m![D % 8]>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &tsq_vrf)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();

    let tck: DmTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();

    let tsk: DmTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();

    let tsk_vrf: VrfTensor<f32, Chip, Cluster, SliceN32, m![D]> = ctx
        .sub
        .begin(tsk.view())
        .fetch::<m![D / 8], m![D % 8]>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = ctx
        .main
        .begin(tck.view())
        .fetch::<m![D / 8], m![D % 8]>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &tsk_vrf)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();

    result_k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);

    result_q
}
