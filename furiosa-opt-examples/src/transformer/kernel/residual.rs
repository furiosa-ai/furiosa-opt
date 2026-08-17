use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::H;

pub(crate) fn forward<Cluster: M, Slice: M>(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let residual_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![H]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(input.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
