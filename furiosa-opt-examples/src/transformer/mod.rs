use furiosa_opt_std::prelude::*;

pub mod axes;

pub(crate) mod kernel;

pub type Chip = m![1];

/// Number of transformer decoder layers in Qwen3-0.5B.
pub const LAYERS: usize = 28;

pub mod ops {
    use super::axes::*;
    use super::kernel::*;
    use super::*;

    type Cluster = m![1 # 2];
    type Slice = m![1 # 256];

    pub(crate) type SliceP4 = m![Dummy64, 1 # 4];
    pub(crate) type SliceN32 = m![N, 1 # 32];

    #[device(chip = 1)]
    pub fn embedding(ctx: &mut Context, input: &HbmTensor<bf16, Chip, m![H]>, out: &mut HbmTensor<bf16, Chip, m![H]>) {
        input.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
    }

    #[device(chip = 1)]
    pub fn projection(
        ctx: &mut Context,
        x: &HbmTensor<bf16, Chip, m![H]>,
        q_weight: &HbmTensor<bf16, Chip, m![Q, H]>,
        k_weight: &HbmTensor<bf16, Chip, m![P, H]>,
        v_weight: &HbmTensor<bf16, Chip, m![P, H]>,
        input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![D]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![D]>,
        kv_offset: &HbmTensor<i32, Chip, m![1]>,
        cos: &HbmTensor<bf16, Chip, m![D]>,
        sin: &HbmTensor<bf16, Chip, m![D]>,
        k_cache: &mut HbmTensor<bf16, Chip, m![T, N, D]>,
        v_cache: &mut HbmTensor<bf16, Chip, m![T, N, D]>,
        q_out: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
    ) {
        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = x.to_dm(&mut ctx.tdma);
        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = rmsnorm::forward(ctx, &x, input_rms_weight);

        let q: DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> = projection::proj_q(ctx, x.view(), q_weight);
        let k: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = projection::proj_k(ctx, x.view(), k_weight);
        projection::proj_v(ctx, x.view(), kv_offset, v_weight, v_cache);

        let q: DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> = rmsnorm::forward_q(ctx, &q, q_rms_weight);
        let k: DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> = rmsnorm::forward_k(ctx, &k, k_rms_weight);

        let q: DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> =
            rope::apply_rope(ctx, &q, &k, kv_offset, cos, sin, k_cache);
        q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    }

    #[device(chip = 1)]
    pub fn attention_forward_first(
        ctx: &mut Context,
        q: &HbmTensor<bf16, Chip, m![N, G, D]>,
        k: &HbmTensor<bf16, Chip, m![T, N, D]>,
        v: &HbmTensor<bf16, Chip, m![T, N, D]>,
        mask: &HbmTensor<f32, Chip, m![T]>,
        max_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
        sum_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
        out_hbm: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
    ) {
        attention::forward_first(ctx, q, k, v, mask, max_hbm, sum_hbm, out_hbm);
    }

    #[device(chip = 1)]
    pub fn attention_forward(
        ctx: &mut Context,
        q: &HbmTensor<bf16, Chip, m![N, G, D]>,
        k: &HbmTensor<bf16, Chip, m![T, N, D]>,
        v: &HbmTensor<bf16, Chip, m![T, N, D]>,
        mask: &HbmTensor<f32, Chip, m![T]>,
        max_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
        sum_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
        out_hbm: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
    ) {
        attention::forward(ctx, q, k, v, mask, max_hbm, sum_hbm, out_hbm);
    }

    #[device(chip = 1)]
    pub fn decoder(
        ctx: &mut Context,
        x: &HbmTensor<bf16, Chip, m![N, G, D]>,
        sum_hbm: &HbmTensor<f32, Chip, m![N, G]>,
        rx_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
        o_weight: &HbmTensor<bf16, Chip, m![H, Q]>,
        post_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
        up_weight: &HbmTensor<bf16, Chip, m![L, H]>,
        gate_weight: &HbmTensor<bf16, Chip, m![L, H]>,
        down_weight: &HbmTensor<bf16, Chip, m![H, L]>,
    ) {
        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![N, G, D]> = x.to_dm(&mut ctx.tdma);

        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![N, G, D]> = attention::norm(ctx, &x, sum_hbm);
        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![Q]> = unsafe { x.reshape() };

        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = projection::proj_o(ctx, x, o_weight);
        let rx: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = rx_hbm.to_dm(&mut ctx.tdma);
        let rx: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = residual::forward(ctx, &x, &rx);
        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = rmsnorm::forward(ctx, &rx, post_rms_weight);
        let x: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> =
            mlp::forward(ctx, x, up_weight, gate_weight, down_weight);

        let rx: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> = residual::forward(ctx, &x, &rx);
        let rx: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = unsafe { rx.reshape() };
        rx.view().to_hbm_view(&mut ctx.tdma, rx_hbm.view_mut());
    }

    #[device(chip = 1)]
    pub fn final_layer(
        ctx: &mut Context,
        input: &HbmTensor<bf16, Chip, m![H]>,
        rms_weight: &HbmTensor<bf16, Chip, m![H]>,
        lm_head_weight: &HbmTensor<bf16, Chip, m![W # 155648 / 8192, W # 155648 % 8192, H]>,
        out: &mut HbmTensor<bf16, Chip, m![Wp]>,
    ) {
        let x: DmTensor<bf16, Chip, lm_head::Cluster, lm_head::Slice, m![H]> = input.to_dm(&mut ctx.tdma);
        let x: DmTensor<bf16, Chip, lm_head::Cluster, lm_head::Slice, m![H]> = rmsnorm::forward(ctx, &x, rms_weight);
        let lm_head_weight: HbmTensorView<'_, bf16, Chip, m![Wp / 8192, Wp % 8192, H]> =
            unsafe { lm_head_weight.view().reshape() };

        lm_head::forward(ctx, &x, lm_head_weight, out);
    }
}
