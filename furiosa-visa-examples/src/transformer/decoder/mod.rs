//! Composable kernel: `transformer_0:output_projection..transformer_1:qkv_projection`
//!
//! Pipeline: O-Projection → Residual Add → RMS Norm → MLP (gate+up+SiLU+down)
//!           → Residual Add → RMS Norm → Q/K/V Projection → RoPE
//!
//! Qwen 2.5 0.5B, 4 PE, W16A16KV16 (bf16), tokenwise seq_len=128.

use crate::transformer::common::{mlp, norm, o_proj, rope};
use crate::transformer::embedding::proj;

use crate::transformer::axes::{
    D,             // = 64, head_dim
    H,             // = 896, hidden_size
    K,             // = 128, k_proj
    M,             // = 4864, mlp_intermediate
    P,             // = 32768, max_position
    Q,             // = 896, q_proj
    R,             // = 2, rope_rot
    S_decode as S, // = 128, sequence length
    V,             // = 128, v_proj
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];

#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &mut Context,
    // Inputs (arg0..arg15)
    matmul_score_v: &HbmTensor<bf16, Chip, m![S, H]>,       // arg0
    o_proj_weight: &HbmTensor<bf16, Chip, m![H, H]>,        // arg1
    hidden_states: &HbmTensor<bf16, Chip, m![S, H]>,        // arg2
    norm_weight_0: &HbmTensor<bf16, Chip, m![H]>,           // arg3
    gate_weight: &HbmTensor<bf16, Chip, m![M, H]>,          // arg4
    up_weight: &HbmTensor<bf16, Chip, m![M, H]>,            // arg5
    down_weight: &HbmTensor<bf16, Chip, m![H, M]>,          // arg6
    norm_weight_1: &HbmTensor<bf16, Chip, m![H]>,           // arg7
    q_weight: &HbmTensor<bf16, Chip, m![Q, H]>,             // arg8
    q_bias: &HbmTensor<bf16, Chip, m![Q]>,                  // arg9
    k_weight: &HbmTensor<bf16, Chip, m![K, H]>,             // arg10
    _k_bias: &HbmTensor<bf16, Chip, m![K]>,                 // arg11
    v_weight: &HbmTensor<bf16, Chip, m![V, H]>,             // arg12
    _v_bias: &HbmTensor<bf16, Chip, m![V]>,                 // arg13
    rope_table: &HbmTensor<bf16, Chip, m![P, D / 2, R, R]>, // arg14
    position_ids: &HbmTensor<i32, Chip, m![S]>,             // arg15
    // Outputs (out#0..out#3, pre-allocated by caller)
    out_q: &mut HbmTensor<bf16, Chip, m![S, Q]>,      // out#0
    out_k: &mut HbmTensor<bf16, Chip, m![S, K]>,      // out#1
    out_v: &mut HbmTensor<bf16, Chip, m![S, V]>,      // out#2
    out_hidden: &mut HbmTensor<bf16, Chip, m![S, H]>, // out#3
) {
    // PyTorch Qwen2Decoder.forward() equivalent:
    //   residual = hidden_states
    //   hidden_states = input_layernorm(hidden_states)
    //   hidden_states = self_attn(hidden_states, ...)
    //   hidden_states = residual + hidden_states
    //   residual = hidden_states
    //   hidden_states = post_attention_layernorm(hidden_states)
    //   hidden_states = mlp(hidden_states)
    //   hidden_states = hidden_states + residual
    //
    // This kernel spans two decoder layers: it finishes layer N (o_proj → MLP)
    // and starts layer N+1 (QKV projection → RoPE).

    // Phase 1: O-projection — output = o_proj(attn_output)  [bias=False]
    let o_proj_out = o_proj::o_proj(ctx, matmul_score_v, o_proj_weight);
    // Phase 2: residual + post_attention_layernorm (fused)
    //   hidden = residual + o_proj_out; hidden = RMSNorm(hidden)
    let (normalized_0, residual_0) = norm::residual_norm(ctx, hidden_states, &o_proj_out, norm_weight_0);
    // Phase 3: MLP — silu(gate_proj(x)) * up_proj(x), then down_proj
    let mlp_out = mlp::mlp(ctx, &normalized_0, gate_weight, up_weight, down_weight);
    // Phase 4: residual + input_layernorm (fused, starts next layer)
    //   hidden = residual + mlp_out; hidden = RMSNorm(hidden)
    let normalized_1 = norm::residual_norm_post(ctx, &residual_0, &mlp_out, norm_weight_1, out_hidden);
    // Phase 5: QKV projections
    //   PyTorch: q = q_proj(x) [bias=True], k = k_proj(x) [bias=True], v = v_proj(x) [bias=True]
    //   K/V bias is omitted in this kernel path
    proj::v_proj(ctx, &normalized_1, v_weight, out_v);
    let q = proj::q_proj(ctx, &normalized_1, q_weight, q_bias);
    let k = proj::k_proj(ctx, &normalized_1, k_weight);
    // Phase 6: RoPE — q, k = rotary_emb(position_ids, q, k)
    rope::rope(ctx, &q, &k, rope_table, position_ids, out_q, out_k);
}
