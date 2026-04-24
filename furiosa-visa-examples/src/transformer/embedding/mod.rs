//! Composable kernel: `embedding..transformer_0:qkv_projection`
//!
//! Pipeline: Embedding lookup -> RMS Norm -> Q/K/V Projection -> RoPE
//!
//! Qwen 2.5 0.5B, 4 PE, W16A16KV16 (bf16), tokenwise seq_len=128.

#[allow(clippy::module_inception)]
mod embedding;
pub(crate) mod proj;
mod rms_norm;
use crate::transformer::common::rope;

use crate::transformer::axes::{
    D,             // = 64, head_dim
    H,             // = 896, hidden_size
    K,             // = 128, k_proj
    P,             // = 32768, max_position
    Q,             // = 896, q_proj
    R,             // = 2, rope_rot
    S_decode as S, // = 128, sequence length
    V,             // = 128, v_proj
    W_vocab as W,  // = 151936, vocab_size
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];

#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &mut Context,
    // Inputs (arg0..arg8)
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,      // arg0
    input_ids: &HbmTensor<i32, Chip, m![S]>,                // arg1
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,             // arg2
    q_weight: &HbmTensor<bf16, Chip, m![Q, H]>,             // arg3
    q_bias: &HbmTensor<bf16, Chip, m![Q]>,                  // arg4
    k_weight: &HbmTensor<bf16, Chip, m![K, H]>,             // arg5
    v_weight: &HbmTensor<bf16, Chip, m![V, H]>,             // arg6
    rope_table: &HbmTensor<bf16, Chip, m![P, D / 2, R, R]>, // arg7
    position_ids: &HbmTensor<i32, Chip, m![S]>,             // arg8
    // Outputs (out#0..out#3, pre-allocated by caller)
    out_q: &mut HbmTensor<bf16, Chip, m![S, Q]>,      // out#0
    out_k: &mut HbmTensor<bf16, Chip, m![S, K]>,      // out#1
    out_v: &mut HbmTensor<bf16, Chip, m![S, V]>,      // out#2
    out_hidden: &mut HbmTensor<bf16, Chip, m![S, H]>, // out#3
) {
    // PyTorch Qwen2Model first layer:
    //   hidden_states = embed_tokens(input_ids)
    //   hidden_states = input_layernorm(hidden_states)
    //   q, k, v = q/k/v_proj(hidden_states)
    //   q, k = rotary_emb(position_ids, q, k)

    // Embedding lookup — embed_tokens(input_ids) → [S=128, H=896]
    let embeddings_dm = embedding::embed(ctx, input_ids, embedding_table);
    // Write embedding output to caller-provided hidden state buffer.
    embeddings_dm.view().to_hbm_view(&mut ctx.tdma, out_hidden.view_mut());
    // Apply input RMSNorm.
    let normalized = rms_norm::rms_norm(ctx, out_hidden, norm_weight);
    // QKV projections
    //   PyTorch: q = q_proj(x) [bias=True], k = k_proj(x) [bias=True], v = v_proj(x) [bias=True]
    //   K/V bias is omitted in this kernel path
    proj::v_proj(ctx, &normalized, v_weight, out_v);
    let q = proj::q_proj(ctx, &normalized, q_weight, q_bias);
    let k = proj::k_proj(ctx, &normalized, k_weight);
    // q, k = rotary_emb(position_ids, q, k)
    rope::rope(ctx, &q, &k, rope_table, position_ids, out_q, out_k);
}
