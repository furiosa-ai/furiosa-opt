//! Composable kernel: `transformer_0:attention`
//!
//! Pipeline: KV Cache Scatter → Q*K^T Matmul → Softmax → Score*V Matmul
//!
//! Qwen 2.5 0.5B, 4 PE, W16A16KV16 (bf16), prefill batch=1 seq_len=1024.
//!
//! ## Model Constants (Qwen 2.5 0.5B)
//!
//! - hidden_size = 896, num_q_heads = 14, num_kv_heads = 2, head_dim = 64
//! - gqa_ratio = 7 (14 Q heads / 2 KV heads)
//! - prefill seq_len = 1024, kv_cache_len = 1124 (1024 + 100 padding)

mod attn_output;
mod attn_weight;
mod kv_cache;
mod softmax;

use crate::transformer::axes::{
    C_kvcache as C, // = 1124, kv_cache_len = seq_len + 100 padding
    D,              // = 64, head_dim
    H,              // = 896, hidden_size = num_q_heads * head_dim = 14 * 64
    K,              // = 128, kv_proj_size = N * D = 2 * 64
    N,              // = 2, num_kv_heads
    S_prefill as S, // = 1024, query sequence length (prefill)
    T,              // = 1024, key/value sequence length
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];

#[device(chip = 1)]
#[expect(clippy::too_many_arguments)]
pub fn forward(
    ctx: &mut Context,
    attn_q: &HbmTensor<bf16, Chip, m![S, H]>,
    attn_k: &HbmTensor<bf16, Chip, m![T, K]>,
    attn_v: &HbmTensor<bf16, Chip, m![T, K]>,
    k_scatter_index: &HbmTensor<i32, Chip, m![1, T]>,
    v_scatter_index: &HbmTensor<i32, Chip, m![1, T]>,
    attention_mask: &HbmTensor<i32, Chip, m![1, S, T]>,
    out_attn: &mut HbmTensor<bf16, Chip, m![S, H]>,
    out_k_cache: &mut HbmTensor<bf16, Chip, m![C, 1, N, D]>,
    out_v_cache: &mut HbmTensor<bf16, Chip, m![C, 1, N, D]>,
) {
    // PyTorch Qwen2Attention.forward():
    //   q, k, v = ...  (already projected + RoPE'd by embedding/decoder kernel)
    //   attn_output = attn(q, k, v, kv_cache, attention_metadata, attention_mask)
    //
    // Decomposed here into: KV cache write → Q×K → softmax → Score×V
    //   scale = head_dim**-0.5 = 64**-0.5 = 0.125 (applied in attn_weight)
    //   GQA: 14 Q-heads / 2 KV-heads = 7 groups (G=7, softmax per group)

    // KV cache scatter — kv_cache.write(k, v) at indexed positions
    kv_cache::cache_kv(
        ctx,
        attn_k,
        attn_v,
        k_scatter_index,
        v_scatter_index,
        out_k_cache,
        out_v_cache,
    );

    // score = softmax(Q @ K^T * 0.125 + mask) — prefill causal attention
    let qk_scores = attn_weight::attn_weight(ctx, attn_q, attn_k);
    let softmax_out = softmax::softmax(ctx, &qk_scores, attention_mask);
    // attn_output = score @ V
    attn_output::attn_output(ctx, &softmax_out, attn_v, out_attn);
}
