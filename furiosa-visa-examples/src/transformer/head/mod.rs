//! Composable kernel: `transformer_-1:output_projection..output`
//!
//! Pipeline: O-Projection → Residual Add → RMS Norm → MLP (gate+up+SiLU+down)
//!           → Residual Add → RMS Norm → LM Head (logits)
//!
//! Qwen 2.5 0.5B, 4 PE, W16A16KV16 (bf16), tokenwise seq_len=128.

mod lm_head;

use crate::transformer::common::{mlp, norm, o_proj};

use crate::transformer::axes::{
    H,             // = 896, hidden_size
    M,             // = 4864, mlp_intermediate
    S_decode as S, // = 128, sequence length
    W_vocab as W,  // = 151936, vocab_size
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];

#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &mut Context,
    // Inputs (arg0..arg8)
    matmul_score_v: &HbmTensor<bf16, Chip, m![S, H]>,  // arg0
    o_proj_weight: &HbmTensor<bf16, Chip, m![H, H]>,   // arg1
    hidden_states: &HbmTensor<bf16, Chip, m![S, H]>,   // arg2
    norm_weight_0: &HbmTensor<bf16, Chip, m![H]>,      // arg3
    gate_weight: &HbmTensor<bf16, Chip, m![M, H]>,     // arg4
    up_weight: &HbmTensor<bf16, Chip, m![M, H]>,       // arg5
    down_weight: &HbmTensor<bf16, Chip, m![H, M]>,     // arg6
    norm_weight_1: &HbmTensor<bf16, Chip, m![H]>,      // arg7
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>, // arg8
    // Output (out#0, pre-allocated by caller)
    out_logits: &mut HbmTensor<bf16, Chip, m![S, W]>, // out#0
) {
    // PyTorch Qwen2ForCausalLM last layer + LM head:
    //   (finish last Qwen2Decoder: o_proj → residual → MLP → residual)
    //   hidden_states = norm(hidden_states)          # final RMSNorm
    //   logits = lm_head(hidden_states)               # [S, W=151936]
    //   (lm_head weight = embed_tokens weight when tie_word_embeddings=True)

    // Phase 1: O-projection — output = o_proj(attn_output)  [bias=False]
    let o_proj_out = o_proj::o_proj(ctx, matmul_score_v, o_proj_weight);
    // Phase 2: residual + post_attention_layernorm (fused)
    let (normalized_0, residual_0) = norm::residual_norm(ctx, hidden_states, &o_proj_out, norm_weight_0);
    // Phase 3: MLP — silu(gate_proj(x)) * up_proj(x), then down_proj
    let mlp_out = mlp::mlp(ctx, &normalized_0, gate_weight, up_weight, down_weight);
    // Phase 4: residual + final RMSNorm (Qwen2Model.norm)
    let normalized_1 = norm::final_norm(ctx, &residual_0, &mlp_out, norm_weight_1);
    // Phase 5: LM Head — lm_head(hidden_states) → [S=128, W=151936]
    //   Uses embedding_table as weight (tie_word_embeddings=True)
    lm_head::lm_head(ctx, &normalized_1, embedding_table, out_logits);
}
