#![allow(clippy::type_complexity)]

use furiosa_opt_examples::transformer::{attention, decoder, embedding, head};
use furiosa_opt_std::prelude::*;

struct QkvAddrs {
    q: u64,
    k: u64,
    v: u64,
    hidden: u64,
}

struct AttentionAddrs {
    score_v: u64,
    hidden: u64,
}

async fn run_embedding(ctx: &mut Context) -> QkvAddrs {
    use furiosa_opt_examples::transformer::axes::{D, H, K, P, Q, R, S_decode as S, V, W_vocab as W};

    // TODO: Fill with actual weight values (mirror PyTorch weight loading)
    let embedding_table = HostTensor::<bf16, m![W, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let input_ids = HostTensor::<i32, m![S]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_weight = HostTensor::<bf16, m![Q, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_bias = HostTensor::<bf16, m![Q]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_weight = HostTensor::<bf16, m![K, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_weight = HostTensor::<bf16, m![V, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let rope_table = HostTensor::<bf16, m![P, D / 2, R, R]>::zero()
        .to_hbm(&mut ctx.pdma)
        .await;
    let position_ids = HostTensor::<i32, m![S]>::zero().to_hbm(&mut ctx.pdma).await;

    let mut out_q = unsafe { HbmTensor::<bf16, m![1], m![S, Q]>::from_addr(0x10036000) };
    let mut out_k = unsafe { HbmTensor::<bf16, m![1], m![S, K]>::from_addr(0x10436000) };
    let mut out_v = unsafe { HbmTensor::<bf16, m![1], m![S, V]>::from_addr(0x10536000) };
    let mut out_hidden = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(0x10636000) };

    launch(
        embedding::forward,
        (
            &mut *ctx,
            &embedding_table,
            &input_ids,
            &norm_weight,
            &q_weight,
            &q_bias,
            &k_weight,
            &v_weight,
            &rope_table,
            &position_ids,
            &mut out_q,
            &mut out_k,
            &mut out_v,
            &mut out_hidden,
        ),
    )
    .await;

    QkvAddrs {
        q: out_q.address(),
        k: out_k.address(),
        v: out_v.address(),
        hidden: out_hidden.address(),
    }
}

async fn run_attention(ctx: &mut Context, qkv: &QkvAddrs) -> AttentionAddrs {
    use furiosa_opt_examples::transformer::axes::{C_kvcache as C, D, H, K, N, S_prefill as S, T};

    let attn_q = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(qkv.q) };
    let attn_k = unsafe { HbmTensor::<bf16, m![1], m![T, K]>::from_addr(qkv.k) };
    let attn_v = unsafe { HbmTensor::<bf16, m![1], m![T, K]>::from_addr(qkv.v) };

    let k_scatter_index = HostTensor::<i32, m![1, T]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_scatter_index = HostTensor::<i32, m![1, T]>::zero().to_hbm(&mut ctx.pdma).await;
    let attention_mask = HostTensor::<i32, m![1, S, T]>::zero().to_hbm(&mut ctx.pdma).await;

    let mut out_attn = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(0x288800) };
    let mut out_k_cache = unsafe { HbmTensor::<bf16, m![1], m![C, 1, N, D]>::from_addr(0x448800) };
    let mut out_v_cache = unsafe { HbmTensor::<bf16, m![1], m![C, 1, N, D]>::from_addr(0x40000) };

    launch(
        attention::forward,
        (
            &mut *ctx,
            &attn_q,
            &attn_k,
            &attn_v,
            &k_scatter_index,
            &v_scatter_index,
            &attention_mask,
            &mut out_attn,
            &mut out_k_cache,
            &mut out_v_cache,
        ),
    )
    .await;

    AttentionAddrs {
        score_v: out_attn.address(),
        hidden: qkv.hidden,
    }
}

async fn run_decoder(ctx: &mut Context, attn: &AttentionAddrs) -> QkvAddrs {
    use furiosa_opt_examples::transformer::axes::{D, H, K, M, P, Q, R, S_decode as S, V};

    let matmul_score_v = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(attn.score_v) };
    // TODO: Fill with actual input data (currently uninitialized placeholder)
    let hidden_states = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(attn.hidden) };

    let o_proj_weight = HostTensor::<bf16, m![H, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight_0 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let gate_weight = HostTensor::<bf16, m![M, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let up_weight = HostTensor::<bf16, m![M, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let down_weight = HostTensor::<bf16, m![H, M]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight_1 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_weight = HostTensor::<bf16, m![Q, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_bias = HostTensor::<bf16, m![Q]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_weight = HostTensor::<bf16, m![K, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_bias = HostTensor::<bf16, m![K]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_weight = HostTensor::<bf16, m![V, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_bias = HostTensor::<bf16, m![V]>::zero().to_hbm(&mut ctx.pdma).await;
    let rope_table = HostTensor::<bf16, m![P, D / 2, R, R]>::zero()
        .to_hbm(&mut ctx.pdma)
        .await;
    let position_ids = HostTensor::<i32, m![S]>::zero().to_hbm(&mut ctx.pdma).await;

    let mut out_q = unsafe { HbmTensor::<bf16, m![1], m![S, Q]>::from_addr(0x20036000) };
    let mut out_k = unsafe { HbmTensor::<bf16, m![1], m![S, K]>::from_addr(0x20436000) };
    let mut out_v = unsafe { HbmTensor::<bf16, m![1], m![S, V]>::from_addr(0x20536000) };
    let mut out_hidden = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(0x20636000) };

    launch(
        decoder::forward,
        (
            &mut *ctx,
            &matmul_score_v,
            &o_proj_weight,
            &hidden_states,
            &norm_weight_0,
            &gate_weight,
            &up_weight,
            &down_weight,
            &norm_weight_1,
            &q_weight,
            &q_bias,
            &k_weight,
            &k_bias,
            &v_weight,
            &v_bias,
            &rope_table,
            &position_ids,
            &mut out_q,
            &mut out_k,
            &mut out_v,
            &mut out_hidden,
        ),
    )
    .await;

    QkvAddrs {
        q: out_q.address(),
        k: out_k.address(),
        v: out_v.address(),
        hidden: out_hidden.address(),
    }
}

async fn run_head(ctx: &mut Context, attn: &AttentionAddrs) {
    use furiosa_opt_examples::transformer::axes::{H, M, S_decode as S, W_vocab as W};

    let matmul_score_v = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(attn.score_v) };
    let hidden_states = unsafe { HbmTensor::<bf16, m![1], m![S, H]>::from_addr(attn.hidden) };

    let o_proj_weight = HostTensor::<bf16, m![H, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight_0 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let gate_weight = HostTensor::<bf16, m![M, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let up_weight = HostTensor::<bf16, m![M, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let down_weight = HostTensor::<bf16, m![H, M]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight_1 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let embedding_table = HostTensor::<bf16, m![W, H]>::zero().to_hbm(&mut ctx.pdma).await;

    let mut out_logits = unsafe { HbmTensor::<bf16, m![1], m![S, W]>::from_addr(0x30000000) };

    launch(
        head::forward,
        (
            &mut *ctx,
            &matmul_score_v,
            &o_proj_weight,
            &hidden_states,
            &norm_weight_0,
            &gate_weight,
            &up_weight,
            &down_weight,
            &norm_weight_1,
            &embedding_table,
            &mut out_logits,
        ),
    )
    .await;
}

/// Qwen 2.5 0.5B: 24 decoder layers.
///
/// Mirrors `Qwen2Model.forward()`:
/// ```python
/// hidden_states = self.get_input_embeddings(input_ids)
/// for i, layer in enumerate(self.layers):       # 24 layers
///     hidden_states = layer(position_ids, hidden_states, kv_caches[i], ...)
/// hidden_states = self.norm(hidden_states)
/// logits = self.logits_processor(hidden_states)
/// ```
#[tokio::test]
#[ignore = "DmaCommandScatter lowering not yet implemented"]
async fn run_qwen() {
    const NUM_LAYERS: usize = 24;
    let mut ctx = Context::acquire();
    let qkv = run_embedding(&mut ctx).await;
    let mut attn = run_attention(&mut ctx, &qkv).await;
    for _ in 1..NUM_LAYERS {
        let qkv = run_decoder(&mut ctx, &attn).await;
        attn = run_attention(&mut ctx, &qkv).await;
    }
    run_head(&mut ctx, &attn).await;
}
