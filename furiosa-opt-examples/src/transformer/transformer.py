#!/usr/bin/env python3
"""Qwen3-0.6B single-token NPU inference: the cpu reference model, and the CLI.

Axis vocabulary = the kernels' type-level contract (mirrors axes.rs)
    H  = 1024    hidden size          N = 8     kv heads
    Q  = 2048    fused query width    G = 2     query heads per kv head
    W  = 151936  vocab                D = 128   head dim
    Wp = 155648  vocab padded to whole 8192-row lm_head tiles
    E  = 32768   max position         L = 3072  mlp intermediate width
                                      T = 512   kv-cache chunk length

    embedding -> { projection -> attention -> decoder } x 28 -> final_layer -> logits [Wp]

Every kernel processes exactly one token, so a prompt token is fed the same way a
generated token is; there is no batched prefill stage.

Usage:
    python transformer.py verify
    python transformer.py run "<prompt>" --model /path/to/Qwen3-0.6B
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import furiosa.torch
from furiosa.torch._module.visa import compile_from_visa
from safetensors import safe_open
from tokenizers import Tokenizer

H, Q, W, Wp, E = 1024, 2048, 151936, 19 * 8192, 32768
N, G, D, L, T = 8, 2, 128, 3072, 512
LAYERS = 28
ROPE_THETA = 1_000_000.0
SEED = 42

# `dma_scatter` indices are byte offsets, so a `[.., N, D]` bf16 row is this wide.
ROW_STRIDE_ND = N * D * 2

# `crate_root` for `compile_from_visa`: the directory holding Cargo.toml.
CRATE_PATH = Path(__file__).parents[2]
DEVICE = 'furiosa'

torch.manual_seed(SEED)
furiosa.torch.set_fusion(8)


@dataclass
class Qwen3Config:
    """HF config driving the cpu reference; every field mirrors an axis above."""

    vocab_size: int = W
    hidden_size: int = H
    intermediate_size: int = L
    num_hidden_layers: int = LAYERS
    num_attention_heads: int = N * G
    num_key_value_heads: int = N
    head_dim: int = D
    max_position_embeddings: int = E
    rope_theta: float = ROPE_THETA
    rms_norm_eps: float = 6.25e-8
    attention_bias: bool = False


# Reference model
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class Attention(nn.Module):
    @staticmethod
    def rope_tables(cfg, dtype):
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
        )
        freqs = torch.outer(torch.arange(cfg.max_position_embeddings).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.q_proj = nn.Linear(
            cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.attention_bias
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size, self.n_kv * self.head_dim, bias=cfg.attention_bias
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size, self.n_kv * self.head_dim, bias=cfg.attention_bias
        )

    def projection(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        def rotate_half(t):
            half = t.shape[-1] // 2
            return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k, v


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Decoder(nn.Module):
    """Second half of a transformer layer: o_proj + residual + post-norm + MLP."""

    def __init__(self, cfg):
        super().__init__()
        self.mlp = MLP(cfg)
        self.o_proj = nn.Linear(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, rx):
        x = rx + self.o_proj(x)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# Kernels
def npu(kernel_name):
    return compile_from_visa(CRATE_PATH, f'transformer::{kernel_name}').to(DEVICE)


# Verify
def sample(*shape):
    return torch.rand(*shape, dtype=torch.bfloat16, requires_grad=False)


def report(name, expected, actual):
    # Δ = expected - actual (the per-element error of the NPU kernel's output
    # against the PyTorch reference), always reported as |Δ|.
    e = expected.detach().cpu().float().flatten()
    a = actual.detach().cpu().float().flatten()
    diff = (e - a).abs()

    # Relative tolerance, floored at bf16 round-off so that near-zero expected
    # values do not get an unreasonably tight bound.
    bf16_eps = 2**-8  # ~half a bf16 ULP at magnitude 1
    tol = torch.maximum(0.01 * e.abs(), torch.full_like(e, bf16_eps))
    within_tol = diff <= tol
    pass_rate = within_tol.float().mean().item() * 100
    ok = pass_rate >= 99.5

    verdict = 'PASS' if ok else 'FAIL'
    print(f'{name:<22}{diff.max().item():>10.4f}{pass_rate:>10.2f}%{verdict:>8}')
    return ok


def verify_projection(cfg, pos=5):
    """ops::projection: input-norm + q/k/v proj + q/k norm + RoPE, scattering
    K/V into row `pos` of a full `[T, N, D]` KV cache."""
    cos, sin = Attention.rope_tables(cfg, torch.bfloat16)
    cos_p, sin_p = cos[pos : pos + 1], sin[pos : pos + 1]  # single position (1, D)

    attn = Attention(cfg).eval().bfloat16()
    rms = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).eval().bfloat16()
    kernel = npu('ops::projection')

    x = sample(1, 1, cfg.hidden_size)  # single token (B, T=1, H)
    with torch.no_grad():
        q, k, v = attn.projection(rms(x), cos_p, sin_p)
    q_cpu = q.transpose(1, 2).reshape(cfg.num_attention_heads, cfg.head_dim)  # (16, D)
    k_cpu = k.transpose(1, 2).reshape(cfg.num_key_value_heads, cfg.head_dim)  # (8, D)
    v_cpu = v.transpose(1, 2).reshape(cfg.num_key_value_heads, cfg.head_dim)  # (8, D)

    # The rope kernel does `rotate_half` as a plain half-swap (no negation) and folds
    # the sign into `sin`, so negate sin's first half here to match.
    sin_p = torch.cat((-sin_p[..., : D // 2], sin_p[..., D // 2 :]), dim=-1)

    q_npu = sample(N, G, D).to(DEVICE)  # q_out    m![N, G, D]
    k_npu = sample(T, N, D).to(DEVICE)  # k_cache  m![T, N, D]
    v_npu = sample(T, N, D).to(DEVICE)  # v_cache  m![T, N, D]

    # dma_scatter indices are byte offsets along the scattered axis: row `pos` of
    # a `[.., N, D]` bf16 table is `pos * N * D * 2` bytes in.
    index_nd = torch.tensor([pos * N * D * 2], dtype=torch.int32).to(
        DEVICE
    )  # indexnd m![1], into [T, N, D]

    kernel(
        x[0, 0].to(DEVICE),  # x                m![H]
        attn.q_proj.weight.to(DEVICE),
        attn.k_proj.weight.to(DEVICE),
        attn.v_proj.weight.to(DEVICE),
        rms.weight.to(DEVICE),  # input_rms_weight
        attn.q_norm.weight.to(DEVICE),
        attn.k_norm.weight.to(DEVICE),
        index_nd,  # indexnd
        cos_p[0].to(DEVICE),  # cos              m![D]
        sin_p[0].to(DEVICE),  # sin              m![D]
        k_npu,
        v_npu,
        q_npu,  # outputs (k_npu/v_npu are the full cache)
    )

    ok = True
    ok &= report(
        'projection q', q_cpu, q_npu.cpu().view(cfg.num_attention_heads, cfg.head_dim)
    )
    ok &= report('projection k', k_cpu, k_npu.cpu()[pos])
    ok &= report('projection v', v_cpu, v_npu.cpu()[pos])
    return ok


def verify_attention(cfg, valid=6):
    """ops::attention_forward_first: single-query attention over one T-block. Exercises the qk and weighted-sum axis gathers.

    forward_first writes the UNNORMALIZED output + running softmax sum; the
    softmax normalization is folded into the decoder kernel, so we divide the
    output by `sum` before comparing against reference softmax attention."""

    q = sample(N * G, D)  # (16, D)
    k = sample(T, N, D)  # kv cache  (T, N, D)
    v = sample(T, N, D)
    mask = torch.zeros(T, dtype=torch.float32)
    mask[:valid] = 1.0  # attend the first `valid` keys (1 = attend, 0 = mask)

    with torch.no_grad():
        qf = q.float().view(N, G, D)
        scores = torch.einsum('ngd,tnd->ngt', qf, k.float()) / (D**0.5)  # (N, G, T)
        scores = scores.masked_fill(mask.view(1, 1, T) == 0, float('-inf'))
        w = torch.softmax(scores, dim=-1)
        out_cpu = torch.einsum('ngt,tnd->ngd', w, v.float())  # (N, G, D)

    max_npu = sample(N, G).float().to(DEVICE)
    sum_npu = sample(N, G).float().to(DEVICE)
    out_npu = sample(N, G, D).to(DEVICE)

    kernel = npu('ops::attention_forward_first')
    kernel(
        q.view(N, G, D).to(DEVICE),  # q     m![N, G, D]
        k.to(DEVICE),  # k     m![T, N, D]
        v.to(DEVICE),  # v     m![T, N, D]
        mask.to(DEVICE),  # mask  m![T]
        max_npu,
        sum_npu,
        out_npu,  # outputs
    )

    out_norm = out_npu.cpu().float() / sum_npu.cpu().float().unsqueeze(-1)  # normalize
    return report('attention out', out_cpu, out_norm.view(N, G, D))


def verify_decoder(cfg):
    """ops::decoder: attention-norm + o_proj + residual + post-norm + MLP.

    `sum = ones` makes the fused attention-norm a no-op, so this reduces to the
    reference Decoder(x, rx). Exercises the MLP down-projection gather."""

    layer = Decoder(cfg).eval().bfloat16()
    kernel = npu('ops::decoder')

    x = sample(Q)  # attention output (already "normalized")  m![Q]
    rx = sample(cfg.hidden_size)  # residual stream                         m![H]
    with torch.no_grad():
        expected = layer(x.view(1, 1, Q), rx.view(1, 1, cfg.hidden_size))[0, 0]

    sum_npu = torch.ones(N, G).to(DEVICE)  # no-op norm
    x_npu = x.view(N, G, D).to(DEVICE)  # x       m![N, G, D]
    rx_npu = rx.clone().to(DEVICE)  # rx_hbm  m![H] (in/out)

    kernel(
        x_npu,
        sum_npu,
        rx_npu,
        layer.o_proj.weight.to(DEVICE),
        layer.post_attention_layernorm.weight.to(DEVICE),
        layer.mlp.up_proj.weight.to(DEVICE),
        layer.mlp.gate_proj.weight.to(DEVICE),
        layer.mlp.down_proj.weight.to(DEVICE),
    )

    return report('decoder', expected, rx_npu.cpu())


def verify_final_layer(cfg):
    """ops::final_layer: final rmsnorm + lm_head projection.

    Only the real vocab logits are checked: the kernel has no masking input to
    guarantee blanked-out values for the `Wp - W` padding rows, so their
    contents aren't part of its contract."""

    rms = RMSNorm(H, cfg.rms_norm_eps).eval().bfloat16()
    lm_head = nn.Linear(H, W, bias=False).eval().bfloat16()
    kernel = npu('ops::final_layer')

    x = sample(H)
    with torch.no_grad():
        expected = lm_head(rms(x))  # (W,)

    out = sample(Wp).to(DEVICE)
    kernel(
        x.to(DEVICE),
        rms.weight.to(DEVICE),
        lm_head.weight.to(DEVICE),
        out,
    )

    return report('final_layer (vocab)', expected, out.cpu()[:W])


def verify():
    """Diff every kernel against the cpu reference. Random weights are enough: we are
    checking that a kernel computes the right function, not the real model."""
    cfg = Qwen3Config()
    print(f'{"kernel":<22}{"max|Δ|":>10}{"within tol":>11}')
    for case in (
        verify_projection,
        verify_attention,
        verify_decoder,
        verify_final_layer,
    ):
        case(cfg)


# Generation
def load_model(model):
    """Loads and uploads every weight, one projection+decoder pair per layer:
    `layers[L]` holds layer `L`'s projection weights and layer `L`'s decoder
    weights, launched as two separate kernels."""

    def load_projection(tensors, layer):
        """Weights consumed by a projection kernel (input rmsnorm + q/k/v proj + q/k norm)."""
        prefix = f'model.layers.{layer}'
        return {
            'input_layernorm': tensors.get_tensor(
                f'{prefix}.input_layernorm.weight'
            ).to(DEVICE),
            'q_proj_w': tensors.get_tensor(f'{prefix}.self_attn.q_proj.weight').to(
                DEVICE
            ),
            'k_proj_w': tensors.get_tensor(f'{prefix}.self_attn.k_proj.weight').to(
                DEVICE
            ),
            'v_proj_w': tensors.get_tensor(f'{prefix}.self_attn.v_proj.weight').to(
                DEVICE
            ),
            'q_proj_n': tensors.get_tensor(f'{prefix}.self_attn.q_norm.weight').to(
                DEVICE
            ),
            'k_proj_n': tensors.get_tensor(f'{prefix}.self_attn.k_norm.weight').to(
                DEVICE
            ),
        }

    def load_decoder(tensors, layer):
        """Weights consumed by a decoder kernel (output proj + post rmsnorm + MLP)."""
        prefix = f'model.layers.{layer}'
        return {
            'post_attention_layernorm': tensors.get_tensor(
                f'{prefix}.post_attention_layernorm.weight'
            ).to(DEVICE),
            'o_proj_w': tensors.get_tensor(f'{prefix}.self_attn.o_proj.weight').to(
                DEVICE
            ),
            'mlp_up': tensors.get_tensor(f'{prefix}.mlp.up_proj.weight').to(DEVICE),
            'mlp_gate': tensors.get_tensor(f'{prefix}.mlp.gate_proj.weight').to(DEVICE),
            'mlp_down': tensors.get_tensor(f'{prefix}.mlp.down_proj.weight').to(DEVICE),
        }

    with safe_open(f'{model}/model.safetensors', framework='pt') as tensors:
        embedding_table = tensors.get_tensor('model.embed_tokens.weight').to(DEVICE)
        final_norm = tensors.get_tensor('model.norm.weight').to(DEVICE)
        lm_head_w = tensors.get_tensor('lm_head.weight').to(DEVICE)

        layers = [
            {
                'projection': load_projection(tensors, layer),
                'decoder': load_decoder(tensors, layer),
            }
            for layer in range(LAYERS)
        ]

    return {
        'embedding_table': embedding_table,
        'final_norm': final_norm,
        'lm_head_w': lm_head_w,
        'layers': layers,
    }


class Pipeline:
    """Everything a token step runs on: the compiled kernels, the uploaded weights, the
    KV cache and RoPE tables it fills in as it goes, and the activation scratch."""

    def __init__(self, kernels, model):
        self.kernels = kernels
        self.model = model
        self.rope_cache = {}  # chunk -> (cos [T,D], sin [T,D])
        self.kv_cache = {}  # (chunk, layer) -> (k [T,N,D], v [T,N,D])

        # `dma_scatter` takes byte offsets, one per KV-cache row.
        self.kv_offsets = torch.from_numpy(
            np.arange(T, dtype=np.int32) * ROW_STRIDE_ND
        ).to(DEVICE)

        # The current chunk is only written up to the query row, so it needs a causal
        # mask; an earlier chunk is always full, so every one of its T keys is attended.
        self.causal = torch.ones(T, T, dtype=torch.float32).tril(diagonal=0).to(DEVICE)
        self.full = torch.ones(T, dtype=torch.float32).to(DEVICE)

        self.x = torch.zeros(H, dtype=torch.bfloat16).to(DEVICE)
        self.q = torch.zeros(N, G, D, dtype=torch.bfloat16).to(DEVICE)
        self.attn_max = torch.zeros(N, G, dtype=torch.float32).to(DEVICE)
        self.attn_sum = torch.zeros(N, G, dtype=torch.float32).to(DEVICE)
        self.attn_out = torch.zeros(N, G, D, dtype=torch.bfloat16).to(DEVICE)
        self.logits = torch.zeros(Wp, dtype=torch.bfloat16).to(DEVICE)

    def mask(self, chunk, kv_chunk, row):
        return self.causal[row] if chunk == kv_chunk else self.full

    def rope(self, chunk):
        """cos/sin for `chunk`'s absolute positions `chunk*T .. chunk*T+T-1`."""
        if chunk not in self.rope_cache:
            half = D // 2
            inv_freq = 1.0 / (
                ROPE_THETA ** (np.arange(0, D, 2, dtype=np.float32) / D)
            )  # (half,)
            positions = (chunk * T + np.arange(T)).astype(np.float32)  # (T,)
            freqs = np.outer(positions, inv_freq)  # (T, half)

            cos = np.empty((T, D), dtype=np.float32)
            sin = np.empty((T, D), dtype=np.float32)
            cos[:, :half] = np.cos(freqs)
            cos[:, half:] = np.cos(freqs)
            sin[:, :half] = -np.sin(freqs)
            sin[:, half:] = np.sin(freqs)

            cos_t = torch.from_numpy(cos).to(torch.bfloat16).to(DEVICE)
            sin_t = torch.from_numpy(sin).to(torch.bfloat16).to(DEVICE)
            self.rope_cache[chunk] = (cos_t, sin_t)
        return self.rope_cache[chunk]

    def kv(self, chunk, layer):
        key = (chunk, layer)
        if key not in self.kv_cache:
            k = torch.zeros(T, N, D, dtype=torch.bfloat16).to(DEVICE)
            v = torch.zeros(T, N, D, dtype=torch.bfloat16).to(DEVICE)
            self.kv_cache[key] = (k, v)
        return self.kv_cache[key]

    def warm(self, chunks):
        """Forces the lazy KV-cache/RoPE allocation for every chunk index this run will
        ever touch, so no kernel launch pays a `.to(DEVICE)` mid-generation."""
        for chunk in chunks:
            self.rope(chunk)
            for layer in range(LAYERS):
                self.kv(chunk, layer)
        self.sample_token()

    def sample_token(self):
        return int(self.logits[:W].argmax().cpu())

    def step(self, token, pos, need_logits):
        """Runs a single token (a real prompt token or a previously-sampled token)
        through the model at absolute (0-indexed) position `pos`, updating the KV
        cache."""
        chunk = pos // T
        row = pos % T

        # `chunk` is fixed within a token step, so the RoPE table and kv-offset row are
        # hoisted here rather than recomputed by each of the 28 `projection` calls.
        cos_full, sin_full = self.rope(chunk)
        cos_row, sin_row = cos_full[row], sin_full[row]
        kv_offset_row = self.kv_offsets[row : row + 1]

        # Device-indexed rows everywhere except the KV-cache scatter, which still
        # goes through the real `dma_scatter`-based kernel with a sliced offset.
        self.kernels['ops::embedding'](self.model['embedding_table'][token], self.x)

        def projection(proj, layer):
            k_cache, v_cache = self.kv(chunk, layer)
            self.kernels['ops::projection'](
                self.x,
                proj['q_proj_w'],
                proj['k_proj_w'],
                proj['v_proj_w'],
                proj['input_layernorm'],
                proj['q_proj_n'],
                proj['k_proj_n'],
                kv_offset_row,
                cos_row,
                sin_row,
                k_cache,
                v_cache,
                self.q,
            )

        def attention(layer):
            for kv_chunk in range(chunk + 1):
                mask = self.mask(chunk, kv_chunk, row)
                k_cache, v_cache = self.kv(kv_chunk, layer)
                args = (
                    self.q,
                    k_cache,
                    v_cache,
                    mask,
                    self.attn_max,
                    self.attn_sum,
                    self.attn_out,
                )
                if kv_chunk == 0:
                    self.kernels['ops::attention_forward_first'](*args)
                else:
                    self.kernels['ops::attention_forward'](*args)

        def decoder(layer_weights):
            self.kernels['ops::decoder'](
                self.attn_out,
                self.attn_sum,
                self.x,
                layer_weights['o_proj_w'],
                layer_weights['post_attention_layernorm'],
                layer_weights['mlp_up'],
                layer_weights['mlp_gate'],
                layer_weights['mlp_down'],
            )

        for layer, layer_weights in enumerate(self.model['layers']):
            projection(layer_weights['projection'], layer)
            attention(layer)
            decoder(layer_weights['decoder'])

        if need_logits:
            self.kernels['ops::final_layer'](
                self.x, self.model['final_norm'], self.model['lm_head_w'], self.logits
            )


def generate(args):
    t0 = time.time()
    names = [
        'embedding',
        'projection',
        'attention_forward_first',
        'attention_forward',
        'decoder',
        'final_layer',
    ]
    kernels = {f'ops::{n}': npu(f'ops::{n}') for n in names}
    compile_time = time.time() - t0

    t0 = time.time()
    model = load_model(args.model)
    load_time = time.time() - t0

    tokenizer = Tokenizer.from_file(f'{args.model}/tokenizer.json')
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    num_tokens = len(ids)

    pipe = Pipeline(kernels, model)

    t0 = time.time()
    max_chunk = (num_tokens + args.max_new_tokens - 1) // T
    pipe.warm(range(max_chunk + 1))
    warmup_time = time.time() - t0

    print(args.prompt, end='', flush=True)

    # Every prompt token is run through the model one at a time, the same way
    # a generated token is -- there is no batched prefill stage.
    token = None
    prefill_time = None
    run_start = time.time()
    with torch.no_grad():
        for pos in range(num_tokens + args.max_new_tokens - 1):
            cur_token = ids[pos] if pos < num_tokens else token
            need_logits = pos >= num_tokens - 1
            pipe.step(cur_token, pos, need_logits)
            if need_logits:
                token = pipe.sample_token()
                if prefill_time is None:
                    prefill_time = time.time() - run_start
                text = tokenizer.decode([token], skip_special_tokens=False)
                print(text, end='', flush=True)
    run_time = time.time() - run_start
    print()

    decode_time = run_time - prefill_time
    prefill_tps = num_tokens / prefill_time
    decode_tps = (args.max_new_tokens - 1) / decode_time

    print(f'kernel compile time: {compile_time:.2f}s', file=sys.stderr)
    print(f'model load time:     {load_time:.2f}s', file=sys.stderr)
    print(f'workspace warm-up:   {warmup_time:.2f}s', file=sys.stderr)
    print(f'time to first token: {prefill_time * 1000:.2f}ms', file=sys.stderr)
    print(
        f'prefill:             {prefill_tps:.1f} tok/s ({num_tokens} tokens)',
        file=sys.stderr,
    )
    print(
        f'decode:              {decode_tps:.1f} tok/s ({args.max_new_tokens - 1} tokens)',
        file=sys.stderr,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', nargs='?', default='verify', choices=['verify', 'run'])
    p.add_argument('prompt', nargs='?', help='prompt to complete (`run` only)')
    p.add_argument(
        '--model', help='directory holding model.safetensors and tokenizer.json'
    )
    p.add_argument('--max-new-tokens', type=int, default=300, help='tokens to generate')
    args = p.parse_args()
    if args.cmd != 'run':
        verify()
    elif args.prompt and args.model:
        generate(args)
    else:
        p.error('run needs a prompt and --model')
