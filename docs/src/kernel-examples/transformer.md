# Case Study: Transformer

This Transformer page uses Qwen3-0.6B as a concrete tutorial case, explaining its model mental model and following one decoder token from a host-prepared embedding to padded vocabulary logits.
Each boundary is checked with a portable host oracle and a reproducible schedule artifact.
Preserve those host-prepared boundaries when adapting it.

The model is a decoder-only transformer with 28 layers.
Each layer transforms one running hidden vector for one token at a time.

```text
token id
   |
   v
[Embedding lookup]
   |
   v
+-----------------------------+
| Decoder layer × 28          |
| RMSNorm → Q/K/V projection  |
| QK-Norm → RoPE → Attention  |
| Output projection + residual|
| RMSNorm → SwiGLU MLP        |
| + residual                  |
+-----------------------------+
   |
   v
[Final RMSNorm] → [LM head] → logits
```

The model uses hidden size 1024, 16 query heads, 8 key/value heads, head dimension 128, and intermediate size 3072.
Grouped-query attention shares each key/value head across two query heads.
RoPE uses theta 1,000,000, and the embedding and output weights are tied by the model design.

| Component | Shape or count |
|---|---|
| Decoder layers | 28 |
| Vocabulary | 151,936 tokens |
| Embedding and tied LM head | `151,936 × 1,024` |
| Attention projections per layer | `6,291,456` parameters |
| SwiGLU MLP per layer | `9,437,184` parameters |
| Total model | approximately 596 million parameters |

The RMSNorm, RoPE, attention, KV-cache, SwiGLU, residual, and output-projection decisions are summarized in this case study and implemented in the local transformer example.

The example is **decode-only and single-token**.
It does not implement prefill, sampling, or device-side token lookup.
The executable source is the [transformer example](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/mod.rs).
This case study explains its decisions and points to the source and tests instead of copying the implementation.

## Runnable decoder flow

The local `furiosa-opt-examples/src/transformer/mod.rs` exposes these device launches:

- `embedding`
- `projection`
- `attention_forward_first`
- `attention_forward`
- `decoder`
- `final_layer`

This local launch map is the canonical execution surface for this case study.
The vISA and API contracts remain in the reference chapters.

## Workload semantics

The workload is one decoder step:

```text
host-prepared embedding → 28 decoder layers → final RMSNorm and LM head → logits [Wp]
```

The current example describes Qwen3-0.6B with 28 layers and hidden size 1024.
Each decoder layer has these boundaries:

1. `projection`: input RMSNorm, Q/K/V projections, Q/K RMSNorm, RoPE, and KV-cache writes.
2. `attention_forward_first` or `attention_forward`: one online-softmax attention chunk.
3. `decoder`: output projection, residual, post-attention RMSNorm, MLP, and residual.

`final_layer` applies the final RMSNorm and computes logits into the padded vocabulary shape.
The exact axis sizes and tensor types are defined in [`axes.rs`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/axes.rs).

## Decision trace

The case study follows the same order as the rest of the book:

1. **Workload semantics**: one token and a growing KV cache define the data movement.
2. **Hard constraints**: every kernel has a typed public boundary.
   The vocabulary output uses `Wp`, while the model vocabulary is `W`.
3. **Kernel boundaries**: the Rust module exposes `embedding`, `projection`, `attention_forward_first`, `attention_forward`, `decoder`, and `final_layer` as separate device entry points.
4. **Mapping and movement**: the source moves host-prepared tensors through HBM and DM and uses the representations required by each engine.
   See [Mapping Tensors](../mapping-tensors/index.md) and [Moving Tensors](../moving-tensors/index.md) for those contracts.
5. **Engine choices**: projection and LM-head contractions use the Contraction Engine.
   RMSNorm and elementwise stages use the Vector Engine.
   Transfers use the DMA and Fetch/Commit paths.
   See [Computing Tensors](../computing-tensors/index.md) for the contracts.
6. **Checks**: host-oracle tests establish values, and a dumped schedule records the static execution plan.
   See [Scheduling and Tuning](../scheduling/index.md).

The case study therefore demonstrates how choices compose.
It does not redefine any individual engine API.

## Host-prepared embedding

The `embedding` entry point is a host-prepared HBM copy.
It accepts an `HbmTensor<bf16, ..., m![H]>` and copies its view into the output HBM tensor with `to_hbm_view`.
It is not a device lookup from token IDs and must not be described as one.

The current source implements this step as [`ops::embedding`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/mod.rs).
The smoke test is [`test_embedding`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/tests/transformer_tests.rs).

## Decoder step

`projection` receives one hidden vector, the projection and RMSNorm weights, the position-dependent `cos` and `sin` vectors, and mutable K/V caches.
It writes Q to HBM and updates the cache.
The first attention chunk uses `attention_forward_first`.
Later chunks use `attention_forward` to continue the online reduction state.

`decoder` consumes the attention output and its reduction state, writes the residual stream back to HBM, and applies the output projection, residual path, RMSNorm, MLP, and final residual path.
The function signatures define the required inputs and outputs.
The implementation details are in [`furiosa-opt-examples/src/transformer/mod.rs`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/mod.rs).

The boundary test [`test_decoder_step_wires_kernel_boundaries`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/tests/transformer_tests.rs) passes host-prepared input through projection, one attention chunk, and decoder with zero weights.
Its scalar-oracle tests separately cover projection, chunked attention, and decoder arithmetic.

## Final layer and padded logits

`final_layer` takes the residual stream, applies RMSNorm, reshapes the model vocabulary weight view to the padded output representation, and writes logits to `m![Wp]`.
The source accepts a separate `lm_head_weight`.
This example makes no tied-embedding claim.

The deterministic test [`test_final_layer_matches_scalar_reference`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/tests/transformer_tests.rs) compares the valid `W` logits with a host RMSNorm-and-dot-product oracle.
Padding is part of the tensor's required shape and storage behavior.
Interpretation of padded entries belongs to the caller.

## Verify and inspect

Run the transformer test target with the repository's normal command:

```bash
cargo furiosa-opt test --test transformer_tests
```

The tests use seeded small fixtures and compare each supported boundary against host equations.
They do not require model files or a sampling loop.
Compile a selected device entry point with `--dump-schedule` to produce schedule data.
Inspect the result with the [Schedule Viewer](../tools/schedule-viewer.md) and follow the [Scheduling and Tuning](../scheduling/index.md) comparison protocol.

Keep changes disciplined: establish the oracle first, record a baseline schedule, change one mapping, movement, engine, or shape choice, and compare the same artifact.
Do not infer throughput from the oracle or from source order.
