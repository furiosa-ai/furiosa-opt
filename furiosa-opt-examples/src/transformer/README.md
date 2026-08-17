# Qwen3 example

Qwen3-0.6B (28 layers, hidden size 1024) decode-only inference, with two aligned implementations:

- `transformer.py` — PyTorch reference and CLI; `verify` diffs every kernel against the reference, `run` generates from a prompt.
- `kernel/` + `mod.rs` — the VISA kernels, and the six `#[device]` entry points a token step launches (`embedding`, `projection`, `attention_forward_first`, `attention_forward`, `decoder`, `final_layer`). `axes.rs` holds the extents their type signatures speak.

## Model

```
embedding → { projection → attention → decoder } × 28 layers → final_layer → logits [Wp]
```

Every kernel processes exactly one token, so a prompt token is fed the same way a generated token is — there is no batched prefill stage. Each layer runs `projection` (input rmsnorm + q/k/v proj + q/k norm + RoPE) → `attention` (online softmax over KV-cache chunks `0..=chunk`) → `decoder` (o_proj + residual + MLP + residual).

## Run

Both Python commands compile the kernels through `furiosa-torch`'s `compile_from_visa`, which the
released wheel does not carry yet; `requirements.txt` pins the revision that does.

### Python, kernel correctness against the reference (coming soon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python transformer.py verify
```

Drives each kernel with random weights and diffs it against the PyTorch reference; no model weights required.

### Python, end-to-end generation via furiosa-torch (coming soon)

```bash
python transformer.py run "The capital of France is" --model /path/to/Qwen3-0.6B
```

`--model` is the directory holding `model.safetensors` and `tokenizer.json`. `--max-new-tokens` defaults to 300; decoding is greedy, so a run needs no seed to be reproducible. Timing (compile, load, warm-up, time-to-first-token, prefill/decode throughput) goes to stderr.

### Rust (kernel smoke tests)

```bash
cargo furiosa-opt test --test transformer_tests -- --nocapture
```
