# MNIST example

A minimal fully-connected MNIST classifier (784 → 256 ReLU → 10) with two
aligned implementations:

- `mnist.py` — PyTorch reference; trains the model and exports weights to
  `../data/mnist/mnist.safetensors`.
- `mod.rs` — VISA kernel (`forward`) run by `../../tests/mnist_tests.rs`,
  which loads the exported weights.

The Rust test requires `mnist.safetensors` to already exist; run
`python mnist.py` first to produce it.

## Run

### Python (training, export, reference inference)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python mnist.py         # train + export + infer (CPU)
python mnist.py infer   # infer only (CPU)
```

### Rust (end-to-end kernel test)

```bash
cargo furiosa-opt test --test mnist_tests -- --nocapture
```
