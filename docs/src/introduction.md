# Introduction

FuriosaAI's Tensor Contraction Processor (TCP) is a massively parallel AI accelerator targeting inference workloads.
Unlike high-level frameworks like PyTorch and XLA, which abstract away memory layouts and hardware scheduling, TCP exposes direct programmer control without requiring the byte-level reasoning of low-level kernel APIs.

TCP's Virtual Instruction Set Architecture (Virtual ISA, or vISA) is the programming interface that exposes this control.
It lets programmers reason in tensors while directly managing memory allocation and tensor unit scheduling.
This manual introduces that interface, targeting two audiences: programmers writing vISA directly and compiler developers generating it.
Both audiences assume basic Rust familiarity.
See [the language manual](https://doc.rust-lang.org/book/) if needed.

> [!WARNING]
> **Alpha Test Build: Experimental Software**
>
> This software is an early, experimental, and incomplete build intended strictly for technical evaluation and internal testing.
>
> Before using this software for any production work, critical tasks, or for important data, you must consult with Furiosa engineers.
>
> Your feedback is vital to our development. Please provide it.

## Installation

Install three pieces:

1. **Rust toolchain (pinned)**: the Furiosa optimizer is a rustc driver, ABI-locked to a specific nightly.

   ```bash
   rustup toolchain install nightly-2026-05-01
   ```

   The same channel is pinned in [`rust-toolchain.toml`](https://github.com/furiosa-ai/furiosa-opt/blob/main/rust-toolchain.toml); cargo activates it automatically when you cd into a project that includes that file.

2. **`cargo-furiosa-opt`**: the cargo subcommand that injects the right `--cfg backend="..."` and (for NPU) pre-compiles kernels.

   ```bash
   cargo install cargo-binstall
   cargo binstall cargo-furiosa-opt
   ```

3. **Furiosa SDK + physical NPU** *(only for `--backend npu`)*: the NPU backend dispatches to real hardware via the SDK's kernel driver and PE runtime (`furiosa-driver-rngd`, `furiosa-smi`, etc.; see the [SDK documentation](https://developer.furiosa.ai/latest/en/)).

   **The `simulation` and `typecheck` backends do not require the SDK.** They run host-side with no NPU dependency, so a customer who only intends to develop or evaluate kernels does not need to install the SDK at all.

   There is **no on-host NPU simulator in the public SDK distribution today**. Without physical NPU access, use `--backend simulation` (host-side interpretation) or `--backend typecheck` (mapping/shape verification only).

## Your First Program

Use [`cargo-generate`](https://cargo-generate.github.io/cargo-generate/) to scaffold a fresh project from the `base-template` starter, which ships with the five worked examples covered in the next chapter:

```bash
cargo install cargo-generate
cargo generate furiosa-ai/furiosa-opt base-template
cd base-template
```

### Layout

```text
base-template/
├── Cargo.toml
├── rust-toolchain.toml
└── src/
    ├── furiosa-opt.tag                       # marker the rustc plugin scans for; must sit at src/
    ├── lib.rs                                # `pub mod kernel;`
    ├── kernel/                               # every #[device] function lives here
    │   ├── mod.rs                            # `pub mod {constant_add,...}_kernel;`
    │   ├── constant_add_kernel.rs            # `#[device] fn constant_add_kernel(...)`
    │   ├── elementwise_mul_kernel.rs
    │   ├── dot_product_kernel.rs
    │   ├── gemv_kernel.rs
    │   └── gemm_kernel.rs
    ├── constant_add.rs                       # `#[tokio::main]` host program; `launch(constant_add_kernel, ...)`
    ├── elementwise_mul.rs
    ├── dot_product.rs
    ├── gemv.rs
    └── gemm.rs
```

The five `#[device]` kernels all live under `src/kernel/` and are re-exported by `src/lib.rs`. Each `src/*.rs` (next to `lib.rs`) is registered as its own `[[bin]]` in `Cargo.toml`, with `path = "src/<name>.rs"` placing the binary's source directly under `src/` — the rustc plugin scans cargo targets rooted at `src/` and silently skips anything under `src/bin/`, `examples/`, or `tests/`, so the explicit `[[bin]]` paths in `Cargo.toml` are load-bearing. Each binary's `main()` only does `launch(kernel, ...)`; the value comparison against a host-side reference lives in a `#[cfg(test)] mod tests` block inside the same file.

### Run a worked example

```bash
# Host-side simulation (default; no NPU hardware required).
cargo furiosa-opt run --release --bin gemm

# Mapping/shape verification only — kernel body runs against phantom (empty) tensors.
cargo furiosa-opt --backend typecheck run --release --bin gemm

# Real NPU dispatch (requires the SDK and a physical NPU; see Installation step 3).
cargo furiosa-opt --backend npu run --release --bin gemm
```

### Verify against the reference

```bash
# Full numeric comparison on simulated values.
cargo furiosa-opt test --release --bin gemm

# Under typecheck the comparison loop trivially passes: `actual` is the
# phantom-empty Vec, so the per-element assertion has zero iterations.
cargo furiosa-opt --backend typecheck test --release --bin gemm
```

Read the [Quick Start](./quick-start.md) chapter alongside the source. Add your own kernel by dropping a new file into `src/kernel/`, appending a `pub mod ...;` line to `src/kernel/mod.rs`, writing a host program under `src/`, and declaring a matching `[[bin]] path = "src/<name>.rs"` in `Cargo.toml`.

## Development Tools

The Furiosa IR Optimizer provides utilities for developing, testing, and optimizing vISA programs on TCP devices.
It complements the [Furiosa SDK's compiler](https://developer.furiosa.ai/latest/en/overview/software_stack.html#furiosa-compiler) by giving developers fine-grained control over program behavior, whether the programmer writes vISA by hand or a compiler generates it.

### Backends

A vISA program is a Rust program that uses the `furiosa-opt-std` API. `cargo furiosa-opt` selects which backend evaluates the kernel by setting `--cfg backend="..."`:

| Backend     | Default? | What runs                                   | Use when                                                                             |
|-------------|----------|---------------------------------------------|--------------------------------------------------------------------------------------|
| `typecheck` |          | Kernel body runs with phantom (empty) tensors | Catching mapping/shape errors fast (value computation skipped)                       |
| `simulation`  | yes      | Full host-side interpretation               | Default for development; verifies numerical correctness                              |
| `npu`       |          | Compiled EDF on hardware (or NVP simulator) | End-to-end including the hardware path                                               |

```bash
# Default: simulation backend, no NPU hardware needed.
cargo furiosa-opt run --release

# Fast mapping/shape verification (kernel body runs with phantom tensors).
cargo furiosa-opt --backend typecheck run --release

# Real NPU dispatch (requires the SDK and a physical NPU).
cargo furiosa-opt --backend npu run --release
```

`cargo check` (under any backend) only runs the type checker; it does **not** execute kernel function bodies and therefore cannot reach mapping assertions like `Collect output packet must be exactly 32 bytes`. Use `--backend typecheck run` for that.

`cargo furiosa-opt` forwards every cargo flag verbatim, so `cargo run`, `cargo test`, `cargo check`, and `cargo build` all have direct equivalents:

```bash
cargo furiosa-opt build              # cargo build with simulation backend
cargo furiosa-opt --backend npu test # cargo test on real NPU
```


### Language Server

`furiosa-rust-analyzer-proxy` is a proxy for `rust-analyzer` that provides standard Rust IDE features with enhanced support for mapping expressions.
It keeps the usual `rust-analyzer` experience while simplifying verbose types like `Stride<Symbol<A>, 8>` into readable mapping expressions like `m![A / 8]`.

For installation and configuration, see the [Language Server appendix](./appendix/language-server.md).


## Book Organization

Each chapter builds on the previous: mapping and moving tensors establish the data model, computing tensors covers the pipeline engines, and scheduling and kernel examples show how to compose them into real programs.

- **[Quick Start](./quick-start.md)**: How vISA programming works, introduced through worked examples covering element-wise operations and tensor contractions.
- **[Mapping Tensors](./mapping-tensors/index.md)**: How logical tensors map to physical memory: axis layout, stride, padding, and tiling.
- **[Moving Tensors](./moving-tensors/index.md)**: How data moves between memory tiers (HBM, DM) and the Tensor Unit via Fetch, Commit, and DMA engines.
- **[Computing Tensors](./computing-tensors/index.md)**: How the Tensor Unit pipeline (Switch, Collect, Contraction, Vector, Cast, Transpose) transforms data each cycle.
- **[Scheduler](./scheduler.md)**: How to control the order and concurrency of operations across contexts.
- **[Kernel Examples](./kernel-examples/index.md)**: End-to-end examples showing how mapping, movement, computation, and scheduling combine into real kernels.

## License

This documentation and the entire `furiosa-opt` repository are licensed under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
