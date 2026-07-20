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

2. **[`cargo-furiosa-opt`](./appendix/cargo-furiosa-opt.md)**: the cargo subcommand that injects the right `--cfg backend="..."` and pre-compiles kernels for NPU.

   ```bash
   cargo +nightly-2026-05-01 install cargo-binstall
   cargo +nightly-2026-05-01 binstall cargo-furiosa-opt
   ```

3. **Furiosa SDK + physical NPU** *(only for `--backend npu`)*: the NPU backend dispatches to real hardware via the SDK's kernel driver and PE runtime (`furiosa-driver-rngd`, `furiosa-smi`, etc.; see the [SDK documentation](https://developer.furiosa.ai/latest/en/)).

   **The `emulation` and `typecheck` backends do not require the SDK.** They run host-side with no NPU dependency, so a customer who only intends to develop or evaluate kernels does not need to install the SDK at all.

## Your First Program

Use [`cargo-generate`](https://cargo-generate.github.io/cargo-generate/) to scaffold a fresh project from the `base-template` starter, which ships with the five worked examples covered in the [Quick Start](./quick-start.md) chapter:

```bash
cargo install cargo-generate
cargo generate furiosa-ai/furiosa-opt base-template
cd base-template
```

### Layout

```text
base-template/
├── Cargo.toml                               # `[package.metadata.furiosa-opt]` marks it a kernel package
├── README.md
├── rust-toolchain.toml
└── src/
    ├── lib.rs                                # `pub mod kernel;`
    ├── kernel/                               # every #[device] function lives here
    │   ├── mod.rs                            # `pub mod {constant_add,...}_kernel;`
    │   ├── constant_add_kernel.rs            # `#[device] fn constant_add_kernel(...)`
    │   ├── elementwise_mul_kernel.rs
    │   ├── dot_product_kernel.rs
    │   ├── gemv_kernel.rs
    │   └── gemm_kernel.rs
    ├── constant_add.rs                       # host binary that `launch()`es its kernel
    ├── elementwise_mul.rs
    ├── dot_product.rs
    ├── gemv.rs
    └── gemm.rs
```

Keep these layout rules intact:

- A package opts into kernel compilation by declaring `[package.metadata.furiosa-opt]` in its `Cargo.toml`.
- Host programs should live as direct `src/*.rs` files and are registered with explicit `[[bin]] path = "src/<name>.rs"` entries in `Cargo.toml`.
- Do not move host programs into `src/bin/`, `examples/`, or `tests/`; the rustc plugin scans cargo targets rooted at `src/` and skips those other locations.

All five kernel examples live under `src/kernel/` and are re-exported through `src/kernel/mod.rs` and `src/lib.rs`.
Each binary's `main()` only calls `launch(kernel, ...)`.
The value comparison against a host-side reference lives in a `#[cfg(test)] mod tests` block inside the same file.

### Run a worked example

```bash
# Host-side emulation (default; no NPU hardware required).
cargo furiosa-opt run --release --bin gemm

# Mapping/shape verification only — kernel body runs against phantom (empty) tensors.
cargo furiosa-opt --backend typecheck run --release --bin gemm

# Real NPU dispatch (requires the SDK and a physical NPU; see Installation step 3).
cargo furiosa-opt --backend npu run --release --bin gemm
```

### Verify against the reference

```bash
# Full numeric comparison on emulated values.
cargo furiosa-opt test --release --bin gemm

# Under typecheck the comparison loop trivially passes: `actual` is the
# phantom-empty Vec, so the per-element assertion has zero iterations.
cargo furiosa-opt --backend typecheck test --release --bin gemm
```

### Add a Kernel

1. Drop `src/kernel/<name>_kernel.rs` with a `#[device(...)] pub fn <name>_kernel(...)`.
2. Append `pub mod <name>_kernel;` to `src/kernel/mod.rs`.
3. Add `src/<name>.rs` as the host program that calls `launch(<name>_kernel, ...)`.
4. Register a matching `[[bin]]` entry in `Cargo.toml` with `path = "src/<name>.rs"`.
5. Run your kernel with `cargo furiosa-opt run --release --bin <name>`.

## Development Tools

The Furiosa IR Optimizer provides utilities for developing, testing, and optimizing vISA programs on TCP devices.
It complements the [Furiosa SDK's compiler](https://developer.furiosa.ai/latest/en/overview/software_stack.html#furiosa-compiler) by giving developers fine-grained control over program behavior, whether the programmer writes vISA by hand or a compiler generates it.

### Backends

A vISA program is a Rust program that uses the `furiosa-opt-std` API. `cargo furiosa-opt` selects which backend evaluates the kernel by setting `--cfg backend="..."`:

| Backend     | Default? | What runs                                   | Use when                                                                             |
|-------------|----------|---------------------------------------------|--------------------------------------------------------------------------------------|
| `typecheck` |          | Kernel body runs with phantom (empty) tensors | Catching mapping/shape errors fast (value computation skipped)                       |
| `emulation`  | yes      | Full host-side interpretation over the physical buffer | Default for development; verifies numerical correctness                              |
| `npu`       |          | Compiled EDF on hardware (or NVP simulator) | End-to-end including the hardware path                                               |

```bash
# Default: emulation backend, no NPU hardware needed.
cargo furiosa-opt run --release

# Fast mapping/shape verification (kernel body runs with phantom tensors).
cargo furiosa-opt --backend typecheck run --release

# Real NPU dispatch (requires the SDK and a physical NPU).
cargo furiosa-opt --backend npu run --release
```

`cargo check` (under any backend) only runs the type checker; it does **not** execute kernel function bodies and therefore cannot reach mapping assertions like `Collect output packet must be exactly 32 bytes`. Use `--backend typecheck run` for that.

`cargo furiosa-opt` forwards every cargo flag verbatim, so `cargo run`, `cargo test`, `cargo check`, and `cargo build` all have direct equivalents:

```bash
cargo furiosa-opt build              # cargo build with emulation backend
cargo furiosa-opt --backend npu test # cargo test on real NPU
```

See the [`cargo furiosa-opt` appendix](./appendix/cargo-furiosa-opt.md) for the complete command reference.


### Language Server

`furiosa-rust-analyzer-proxy` is a proxy for `rust-analyzer` that provides standard Rust IDE features with enhanced support for mapping expressions.
It keeps the usual `rust-analyzer` experience while simplifying verbose types like `Stride<Symbol<A>, 8>` into readable mapping expressions like `m![A / 8]`.

For installation and configuration, see the [Language Server appendix](./appendix/language-server.md).

### Schedule Viewer

The Schedule Viewer visualizes the execution timeline to help identify performance bottlenecks.
Use `furiosa-opt` to export a schedule JSON file, then open it with `furiosa-schedule-viewer`.

For installation and usage, see the [Schedule Viewer appendix](./appendix/schedule-viewer.md).

## Book Organization

Each chapter builds on the previous: mapping and moving tensors establish the data model, computing tensors covers the pipeline engines, and scheduling and kernel examples show how to compose them into real programs.

- **[Quick Start](./quick-start.md)**: How vISA programming works, introduced through worked examples covering element-wise operations and tensor contractions.
- **[Mapping Tensors](./mapping-tensors/index.md)**: How logical tensors map to physical memory: axis layout, stride, padding, and tiling.
- **[Moving Tensors](./moving-tensors/index.md)**: How data moves between memory tiers (HBM, DM) and the Tensor Unit via Fetch, Commit, and DMA engines.
- **[Computing Tensors](./computing-tensors/index.md)**: How the Tensor Unit pipeline (Switch, Collect, Contraction, Vector, Cast, Transpose) transforms data each cycle.
- **[Scheduling](./scheduling.md)**: How operations are ordered and executed concurrently across contexts.
- **[Kernel Examples](./kernel-examples/index.md)**: End-to-end examples showing how mapping, movement, computation, and scheduling combine into real kernels.

## License

This documentation and the entire `furiosa-opt` repository are licensed under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
