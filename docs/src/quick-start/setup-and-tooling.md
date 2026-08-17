# Setup and Tooling

## Installation and First Program

This chapter owns the installation and first-project command path.
Run it before reading the kernel patterns.

### Host Requirements

The host must be `x86_64-unknown-linux-gnu` on Ubuntu 22.04 or newer, since the published binaries need `GLIBC_2.34` and 20.04 ships 2.31.
On macOS or arm64, run everything inside a `linux/amd64` container.

```bash
sudo apt install build-essential libclang-dev   # every build
sudo apt install gcc-aarch64-linux-gnu          # only for the NPU build (cargo furiosa-opt)
```

Install these before the toolchain below, which does not cover them: rustc links host binaries with the system `cc`, and `furiosa-opt-std`'s build script loads `libclang.so`.

### Install and Run the Starter

The Furiosa optimizer is a rustc driver and is ABI-locked to the pinned nightly.
Install the toolchain and host tools, create the standard starter, then run it on the CPU and
compile its kernels:

```bash
rustup toolchain install nightly-2026-05-01
cargo +nightly-2026-05-01 install cargo-binstall
cargo +nightly-2026-05-01 binstall cargo-furiosa-opt
cargo install cargo-generate
cargo generate furiosa-ai/furiosa-opt base-template
cd base-template
cargo run --release --bin gemm
cargo furiosa-opt compile
```

`rust-toolchain.toml` pins the same channel in the generated project.
Cargo activates it automatically in that directory.

A plain `cargo` invocation builds and runs on the CPU. `cargo furiosa-opt` builds for the NPU and
pre-compiles the kernels.

### NPU Requirements

The NPU build also requires the Furiosa SDK and a physical NPU.
It dispatches through the SDK kernel driver and PE runtime (`furiosa-driver-rngd`, `furiosa-smi`, and related components).
See the [SDK documentation](https://developer.furiosa.ai/latest/en/).

The CPU build does not require the SDK; it runs host-side with no NPU dependency.

## The Generated Project

The command path scaffolds the `base-template` starter.
The template is the standard executable example, and continuous integration (CI) generates the current source under an arbitrary name before testing it on the CPU.
Read [Tensor and Contraction](./tensor-and-contraction.md) and [Kernel Validation](./kernel-validation.md) before adapting the generated template.
Use the Kernel Authoring Skill when an agent implements the change.

#### Template Layout

The generated `Cargo.toml` declares `[package.metadata.furiosa-opt]`, which marks the package for kernel compilation.
`src/kernel/` contains each `#[device]` function and `src/kernel/mod.rs` re-exports those kernel modules.
Direct `src/*.rs` files are host binaries that call `launch(kernel, ...)` and contain their host-oracle tests.
Each host binary has an explicit `[[bin]]` entry with `path = "src/<name>.rs"` in `Cargo.toml`.
Do not move host binaries into `src/bin/`, `examples/`, or `tests/`, because the compiler plugin scans cargo targets rooted at `src/`.

```text
base-template/
├── Cargo.toml
├── rust-toolchain.toml
└── src/
    ├── lib.rs
    ├── kernel/
    │   ├── mod.rs
    │   └── <name>_kernel.rs
    └── <name>.rs
```

#### Add a Kernel

1. Add `src/kernel/<name>_kernel.rs` with a `#[device(...)] pub fn <name>_kernel(...)` function.
2. Re-export it from `src/kernel/mod.rs`.
3. Add `src/<name>.rs` with the host program, host oracle, and a call to `launch(<name>_kernel, ...)`.
4. Register the host program as a `[[bin]]` target in `Cargo.toml`.
5. Run the kernel with the Kernel Optimizer command reference.

### Development Tools

The Furiosa IR Optimizer complements the [Furiosa SDK compiler](https://developer.furiosa.ai/latest/en/overview/software_stack.html#furiosa-compiler).
It gives programmers fine-grained control when they write vISA by hand or generate it from another compiler.

#### Kernel Optimizer

The [Kernel Optimizer tool reference](../tools/kernel-optimizer.md) owns CLI syntax and compilation behavior.

#### Language Server

`furiosa-rust-analyzer-proxy` provides standard Rust IDE features with readable Furiosa mapping expressions.
It renders verbose mapping types such as `Stride<Symbol<A>, 8>` as `m![A / 8]`.
See the [Language Server tool reference](../tools/language-server.md) for installation and configuration.

#### Schedule Viewer

The Schedule Viewer visualizes the execution timeline to help identify performance bottlenecks.
Use `furiosa-opt` to export a schedule JSON file, then open it with `furiosa-schedule-viewer`.

For installation and usage, see the [Schedule Viewer tool reference](../tools/schedule-viewer.md).
