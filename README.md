# Furiosa Optimizer

[![Build](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/build.yml/badge.svg)](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/build.yml)
[![Deploy](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/deploy.yml/badge.svg)](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/deploy.yml)
[![Book](https://img.shields.io/badge/docs-book-blue)](https://developer.furiosa.ai/furiosa-opt/book/)
[![Rustdoc](https://img.shields.io/badge/docs-rustdoc-blue)](https://developer.furiosa.ai/furiosa-opt/rustdoc/furiosa_opt_std/)

Crates for the Furiosa NPU optimizer.

## Repository Structure

```
furiosa-opt/
├── furiosa-mapping/
├── furiosa-mapping-macro/
├── furiosa-opt-macro/
├── furiosa-opt-std/
├── furiosa-opt-examples/
└── docs/
```

## Setup

### Prerequisites

`emulation` and `typecheck` are pure Rust and build on any host arch (e.g. macOS
Apple silicon); the only build dependency is `bindgen`'s `libclang`. The `npu`
backend is `x86_64-unknown-linux-gnu` only — the SDK and NPU hardware run there.

#### Ubuntu (jammy / noble / resolute)

```bash
sudo apt install libclang-dev gcc-aarch64-linux-gnu
```

- `libclang-dev` — `furiosa-opt-std/build.rs` runs `bindgen`, which loads `libclang.so`.
- `gcc-aarch64-linux-gnu` — `aarch64-linux-gnu-{gcc,as,ld,objcopy}` are invoked when the compiler produces NPU device binaries (`*.bin`).

#### macOS (Apple silicon)

```bash
xcode-select --install                       # provides libclang for bindgen
rustup target add aarch64-apple-darwin       # usually the default host target
```

`cargo-furiosa-opt` is a [rustc driver](https://rustc-dev-guide.rust-lang.org/rustc-driver/intro.html) and is ABI-locked to a specific rustc nightly. Install the matching toolchain (also pinned in [`rust-toolchain.toml`](rust-toolchain.toml)) and the binary via [`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall):

```bash
rustup toolchain install nightly-2026-05-01
cargo +nightly-2026-05-01 install cargo-binstall
cargo +nightly-2026-05-01 binstall cargo-furiosa-opt
```

Run as a cargo subcommand:

```bash
cargo furiosa-opt test --test mnist_tests
```

## Build and Test

```bash
make check    # cargo check --workspace --all-targets
make fmt      # cargo fmt --all -- --check
make clippy   # cargo clippy --workspace --all-targets -- -D warnings
make test     # cargo test --workspace --release
```

## Documentation

```bash
make mdbook-install   # install mdbook + plugins
make mdbook-serve     # serve docs locally
make mdbook-build     # build static HTML
make mdbook-test      # test code blocks in mdbook
```

## Tooling

### Language Server

`furiosa-rust-analyzer-proxy` is a proxy for `rust-analyzer` that provides standard Rust IDE features with enhanced support for mapping expressions.
It delegates regular Rust language-server behavior to `rust-analyzer` and rewrites editor-facing results so verbose types like `Stride<Symbol<A>, 8>` appear as readable mapping expressions like `m![A / 8]`.

![furiosa-rust-analyzer-proxy demo](assets/furiosa-rust-analyzer-proxy-demo.png)

For installation and configuration, see the [Language Server documentation](https://developer.furiosa.ai/furiosa-opt/book/appendix/language-server.html).

### Schedule Viewer

The Schedule Viewer visualizes the execution timeline to help identify performance bottlenecks.
Use `furiosa-opt` to export a schedule JSON file, then open it with `furiosa-schedule-viewer`.

For installation and usage, see the [Schedule Viewer documentation](https://developer.furiosa.ai/furiosa-opt/book/appendix/schedule-viewer.html).

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
