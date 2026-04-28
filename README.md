# Furiosa Optimizer

[![Build](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/build.yml/badge.svg)](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/build.yml)
[![Deploy](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/deploy.yml/badge.svg)](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/deploy.yml)
[![Book](https://img.shields.io/badge/docs-book-blue)](https://developer.furiosa.ai/furiosa-opt/book/)
[![Rustdoc](https://img.shields.io/badge/docs-rustdoc-blue)](https://developer.furiosa.ai/furiosa-opt/rustdoc/furiosa_visa_std/)

Crates for the Furiosa NPU optimizer.

## Repository Structure

```
furiosa-opt/
├── furiosa-mapping/
├── furiosa-mapping-macro/
├── furiosa-opt-macro/
├── furiosa-visa-std/
├── furiosa-visa-examples/
└── docs/
```

## Setup

### Prerequisites

Ubuntu (jammy / noble / resolute):

```bash
sudo apt install libclang-dev gcc-aarch64-linux-gnu
```

- `libclang-dev` — `furiosa-visa-std/build.rs` runs `bindgen`, which loads `libclang.so`.
- `gcc-aarch64-linux-gnu` — `aarch64-linux-gnu-{gcc,as,ld,objcopy}` are invoked when the compiler produces NPU device binaries (`*.bin`).

`cargo-furiosa-opt` is a [rustc driver](https://rustc-dev-guide.rust-lang.org/rustc-driver/intro.html) and is ABI-locked to a specific rustc nightly. Install the matching toolchain (also pinned in [`rust-toolchain.toml`](rust-toolchain.toml)) and the binary via [`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall):

```bash
rustup toolchain install nightly-2025-12-12

cargo install cargo-binstall
cargo binstall cargo-furiosa-opt
```

Run as a cargo subcommand:

```bash
cargo furiosa-opt test --test mnist_tests   # set FURIOSA_OPT_NPUS=<chip>[,<chip>...] to pick non-default chips
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

### `furiosa-rust-analyzer-proxy`

A rust-analyzer wrapper that rewrites mapping types like `Pair<Stride<Symbol<K>, 8>, Modulo<Symbol<N>, 128>>` into the more readable `m![K / 8, N % 128]`. Download the [release binary](https://github.com/furiosa-ai/furiosa-opt/releases) and point your IDE at it; e.g. in VSCode `settings.json`:

```jsonc
{
  "rust-analyzer.server.path": "/usr/local/bin/furiosa-rust-analyzer-proxy",
  "rust-analyzer.inlayHints.maxLength": null  // recommended to reduce '_' truncation
}
```

The proxy delegates to `rust-analyzer` from `PATH` by default; override with `FURIOSA_RUST_ANALYZER_PROXY_UPSTREAM`.

![furiosa-rust-analyzer-proxy demo](assets/furiosa-rust-analyzer-proxy-demo.png)

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
