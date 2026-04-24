# Furiosa Optimizer

[![Build](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/build.yml/badge.svg)](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/build.yml)
[![Deploy](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/deploy.yml/badge.svg)](https://github.com/furiosa-ai/furiosa-opt/actions/workflows/deploy.yml)
[![Book](https://img.shields.io/badge/docs-book-blue)](https://jubilant-adventure-v3r1p2w.pages.github.io/book/)
[![Rustdoc](https://img.shields.io/badge/docs-rustdoc-blue)](https://jubilant-adventure-v3r1p2w.pages.github.io/rustdoc/furiosa_visa_std/)

Public crates for the Furiosa NPU optimizer.
Private crates (compiler, CLI, device tests) live in [`furiosa-ai/npu-opt`](https://github.com/furiosa-ai/npu-opt).

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

Install Rust nightly via [rustup](https://www.rust-lang.org/tools/install).

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

![furiosa-rust-analyzer-proxy demo](assets/furiosa-rust-analyzer-proxy-demo.png)

We also provides `furiosa-rust-analyzer-proxy` to improve your developer experience. It converts hard-to-read mappiing types like 
`Pair<Stride<Symbol<K, _>, 8>, Modulo<Symbol<N, _>, 128>>` into a mapping expression `m![K / 8, N % 128]` for better readability.

To use it, download the binary from [GitHub Releases](https://github.com/furiosa-ai/furiosa-opt/releases) (`furiosa-rust-analyzer-proxy-x86_64-unknown-linux-gnu`),
then setup IDE to use proxy as a rust-analyzer.

e.g. In VSCode, add the following to your `settings.json`:

```jsonc
{
  "rust-analyzer.server.path": "/usr/local/bin/furiosa-rust-analyzer-proxy",
  // Optional; Recommended for better transformation
  "rust-analyzer.inlayHints.maxLength": null
}
```

By default, the proxy uses `rust-analyzer` found in `PATH` as the upstream server.
To specify a custom `rust-analyzer` path, set the `FURIOSA_RUST_ANALYZER_PROXY_UPSTREAM` environment variable.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
