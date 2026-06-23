.PHONY: help check clippy dylint fmt test test-typecheck mdbook-install mdbook-serve mdbook-build mdbook-test mdbook-test-typecheck test-no-run licenses

DOCS_DIR := docs

# mdbook invokes rustdoc from a temp dir, so the rustup shim there can't see
# rust-toolchain.toml and falls back to the default toolchain (often stable).
# Pin RUSTUP_TOOLCHAIN to the channel parsed from rust-toolchain.toml so book
# doctests compile with the same nightly as `cargo test`.
RUST_TOOLCHAIN_CHANNEL := $(shell sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)

help:
	@echo "Available commands:"
	@echo "  make check          - Run cargo check"
	@echo "  make clippy         - Run clippy linter"
	@echo "  make fmt            - Run code formatter check"
	@echo "  make test           - Run tests in release mode"
	@echo "  make test-typecheck - Run tests in release mode with --cfg backend=\"typecheck\""
	@echo "  make mdbook-install - Install mdbook utility and plugins"
	@echo "  make mdbook-serve   - Serve docs locally"
	@echo "  make mdbook-build   - Build static HTML documentation"
	@echo "  make mdbook-test    - Test code blocks in mdbook"
	@echo "  make licenses       - Regenerate THIRD-PARTY-LICENSES"

check:
	cargo check --workspace --all-targets

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clippy-npu:
	CARGO_TARGET_DIR=target/npu \
	  cargo furiosa-opt --backend npu clippy -p furiosa-opt-std --all-targets -- -D warnings


fmt:
	cargo fmt --all -- --check

mdbook-install:
	cargo install mdbook mdbook-mermaid mdbook-pdf
	mdbook-mermaid install $(DOCS_DIR)

mdbook-serve:
	mdbook serve $(DOCS_DIR) --hostname 0.0.0.0 --open

mdbook-build:
	mdbook build $(DOCS_DIR)

test-no-run:
	cargo test --workspace --release --no-run

mdbook-test: export RUSTUP_TOOLCHAIN = $(RUST_TOOLCHAIN_CHANNEL)
mdbook-test: export CARGO_TARGET_DIR = target/mdbook-test
mdbook-test:
	cargo test -p furiosa-opt-std --release --no-run
	mdbook test $(DOCS_DIR) -L $(CARGO_TARGET_DIR)/release/deps/

mdbook-test-typecheck: export RUSTUP_TOOLCHAIN = $(RUST_TOOLCHAIN_CHANNEL)
mdbook-test-typecheck: export CARGO_TARGET_DIR = target/typecheck
mdbook-test-typecheck:
	cargo furiosa-opt --backend typecheck test -p furiosa-opt-std --release --no-run
	RUSTFLAGS='--cfg backend="typecheck"' mdbook test $(DOCS_DIR) -L $(CARGO_TARGET_DIR)/release/deps/

test: test-no-run
	cargo test --workspace --release

test-typecheck:
	RUSTDOCFLAGS='--cfg backend="typecheck"' CARGO_TARGET_DIR=target/typecheck \
	  cargo furiosa-opt --backend typecheck test --workspace --release

licenses:
	cargo about generate about.hbs -o THIRD-PARTY-LICENSES
