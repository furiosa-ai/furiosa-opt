.PHONY: help check clippy dylint fmt test test-typecheck mdbook-install mdbook-serve mdbook-build mdbook-test mdbook-test-typecheck test-no-run licenses

DOCS_DIR := docs

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
	rustup run nightly-2026-05-01 mdbook serve $(DOCS_DIR) --hostname 0.0.0.0 --open

mdbook-build:
	mdbook build $(DOCS_DIR)

test-no-run:
	cargo test --workspace --release --no-run

mdbook-test: test-no-run
	rustup run nightly-2026-05-01 mdbook test $(DOCS_DIR) -L target/release/deps/

mdbook-test-typecheck:
	CARGO_TARGET_DIR=target/typecheck \
	  cargo furiosa-opt --backend typecheck test --workspace --release --no-run
	RUSTFLAGS='--cfg backend="typecheck"' rustup run nightly-2026-05-01 \
	  mdbook test $(DOCS_DIR) -L target/typecheck/release/deps/

test: test-no-run
	cargo test --workspace --release

test-typecheck:
	RUSTDOCFLAGS='--cfg backend="typecheck"' CARGO_TARGET_DIR=target/typecheck \
	  cargo furiosa-opt --backend typecheck test --workspace --release

licenses:
	cargo about generate about.hbs -o THIRD-PARTY-LICENSES
