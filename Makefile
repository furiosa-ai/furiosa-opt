.PHONY: help check clippy fmt test mdbook-install mdbook-serve mdbook-build mdbook-test test-no-run licenses

DOCS_DIR := docs

help:
	@echo "Available commands:"
	@echo "  make check          - Run cargo check"
	@echo "  make clippy         - Run clippy linter"
	@echo "  make fmt            - Run code formatter check"
	@echo "  make test           - Run tests in release mode"
	@echo "  make mdbook-install - Install mdbook utility and plugins"
	@echo "  make mdbook-serve   - Serve docs locally"
	@echo "  make mdbook-build   - Build static HTML documentation"
	@echo "  make mdbook-test    - Test code blocks in mdbook"
	@echo "  make licenses       - Regenerate THIRD-PARTY-LICENSES"

check:
	cargo check --workspace --all-targets

clippy:
	cargo clippy --workspace --all-targets -- -D warnings


fmt:
	cargo fmt --all -- --check

mdbook-install:
	cargo install mdbook mdbook-mermaid mdbook-pdf
	mdbook-mermaid install $(DOCS_DIR)

mdbook-serve:
	rustup run nightly-2025-12-12 mdbook serve $(DOCS_DIR) --hostname 0.0.0.0 --open

mdbook-build:
	mdbook build $(DOCS_DIR)

test-no-run:
	cargo test --workspace --release --no-run

mdbook-test: test-no-run
	rustup run nightly-2025-12-12 mdbook test $(DOCS_DIR) -L target/release/deps/

test: test-no-run
	cargo test --workspace --release

licenses:
	cargo about generate about.hbs -o THIRD-PARTY-LICENSES
