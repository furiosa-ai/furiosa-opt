DOCS_DIR := docs
PREBUILT_DIR ?= .prebuilt
FURIOSA_OPT_GITHUB_REPO ?= furiosa-ai/furiosa-opt

# mdbook invokes rustdoc from a temp dir, so the rustup shim there can't see
# rust-toolchain.toml and falls back to the default toolchain (often stable).
# Pin RUSTUP_TOOLCHAIN to the channel parsed from rust-toolchain.toml so book
# doctests compile with the same nightly as `cargo test`.
RUST_TOOLCHAIN_CHANNEL := $(shell sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make check              - Run cargo check"
	@echo "  make clippy             - Run clippy linter"
	@echo "  make fmt                - Run code formatter check"
	@echo "  make test               - Run tests in release mode"
	@echo "  make test-typecheck     - Run tests in release mode with --cfg backend=\"typecheck\""
	@echo "  make mdbook-install     - Install mdbook utility and plugins"
	@echo "  make mdbook-serve       - Serve docs locally"
	@echo "  make mdbook-build       - Build static HTML documentation"
	@echo "  make mdbook-test        - Test code blocks in mdbook"
	@echo "  make licenses           - Regenerate THIRD-PARTY-LICENSES"
	@echo "  make download-release   - Download release artifacts from GitHub"

.PHONY: check
check:
	cargo check --workspace --all-targets

.PHONY: clippy
clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clippy-npu:
	CARGO_TARGET_DIR=target/npu \
	  cargo furiosa-opt --backend npu clippy -p furiosa-opt-std --all-targets -- -D warnings


.PHONY: fmt
fmt:
	cargo fmt --all -- --check

.PHONY: mdbook-install
mdbook-install:
	cargo install mdbook mdbook-mermaid mdbook-pdf
	mdbook-mermaid install $(DOCS_DIR)

.PHONY: mdbook-serve
mdbook-serve:
	mdbook serve $(DOCS_DIR) --hostname 0.0.0.0 --open

.PHONY: mdbook-build
mdbook-build:
	mdbook build $(DOCS_DIR)

.PHONY: test-no-run
test-no-run:
	cargo test --workspace --release --no-run

.PHONY: mdbook-test
mdbook-test: export RUSTUP_TOOLCHAIN = $(RUST_TOOLCHAIN_CHANNEL)
mdbook-test: export CARGO_TARGET_DIR = target/mdbook-test
mdbook-test:
	cargo test -p furiosa-opt-std --release --no-run
	mdbook test $(DOCS_DIR) -L $(CARGO_TARGET_DIR)/release/deps/

.PHONY: mdbook-test-typecheck
mdbook-test-typecheck: export RUSTUP_TOOLCHAIN = $(RUST_TOOLCHAIN_CHANNEL)
mdbook-test-typecheck: export CARGO_TARGET_DIR = target/typecheck
mdbook-test-typecheck:
	cargo furiosa-opt --backend typecheck test -p furiosa-opt-std --release --no-run
	RUSTFLAGS='--cfg backend="typecheck"' mdbook test $(DOCS_DIR) -L $(CARGO_TARGET_DIR)/release/deps/

.PHONY: test
test: test-no-run
	cargo test --workspace --release

.PHONY: test-typecheck
test-typecheck:
	RUSTDOCFLAGS='--cfg backend="typecheck"' CARGO_TARGET_DIR=target/typecheck \
	  cargo furiosa-opt --backend typecheck test --workspace --release

.PHONY: licenses
licenses:
	cargo about generate about.hbs -o THIRD-PARTY-LICENSES

.PHONY: download-release
download-release:
	@set -e; \
	VERSION=v$$(awk -F'"' '/^version[[:space:]]*=[[:space:]]*"/{print $$2; exit}' furiosa-mapping/Cargo.toml); \
	BRANCH=$$(git branch --show-current); \
	if [ "$$BRANCH" = "snapshot" ]; then \
	    RELEASE_TAG=$$(git log -1 --format=%B HEAD | grep 'Snapshot tag:' | awk '{print $$3}'); \
	    [ -n "$$RELEASE_TAG" ] || { echo "ERROR: 'Snapshot tag:' not found in the latest commit message" >&2; exit 1; }; \
	elif [ "$$BRANCH" = "main" ]; then \
	    RELEASE_TAG="$$VERSION"; \
	else \
	    echo "ERROR: download-release must be run from the 'main' or 'snapshot' branch (current: '$$BRANCH')" >&2; \
	    exit 1; \
	fi; \
	TARGET=$$(rustc -vV | sed -n 's/^host: //p'); \
	mkdir -p "$(PREBUILT_DIR)"; \
	echo "Downloading $$RELEASE_TAG (artifacts $$VERSION) for $$TARGET from $(FURIOSA_OPT_GITHUB_REPO)..."; \
	gh release download "$$RELEASE_TAG" \
	    --repo "$(FURIOSA_OPT_GITHUB_REPO)" \
	    --dir "$(PREBUILT_DIR)" \
	    --clobber; \
	if [ "$$(uname)" = "Darwin" ]; then TAR=gtar; else TAR=tar; fi; \
	$$TAR -xzf "$(PREBUILT_DIR)/cargo-furiosa-opt-$$VERSION-$$TARGET.tgz" -C "$(PREBUILT_DIR)/"; \
	printf 'cp %s ~/.cargo/bin/cargo-furiosa-opt\n' \
	    "$$(pwd)/$(PREBUILT_DIR)/cargo-furiosa-opt-$$VERSION-$$TARGET/cargo-furiosa-opt" \
	    > "$(PREBUILT_DIR)/activate.sh"; \
	printf 'cp %s ~/.cargo/bin/furiosa-opt-driver\n' \
	    "$$(pwd)/$(PREBUILT_DIR)/cargo-furiosa-opt-$$VERSION-$$TARGET/furiosa-opt-driver" \
	    >> "$(PREBUILT_DIR)/activate.sh"; \
	printf 'export FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT=%s\n' \
	    "$$(pwd)/$(PREBUILT_DIR)/libfuriosa_mapping_impl-$$VERSION-$$TARGET.a" \
	    >> "$(PREBUILT_DIR)/activate.sh"; \
	printf 'export FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT=%s\n' \
	    "$$(pwd)/$(PREBUILT_DIR)/libfuriosa_opt_lower_impl-$$VERSION-$$TARGET.a" \
	    >> "$(PREBUILT_DIR)/activate.sh"

	@echo ""; \
	echo "Run: source $(PREBUILT_DIR)/activate.sh"; \
	echo ""; \
	echo "This will:"; \
	echo "  - Install cargo-furiosa-opt to ~/.cargo/bin/"; \
	echo "  - Set FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT"; \
	echo "  - Set FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT"
