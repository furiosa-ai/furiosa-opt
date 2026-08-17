# The furiosa-opt CLI: docker run -v "$PWD":/work ghcr.io/furiosa-ai/furiosa-opt:<tag> --backend npu build
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential libclang-dev gcc-aarch64-linux-gnu \
 && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
COPY rust-toolchain.toml /opt/toolchain/
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none \
 && (cd /opt/toolchain && rustup show) \
 && rustup default "$(rustup toolchain list | awk '{print $1; exit}')" \
 && rustup component add rustc-dev llvm-tools

COPY dist /tmp/dist
RUN tar xzf /tmp/dist/cargo-furiosa-opt-*.tgz -C /tmp/dist \
 && install /tmp/dist/cargo-furiosa-opt-*/cargo-furiosa-opt /tmp/dist/cargo-furiosa-opt-*/furiosa-opt-driver /opt/cargo/bin/ \
 && install -D /tmp/dist/libfuriosa_mapping_impl-*.a /opt/furiosa/libfuriosa_mapping_impl.a \
 && install -D /tmp/dist/libfuriosa_opt_lower_impl-*.a /opt/furiosa/libfuriosa_opt_lower_impl.a \
 && rm -rf /tmp/dist \
 && cargo furiosa-opt --help >/dev/null
ENV FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT=/opt/furiosa/libfuriosa_mapping_impl.a \
    FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT=/opt/furiosa/libfuriosa_opt_lower_impl.a

WORKDIR /work
ENTRYPOINT ["cargo", "furiosa-opt"]
