# furiosa-opt-rt

Runs device functions on a device from the host. A `Device` opens the chips, a
`Function` is a device function loaded on it from its image, a `Launch` runs it on buffers. Why it is built the way it is: [DESIGN.md](DESIGN.md).
What each type does: `cargo doc -p furiosa-opt-rt --open`.

```
furiosa-opt-rt/
├── src/        host library and the `furiosa-opt-rt` CLI
├── abi/        what compiler, host, bootloader and firmware agree on (no_std)
├── ipc/        the host ↔ firmware wire (no_std)
├── bootloader/ source of the bootloader accepted by secure boot
└── prebuilt/   the signed bootloader and vendor-built firmware images embedded by the crate
```

## Build

The public checkout and packaged crate embed both images from `prebuilt/`; building the host
runtime needs no device cross-compiler.

```bash
cargo build -p furiosa-opt-rt --release
```

## Run

```bash
target/release/furiosa-opt-rt function.bin --input a.bin --output c.bin
```

The device opens with the topology the function was compiled for, on any exposed chips unless
`--among 0,1` narrows them. Each `--input` and `--output` file holds one argument's bytes for every
chip, in chip order; an output file is overwritten with what the function left.

`RUST_LOG=furiosa_opt_firmware=debug` has each cluster report where every launch's time went; the
lines appear in `/sys/kernel/debug/rngd/mgmt<device>/pe_log/pe<n>_log`.

## Test

```bash
cargo test -p furiosa-opt-rt --lib                  # host runtime
(cd ipc && cargo test); (cd abi && cargo test)      # contracts
```

On hardware, `furiosa-opt-examples` exercises the whole path: `cargo furiosa-opt test -p
furiosa-opt-examples --test pe_count_tests`.
