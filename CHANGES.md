# Changelog

All notable changes to `furiosa-opt` are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [v0.5.0]

### Changed
Breaking changes come first, each with what to write instead.

- Transfers no longer take an address. Call `to_hbm` and `to_dm` without one, and reach for `to_hbm_at` or `to_dm_at` only when the address matters. Build a DM tensor with `DmTensor::new` in place of `DmTensor::from_addr`. The runtime allocator places what `to_hbm` produces.
- Host buffers convert through `from_vec` and `into_vec`, replacing `from_buf` and `into_buf`.
- `Tensor::reshape` consumes the tensor it reshapes, on owned tensors and on views alike, and a size mismatch now fails to compile instead of at run time.
- Reading a vector stash twice no longer compiles. The operand type carries the read-once rule that was previously a convention.
- `furiosa-mapping`: `replace_padding` is gone. Write `padding(_, Top)` for the same result.
- A dense contraction no longer materializes the broadcast of its operands, which dominated wide-output GEMM on the host. Output is byte for byte what it was.
- Host to HBM and HBM to host transfers on the NPU backend go through the DMA heap without a staging copy.
- A binary built for `backend="npu"` carries its compiled kernels, so it runs on a machine that has no compiler installed.

### Added
- A kernel can return several HBM tensors as a tuple, in the order it returns them.
- Profiling a run on the device. Install a `tracing` subscriber on the `span::npu` target and each launch reports its spans with their cycle windows. `furiosa-opt-examples` has a minimal example.
- Gathering and scattering DRAM rows through an index tensor: `dma_gather_scaled`, `dma_gather_unscaled`, and `dma_scatter`, with the `dma_tails` mapping primitive to describe the burst packet's tail axis.
- Filling a DM tensor view with a typed constant: `memset(value)`.
- Baked table lookup as a fetch adapter, `fetch_table_lookup`, covering paired nibbles and a non-paired `f8e4m3` to `bf16` table, plus `fetch_zero_point_sub` for zero-point subtraction on fetch.
- Device scalar types `i5`, `i9`, and `f8e5m2`. A 4-bit tensor packs two codes per byte.
- A tile offset can be an expression, such as `tile(i * 2)`.
- Engine configurations report what is wrong before a run rather than producing a bad result. `furiosa-opt-lower` publishes the verifications, and the switch engine checks a named `SwitchConfig` and its custom-broadcast snoop bitmap through a structured `SwitchError`.
- The `emulation` backend runs clip, float, floating-point divide, and local reduce vector nodes, which it used to reject.
- The book has an appendix on the `cargo furiosa-opt` CLI.

### Removed
- The `simulation` backend and its `MathStorage` interpreter. `emulation` replaces it and is what a plain `cargo build` or `cargo test` now uses.

### Fixed
- A `tile` start offset on the fetch path counted bytes instead of elements, so a `bf16` tile that asked for element 16 landed on element 8; it now counts elements, as the commit side and the DMA path always did, and the `commit_view_tile` example guards it.
- A tensor that a nested loop body read or aliased from an enclosing scope produced a wrong result; it now lowers as written.
- A write through an `HbmTensorViewMut` covering a whole parameter never reached the caller's tensor; it now lands in place.
- A valid Time Reducer fold with a `Sequential` accumulator was rejected for exceeding the accumulator buffer; the bound now counts only the axis each mode pads, and the error spells out the product it checked.
- A device call the compiler could not translate, a generic `#[device]` entrypoint, and a scalar cast it could not evaluate each surfaced as an internal compiler error; they now report what is wrong.
- A process could not start while chip 0 was busy, even with every other chip idle; it now takes the first free NPU among the chips the host exposes.

## [v0.4.0]

### Changed
- HBM, DM, and TU tensors now carry an optional address; explicit-address transfers moved to dedicated `*_at` methods (`to_trf_at`, `to_vrf_at`, `to_dm_at`, `to_hbm_at`, `to_dm_pcopy_at`, `commit_at`).

### Added
- `furiosa-mapping`: escaped constant numbers in `m!` and `i!`.

### Fixed
- Fixed secure boot support.

## [v0.3.0]

### Added
- Added `simulation` and `typecheck` backends.

### Changed
- Revised some APIs of virtual ISA.
- `Tensor::reshape` reworked as a `RawTensor` primitive in `furiosa-visa-std`.

### Fixed
- Fixed the binary path to download in `cargo-furiosa-opt`.

## [v0.2.0] - Initial release
