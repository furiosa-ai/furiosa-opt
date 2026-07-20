# Changelog

All notable changes to `furiosa-opt` are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed
- Removed the `simulation` backend and `MathStorage` (its `ArrayD<Opt<D>>` mapping-expression interpreter). `emulation` is now the default host-side backend for a plain `cargo build`/`cargo test`.

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
