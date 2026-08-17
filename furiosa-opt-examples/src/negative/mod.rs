//! Kernels that MUST fail to compile: each one states something the language or the hardware does
//! not allow, and exists so the compiler's rejection stays pinned.
//!
//! This is the intent-based half of the failing examples, and the distinction is the point:
//!
//! - A kernel **here** is a rejection fixture. Failing is the expected outcome, and the failure is
//!   part of the contract: `snapshot.toml` pins the exact stage and a substring of the diagnostic,
//!   so a message that drifts or a rejection that disappears both break the gate.
//! - A kernel that is a perfectly legal program the compiler cannot lower yet lives in the private
//!   `npu-opt-examples` crate under `unsupported`. Those are expected to move to `edf` as the
//!   compiler improves; the ones here are not.
//!
//! Pass and fail run the other way round here, and the gate knows it: for a kernel in this module
//! being refused is the pass, so a *later* stage is the regression. A fixture that starts compiling
//! fails the gate instead of being reported as an improvement, and the summary counts these kernels
//! on their own line (refused vs compiled) rather than inside `Passing`.
//!
//! A snapshot entry of `"edf"` here is therefore a recorded failure, not a pass: the program compiled
//! when it should have been refused, so the check is still missing. The 17 such entries today are all
//! `switch_assertions::invalid_*`.
//!
//! Most of these kernels still run on the CPU, because what they violate is a
//! device-translation rule rather than an arithmetic one. The answer-key tests exercise that,
//! asserting the panic where the emulator catches it too; only compilation must fail.

pub mod contract_outer_assertions;
pub mod dma;
pub mod generic_device;
pub mod memset;
pub mod runtime_if_scalar;
pub mod runtime_panic;
pub mod scalar_cast_diag;
pub mod scatter_gather;
pub mod switch_assertions;
pub mod vector_engine;
