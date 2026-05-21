//! Tensor Unit pipeline engines.
//!
//! This module owns the **pipeline adjacency matrix**: all `CanApplyXxx`
//! marker traits and the `impl CanApplyXxx for PositionYyy {}` edges that
//! gate which source typestate can enter which engine.
//!
//! Each engine submodule owns its full surface: the `PositionXxx` marker and
//! `XxxTensor` type alias it produces, the `verify_*` helper, and the inherent
//! impl on `TuTensor<P: CanApplyXxx, ...>` that carries the entry method.
//!
//! # Pipeline graph
//!
//! Each `XxxTensor` block below lists *every* outgoing edge from that
//! typestate. The set is normative: it must equal the `impl CanApplyYyy for
//! PositionXxx {}` lines below — that is the wire-up. `commit` / `commit_view`
//! are only available from flit-normalised positions (Collect onwards); the
//! pre-Collect stages (Begin / Fetch / Switch) must go through `collect` first.
//!
//! ```text
//! BeginTensor          (PositionBegin)
//!     └── fetch         →  FetchTensor
//!
//! FetchTensor          (PositionFetch)
//!     ├── switch        →  SwitchTensor
//!     └── collect       →  CollectTensor
//!
//! SwitchTensor         (PositionSwitch)
//!     └── collect       →  CollectTensor
//!
//! CollectTensor        (PositionCollect)
//!     ├── to_trf        →  TrfTensor
//!     ├── to_vrf        →  VrfTensor
//!     ├── contract_outer(trf)
//!     │     →  ContractOuterTensor  ─contract_packet→  ContractPacketTensor
//!     │     ─contract_time→  ContractTimeTensor  ─contract_lane→  ContractTensor
//!     ├── cast          →  CastTensor
//!     ├── transpose     →  TransposeTensor
//!     ├── vector_init   →  VectorInitTensor   (handed to `crate::engine::vector`)
//!     ├── commit        →  DmTensor
//!     └── commit_view   →  (writes to existing view)
//!
//! ContractTensor       (PositionContraction)
//!     ├── cast          →  CastTensor
//!     ├── transpose     →  TransposeTensor
//!     ├── vector_init   →  VectorInitTensor
//!     ├── commit        →  DmTensor
//!     └── commit_view   →  (writes to existing view)
//!
//! VectorFinalTensor    (PositionVectorFinal — produced by `VectorTensor::vector_final`)
//!     ├── cast          →  CastTensor
//!     ├── transpose     →  TransposeTensor
//!     ├── to_vrf        →  VrfTensor
//!     ├── commit        →  DmTensor
//!     └── commit_view   →  (writes to existing view)
//!
//! CastTensor           (PositionCast)
//!     ├── transpose     →  TransposeTensor
//!     ├── commit        →  DmTensor
//!     └── commit_view   →  (writes to existing view)
//!
//! TransposeTensor      (PositionTranspose)
//!     ├── commit        →  DmTensor
//!     └── commit_view   →  (writes to existing view)
//! ```

pub mod cast;
pub mod collect;
pub mod commit;
pub mod contraction;
pub mod fetch;
pub mod switch;
pub mod transpose;
pub mod vector;

// Re-exports so `use crate::engine::*` (and the prelude) bring engine-facing
// types into scope.
pub use cast::*;
pub use collect::*;
pub use contraction::*;
pub use fetch::*;
pub use switch::*;
pub use transpose::*;

use crate::engine::vector::tensor::PositionVectorFinal;
use crate::tensor::tu::{Position, PositionBegin};

/// Size of a single flit in bytes.
///
/// Data flows through the switching network in flit-sized units.
/// Both the collect engine and cast engine normalize packets to exactly one flit.
pub(crate) const FLIT_BYTES: usize = 32;

pub(crate) fn align_up(a: usize, b: usize) -> usize {
    assert_ne!(b, 0);
    a.div_ceil(b) * b
}

pub(crate) fn exact_div(a: usize, b: usize) -> Option<usize> {
    if a.is_multiple_of(b) { Some(a / b) } else { None }
}

// ============================================================================
// `CanApplyXxx` marker traits — pipeline adjacency.
//
// `impl CanApplyXxx for PositionYyy {}` reads as "the `Yyy` typestate can enter
// the `Xxx` engine". These are the *only* edges in the pipeline graph; adding
// or removing one here is how the topology changes.
// ============================================================================

/// Source positions that can enter the Fetch Engine.
pub trait CanApplyFetch: Position {}

/// Source positions that can enter the Switch Engine.
pub trait CanApplySwitch: Position {}

/// Source positions that can enter the Collect Engine.
pub trait CanApplyCollect: Position {}

/// Source positions that can store to the TRF.
pub trait CanApplyToTrf: Position {}

/// Source positions that can store to the VRF.
pub trait CanApplyToVrf: Position {}

/// Source positions that can enter the Outer stage (Contraction Engine entry).
pub trait CanApplyContractOuter: Position {}

/// Source positions that can enter the Vector Engine.
pub trait CanApplyVectorInit: Position {}

/// Source positions that can enter the Cast Engine.
pub trait CanApplyCast: Position {}

/// Source positions that can enter the Transpose Engine.
pub trait CanApplyTranspose: Position {}

/// Source positions that can commit to data memory.
///
/// Only positions with a flit-normalized (32-byte) packet can commit — the
/// pre-Collect stages (`Begin`, `Fetch`, `Switch`) are excluded.
pub trait CanApplyCommit: Position {}

impl CanApplyFetch for PositionBegin {}

impl CanApplySwitch for PositionFetch {}

impl CanApplyCollect for PositionFetch {}
impl CanApplyCollect for PositionSwitch {}

impl CanApplyToTrf for PositionCollect {}

impl CanApplyToVrf for PositionCollect {}
impl CanApplyToVrf for PositionVectorFinal {}

impl CanApplyContractOuter for PositionCollect {}

impl CanApplyVectorInit for PositionCollect {}
impl CanApplyVectorInit for PositionContraction {}

impl CanApplyCast for PositionCollect {}
impl CanApplyCast for PositionContraction {}
impl CanApplyCast for PositionVectorFinal {}

impl CanApplyTranspose for PositionCollect {}
impl CanApplyTranspose for PositionContraction {}
impl CanApplyTranspose for PositionVectorFinal {}
impl CanApplyTranspose for PositionCast {}

impl CanApplyCommit for PositionCollect {}
impl CanApplyCommit for PositionContraction {}
impl CanApplyCommit for PositionVectorFinal {}
impl CanApplyCommit for PositionCast {}
impl CanApplyCommit for PositionTranspose {}
