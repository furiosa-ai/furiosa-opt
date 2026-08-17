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
//! are only available from flit-normalized positions (Collect onwards); the
//! pre-Collect stages (Begin / Fetch / Switch) must go through `collect` first.
//!
//! ```text
//! BeginTensor              (PositionBegin)
//!     └── fetch                    →  FetchTensor
//!
//! FetchTensor              (PositionFetch)
//!     ├── fetch_mask               →  FetchMaskTensor
//!     ├── fetch_table_lookup       →  FetchTableLookupTensor
//!     ├── fetch_cast               →  FetchCastTensor
//!     ├── fetch_zero_point_sub     →  FetchZeroPointSubTensor
//!     ├── switch                   →  SwitchTensor       (fetch adapter skipped)
//!     └── collect                  →  CollectTensor      (fetch adapter skipped)
//!
//! FetchMaskTensor          (PositionFetchMask)
//!     ├── fetch_table_lookup       →  FetchTableLookupTensor
//!     ├── fetch_cast               →  FetchCastTensor
//!     ├── fetch_zero_point_sub     →  FetchZeroPointSubTensor
//!     ├── switch                   →  SwitchTensor
//!     └── collect                  →  CollectTensor
//!
//! FetchTableLookupTensor   (PositionFetchTableLookup)
//!     ├── fetch_cast               →  FetchCastTensor
//!     ├── fetch_zero_point_sub     →  FetchZeroPointSubTensor
//!     ├── switch                   →  SwitchTensor
//!     └── collect                  →  CollectTensor
//!
//! FetchCastTensor          (PositionFetchCast)
//!     ├── fetch_zero_point_sub     →  FetchZeroPointSubTensor
//!     ├── switch                   →  SwitchTensor
//!     └── collect                  →  CollectTensor
//!
//! FetchZeroPointSubTensor  (PositionFetchZeroPointSub)  [i5/i9 staging]
//!     ├── switch                   →  SwitchTensor
//!     └── collect                  →  CollectTensor
//!     (the i5/i9 stream can only reach `contract_outer` after collect; it is
//!      not MaterializableScalar, so to_trf/to_vrf/transpose/commit reject it)
//!
//! SwitchTensor             (PositionSwitch)
//!     └── collect                  →  CollectTensor
//!
//! CollectTensor            (PositionCollect)
//!     ├── to_trf                   →  TrfTensor
//!     ├── to_vrf                   →  VrfTensor
//!     ├── contract_outer(trf)
//!     │     →  ContractOuterTensor  ─contract_packet→  ContractPacketTensor
//!     │     ─contract_time→  ContractTimeTensor  ─contract_lane→  ContractTensor
//!     ├── cast                     →  CastTensor
//!     ├── transpose                →  TransposeTensor
//!     ├── vector_init              →  VectorInitTensor   (handed to `crate::engine::vector`)
//!     └── commit_trim              →  CommitTrimTensor
//!
//! ContractTensor           (PositionContraction)
//!     ├── cast                     →  CastTensor
//!     ├── transpose                →  TransposeTensor
//!     ├── vector_init              →  VectorInitTensor
//!     └── commit_trim              →  CommitTrimTensor
//!
//! VectorFinalTensor        (PositionVectorFinal — produced by `VectorTensor::vector_final`)
//!     ├── cast                     →  CastTensor
//!     ├── transpose                →  TransposeTensor
//!     ├── to_vrf                   →  VrfTensor
//!     └── commit_trim              →  CommitTrimTensor
//!
//! CastTensor               (PositionCast)
//!     ├── transpose                →  TransposeTensor
//!     └── commit_trim              →  CommitTrimTensor
//!
//! TransposeTensor          (PositionTranspose)
//!     └── commit_trim              →  CommitTrimTensor
//!
//! CommitTrimTensor         (PositionCommitTrim)
//!     ├── commit_cast              →  CommitCastTensor
//!     ├── commit_cast_relu         →  CommitCastTensor   (the same cast, ReLU fused)
//!     ├── commit_valid_count_pack  →  CommitValidCountPackTensor
//!     ├── commit                   →  DmTensor
//!     └── commit_view              →  (writes to existing view)
//!
//! CommitCastTensor         (PositionCommitCast)
//!     ├── commit                   →  DmTensor
//!     └── commit_view              →  (writes to existing view)
//!
//! CommitValidCountPackTensor   (PositionCommitValidCountPack)
//!     ├── commit                   →  DmTensor
//!     └── commit_view              →  (writes to existing view)
//! ```

pub mod cast;
pub mod collect;
pub mod commit;
pub mod commit_adapter;
pub mod contraction;
pub mod fetch;
pub mod fetch_adapter;
pub mod switch;
pub mod transpose;
pub mod vector;

// Re-exports so `use crate::engine::*` (and the prelude) bring engine-facing
// types into scope.
pub use cast::*;
pub use collect::*;
pub use commit_adapter::{
    CommitCastTensor, CommitTrimTensor, CommitValidCountPackTensor, PositionCommitCast, PositionCommitTrim,
    PositionCommitValidCountPack,
};
pub use contraction::*;
pub use fetch::*;
pub use fetch_adapter::*;
pub use switch::*;
pub use transpose::*;

use crate::engine::vector::tensor::PositionVectorFinal;
use crate::tensor::tu::{Position, PositionBegin};

// ============================================================================
// `CanApplyXxx` marker traits — pipeline adjacency.
//
// `impl CanApplyXxx for PositionYyy {}` reads as "the `Yyy` typestate can enter
// the `Xxx` engine". These are the *only* edges in the pipeline graph; adding
// or removing one here is how the topology changes.
// ============================================================================

/// Source positions that can enter the Fetch Sequencer stage.
pub trait CanApplyFetch: Position {}

/// Source positions that can enter the Fetch Adapter's masking stage.
pub trait CanApplyFetchMask: Position {}

/// Source positions that can enter the Fetch Adapter's table-lookup stage.
pub trait CanApplyFetchTableLookup: Position {}

/// Source positions that can enter the Fetch Adapter's type-casting stage.
pub trait CanApplyFetchCast: Position {}

/// Source positions that can enter the Fetch Adapter's zero-point-subtraction
/// stage, which widens an integer stream to its contraction-engine staging type (i4->i5,
/// i8->i9).
pub trait CanApplyFetchZeroPointSub: Position {}

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

/// Source positions that can enter the Commit Adapter's trimming stage.
pub trait CanApplyCommitTrim: Position {}

/// Source positions that can enter the Commit Adapter's type-casting
/// stage (which folds in an optional ReLU at the hardware level).
pub trait CanApplyCommitCast: Position {}

/// Source positions that can enter the Commit Adapter's
/// valid-count-packing stage.
pub trait CanApplyCommitValidCountPack: Position {}

/// Source positions that can commit to data memory.
///
/// Only positions with a flit-normalized (32-byte) packet can commit — the
/// pre-Collect stages (`Begin`, `Fetch`, `Switch`) are excluded.
pub trait CanApplyCommit: Position {}

impl CanApplyFetch for PositionBegin {}

impl CanApplyFetchMask for PositionFetch {}

impl CanApplyFetchTableLookup for PositionFetch {}
impl CanApplyFetchTableLookup for PositionFetchMask {}

impl CanApplyFetchCast for PositionFetch {}
impl CanApplyFetchCast for PositionFetchMask {}
impl CanApplyFetchCast for PositionFetchTableLookup {}

// Zero-point subtraction may follow any fetch-adapter stage (its output i5/i9
// staging then flows through switch/collect only into `contract_outer`).
impl CanApplyFetchZeroPointSub for PositionFetch {}
impl CanApplyFetchZeroPointSub for PositionFetchMask {}
impl CanApplyFetchZeroPointSub for PositionFetchTableLookup {}
impl CanApplyFetchZeroPointSub for PositionFetchCast {}

impl CanApplySwitch for PositionFetch {}
impl CanApplySwitch for PositionFetchMask {}
impl CanApplySwitch for PositionFetchTableLookup {}
impl CanApplySwitch for PositionFetchCast {}
impl CanApplySwitch for PositionFetchZeroPointSub {}

impl CanApplyCollect for PositionFetch {}
impl CanApplyCollect for PositionFetchMask {}
impl CanApplyCollect for PositionFetchTableLookup {}
impl CanApplyCollect for PositionFetchCast {}
impl CanApplyCollect for PositionFetchZeroPointSub {}
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

// Commit Adapter pipeline (per HW spec):
//   Main: trim → cast(+ReLU) → commit
//   Sub:  trim → valid_count_pack → commit
//
// `trim` is mandatory and runs first: it is the only adapter stage reachable
// off the source engines. `cast` / `valid_count_pack` chain after `trim`, and
// `commit` / `commit_view` (the sequencer stage) are reachable only from a
// post-trim adapter position, so every commit is trimmed first.
impl CanApplyCommitTrim for PositionCollect {}
impl CanApplyCommitTrim for PositionContraction {}
impl CanApplyCommitTrim for PositionVectorFinal {}
impl CanApplyCommitTrim for PositionCast {}
impl CanApplyCommitTrim for PositionTranspose {}

impl CanApplyCommitCast for PositionCommitTrim {}

impl CanApplyCommitValidCountPack for PositionCommitTrim {}

impl CanApplyCommit for PositionCommitTrim {}
impl CanApplyCommit for PositionCommitCast {}
impl CanApplyCommit for PositionCommitValidCountPack {}
