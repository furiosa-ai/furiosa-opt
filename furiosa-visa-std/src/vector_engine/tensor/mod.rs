//! Vector Engine tensor types for pipeline operations.
//!
//! This module provides tensor types that track VE pipeline stages at compile time
//! using the typestate pattern. Operations are expressed as method chaining, with
//! stage transitions enforced by the type system.
//!
//! # Key Types
//! - `VectorTensor`: Unified tensor type for all VE pipeline stages
//! - `VectorInitTensor`: VE input after `vector_init()`, before choosing the first block
//! - `VectorTensorPair`: Pair of tensors for two-group (interleaved) operations
//! - `VectorInterSliceReduceTensor`: After inter-slice reduce
//!
//! # Initialization / Entry (on CollectTensor / AccumulationTensor)
//! ```text
//! vector_init()                              → VectorInitTensor
//!   ├─ vector_intra_slice_branch(BranchMode) → VectorBranchTensor           { VeOrder::IntraFirst }
//!   ├─ vector_intra_slice_unzip(...)         → VectorTensorPair             { VeOrder::IntraFirst }
//!   └─ vector_inter_slice_reduce(op)         → VectorInterSliceReduceTensor { VeOrder::InterFirst }
//! ```
//!
//! # Exit
//! `vector_final()` exits the VE pipeline and returns a `VectorFinalTensor`.
//! From there, `commit`, `cast`, `to_vrf`, or `transpose` are available.
//!
//! # Four Pipeline Paths
//! ```text
//! Path 1 (intra only):     vector_init → vector_intra_slice_branch → [intra stages] → vector_final
//! Path 2 (intra → inter):  vector_init → vector_intra_slice_branch → [intra stages] → vector_inter_slice_reduce → vector_final
//! Path 3 (inter only):     vector_init → vector_inter_slice_reduce → vector_final
//! Path 4 (inter → intra):  vector_init → vector_inter_slice_reduce → vector_intra_slice_branch → [intra stages] → vector_final
//! ```
//!
//! # Intra-Slice Stage Flow (IntraSliceStage)
//! ```text
//! Branch → Logic → Fxp → FxpToFp → Narrow(Way4) → Fp → IntraSliceReduce → FpDiv → Widen(Way8) → FpToFxp → Clip
//! ```
//!
//! # Filter
//! `vector_filter` is available after intra-slice stages and after inter-slice reduce
//! (when `VeOrder::IntraFirst`). It applies VE-level branch filtering.
//!
//! # VeOrder Tracking
//! The `const VE_ORDER: VeOrder` parameter tracks which unit was entered first:
//! - `IntraFirst` — after `vector_intra_slice_branch` / `vector_intra_slice_unzip`
//! - `InterFirst` — after `vector_inter_slice_reduce` from `VectorInitTensor`
//!
//! Once set, `VeOrder` is preserved through all subsequent operations.
//! It prevents re-entering the same VE block as an initial entry path.

mod vector_tensor;
mod vector_tensor_pair;
mod verify;

use furiosa_mapping::Pair;
use furiosa_opt_macro::m;
pub use vector_tensor::*;
pub use vector_tensor_pair::*;

pub(crate) type VeTensorShape<Chip, Cluster, Slice, Time, Packet> =
    m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];
