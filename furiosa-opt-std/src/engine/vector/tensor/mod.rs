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
//! # Initialization / Entry (on CollectTensor / ContractTensor)
//! ```text
//! vector_init()                              → VectorInitTensor
//!   ├─ vector_intra_slice_tag(TagMode) → VectorBranchTensor           { VeOrder::IntraFirst }
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
//! Path 1 (intra only):     vector_init → vector_intra_slice_tag → [intra stages] → vector_final
//! Path 2 (intra → inter):  vector_init → vector_intra_slice_tag → [intra stages] → vector_inter_slice_reduce → vector_final
//! Path 3 (inter only):     vector_init → vector_inter_slice_reduce → vector_final
//! Path 4 (inter → intra):  vector_init → vector_inter_slice_reduce → vector_intra_slice_tag → [intra stages] → vector_final
//! ```
//!
//! # Intra-Slice Stage Flow (IntraSliceStage)
//! ```text
//! Tag → Logic → Fxp → FxpToFp → Narrow(Way4) → Fp → IntraSliceReduce → FpDiv → Widen(Way8) → FpToFxp → Clip
//! ```
//!
//! # Filter
//! `vector_filter` is available after intra-slice stages and after inter-slice reduce
//! (when `VeOrder::IntraFirst`). It applies VE-level branch filtering.
//!
//! # VeOrder Tracking
//! The `const VE_ORDER: VeOrder` parameter tracks which unit was entered first:
//! - `IntraFirst` — after `vector_intra_slice_tag` / `vector_intra_slice_unzip`
//! - `InterFirst` — after `vector_inter_slice_reduce` from `VectorInitTensor`
//!
//! Once set, `VeOrder` is preserved through all subsequent operations.
//! It prevents re-entering the same VE block as an initial entry path.

mod vector_tensor;
mod vector_tensor_pair;
mod verify;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
pub use vector_tensor::*;
pub use vector_tensor_pair::*;

use crate::context::Tu;
use crate::engine::CanApplyVectorInit;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::CurrentBackend;
use crate::tensor::tu::{Position, TuTensor};

pub(crate) type VeTensorShape<Chip, Cluster, Slice, Time, Packet> =
    m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

/// After the vector engine (`vector_final`).
///
/// Produced by [`vector_tensor::VectorTensor::vector_final`].
#[derive(Debug)]
pub struct PositionVectorFinal;

impl Position for PositionVectorFinal {}

/// Tensor after the vector engine (after `vector_final()`).
pub type VectorFinalTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionVectorFinal, D, Chip, Cluster, Slice, Time, Packet, B>;

//
// The Vector Engine accepts only `VeScalar` inputs (hardware constraint), so
// the bound lives on the impl rather than on a wider trait.
// ANCHOR: vector_init_impl
impl<'l, const T: Tu, P: CanApplyVectorInit, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Initializes Vector Engine processing for this tensor.
    #[primitive(TuTensor::vector_init)]
    pub fn vector_init(self) -> VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet> {
        VectorInitTensor::new(self.ctx, self.inner)
    }
}
// ANCHOR_END: vector_init_impl
