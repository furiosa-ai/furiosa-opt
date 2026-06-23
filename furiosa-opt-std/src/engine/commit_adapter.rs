//! Commit Adapter: per-element transforms applied before the Commit
//! Engine writes to DM.
//!
//! Each stage is a chainable method on a flit-normalized `TuTensor`,
//! returning its own typestate tensor. The chain ends in `.commit(...)`
//! (or `.commit_view(...)`) which performs the actual DM write.
//!
//! Trimming runs first and happens on almost every commit, so the chain
//! is ordered trim-first. The hardware pipeline is `trim → cast(+ReLU)`
//! (main) or `trim → valid_count_pack` (sub):
//!
//! - `commit_trim::<OutPacket>()` → `CommitTrimTensor`
//! - `commit_cast::<OutD>(Activation)` → `CommitCastTensor` (main; ReLU only ever fuses into the cast)
//! - `commit_valid_count_pack(count)` → `CommitValidCountPackTensor` (sub)

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::cast::Cast;
use crate::context::*;
use crate::engine::{CanApplyCommitCast, CanApplyCommitTrim, CanApplyCommitValidCountPack};
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::tu::{Position, TuTensor};

pub(super) use furiosa_opt_lower::COMMIT_VALID_PACKET_SIZES;

/// Element-wise activation fused into the Commit Adapter's type-casting
/// stage. ReLU is not a standalone hardware stage: it exists only fused
/// with a narrowing cast (e.g. `f32` → `bf16` + ReLU).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Activation {
    /// Plain cast, no activation.
    #[default]
    None,
    /// Clamp negative values to zero, fused into the cast.
    Relu,
}

/// After the Commit Adapter's trimming stage.
#[derive(Debug)]
pub struct PositionCommitTrim;

impl Position for PositionCommitTrim {}

/// Tensor streamed after `commit_trim`.
pub type CommitTrimTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCommitTrim, D, Chip, Cluster, Slice, Time, Packet, B>;

/// After the Commit Adapter's type-casting stage.
#[derive(Debug)]
pub struct PositionCommitCast;

impl Position for PositionCommitCast {}

/// Tensor streamed after `commit_cast`.
pub type CommitCastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCommitCast, D, Chip, Cluster, Slice, Time, Packet, B>;

/// After the Commit Adapter's valid-count-packing stage.
#[derive(Debug)]
pub struct PositionCommitValidCountPack;

impl Position for PositionCommitValidCountPack {}

/// Tensor streamed after `commit_valid_count_pack`.
pub type CommitValidCountPackTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCommitValidCountPack, D, Chip, Cluster, Slice, Time, Packet, B>;

// ANCHOR: commit_trim_impl
impl<'l, const T: Tu, P: CanApplyCommitTrim, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's trimming stage.
    ///
    /// Drops the trailing padding from each flit so DM stores only valid
    /// elements. `OutPacket` is the post-trim layout the kernel
    /// promises; the compiler derives the trim count from the input and
    /// output mappings.
    #[primitive(TuTensor::commit_trim)]
    pub fn commit_trim<OutPacket: M>(self) -> CommitTrimTensor<'l, T, D, Chip, Cluster, Slice, Time, OutPacket, B> {
        verify_commit_trim::<D, Packet, OutPacket>();
        // `transpose(false)` is type-system filler; real trim lowering lands with the backend wiring.
        CommitTrimTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: commit_trim_impl

// ANCHOR: commit_cast_impl
impl<'l, const T: Tu, P: CanApplyCommitCast, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's type-casting stage, optionally fusing a
    /// ReLU.
    ///
    /// Folds an `f32` → `bf16` (or other narrowing) cast into the commit
    /// path, leaving the [Cast Engine](crate::engine::cast) free for
    /// sub-context Vector Engine work. `activation` selects the optional
    /// fused ReLU; ReLU has no standalone hardware stage.
    #[primitive(TuTensor::commit_cast)]
    pub fn commit_cast<OutD: Scalar>(
        self,
        _activation: Activation,
    ) -> CommitCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: Cast<OutD>,
    {
        verify_commit_cast::<D, OutD>();
        CommitCastTensor::new(self.ctx, self.inner.map(|v| v.map(|v| v.cast())))
    }
}
// ANCHOR_END: commit_cast_impl

// ANCHOR: commit_valid_count_pack_impl
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitValidCountPack,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's valid-count-packing stage (sub-context
    /// only). The count comes from a per-call argument; the trailing
    /// elements are discarded. The packed stream keeps the input
    /// `Time` / `Packet` shape at this skeleton stage.
    // TODO: `_valid_count` is currently discarded. The backend
    // `TuOperationCommitValidCountPack` record does not store it yet.
    #[primitive(TuTensor::commit_valid_count_pack)]
    pub fn commit_valid_count_pack(
        self,
        _valid_count: usize,
    ) -> CommitValidCountPackTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B> {
        verify_commit_valid_count_pack::<D, Time, Packet>();
        CommitValidCountPackTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: commit_valid_count_pack_impl

/// Verifies the Commit Adapter's trimming-stage constraints.
///
/// 1. Output packet must be 8, 16, 24, or 32 bytes (where 32 means no
///    trim).
/// 2. `OutPacket` must be a `# m` → `# n` style resize of the input
///    `Packet` (`n` ≤ `m`).
pub(crate) fn verify_commit_trim<D: Scalar, Packet: M, OutPacket: M>() {
    let out_packet_bytes = D::size_in_bytes_from_length(OutPacket::SIZE);
    assert!(
        COMMIT_VALID_PACKET_SIZES.contains(&out_packet_bytes),
        "commit_trim output packet must be one of {COMMIT_VALID_PACKET_SIZES:?} bytes, got {out_packet_bytes}",
    );

    let out_packet = OutPacket::to_value();
    let expected_packet = Packet::to_value();
    assert!(
        out_packet.is_resize_of(&expected_packet),
        "commit_trim packet mismatch. Expected {expected_packet} or a trimming of it, got {out_packet}",
    );
}

#[allow(clippy::extra_unused_type_parameters)]
fn verify_commit_cast<D: Scalar, OutD: Scalar>() {
    todo!("commit_cast is not yet implemented")
}

#[allow(clippy::extra_unused_type_parameters)]
fn verify_commit_valid_count_pack<D: Scalar, Time: M, Packet: M>() {
    todo!("commit_valid_count_pack is not yet implemented")
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::verify_commit_trim;
    use crate::scalar::bf16;

    mod valid {
        use super::*;

        axes![N = 8];

        #[test]
        fn full_trim() {
            verify_commit_trim::<i8, m![N # 32], m![N]>();
        }

        #[test]
        fn partial_trim() {
            verify_commit_trim::<i8, m![N # 32], m![N # 16]>();
        }

        #[test]
        fn no_trim() {
            verify_commit_trim::<i8, m![N # 32], m![N # 32]>();
        }

        #[test]
        fn bf16() {
            verify_commit_trim::<bf16, m![N # 16], m![N]>();
        }

        #[test]
        fn f32() {
            verify_commit_trim::<f32, m![N # 8], m![N]>();
        }

        #[test]
        fn single_time_step() {
            verify_commit_trim::<i8, m![N # 32], m![N # 8]>();
        }

        #[test]
        fn non_padding_resize() {
            verify_commit_trim::<bf16, m![N # 16], m![N = 4]>();
        }
    }

    mod invalid {
        use super::*;

        axes![N = 8, X = 8];
    }
}
