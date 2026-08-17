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
//! - `commit_cast::<OutD>()` / `commit_cast_relu::<OutD>()` → `CommitCastTensor` (main)
//! - `commit_valid_count_pack(count)` → `CommitValidCountPackTensor` (sub)

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::CommitCast;
use crate::constraints;
use crate::context::*;
use crate::engine::{CanApplyCommitCast, CanApplyCommitTrim, CanApplyCommitValidCountPack};
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// After the Commit Adapter's trimming stage.
#[derive(Debug)]
pub struct PositionCommitTrim;

impl Position for PositionCommitTrim {}

/// Tensor streamed after `commit_trim`.
pub type CommitTrimTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCommitTrim, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    CommitTrimTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_aligned_by_access_width_max_flit::<D, Packet>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

/// After the Commit Adapter's type-casting stage.
#[derive(Debug)]
pub struct PositionCommitCast;

impl Position for PositionCommitCast {}

/// Tensor streamed after `commit_cast` or `commit_cast_relu`.
pub type CommitCastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCommitCast, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    CommitCastTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

/// After the Commit Adapter's valid-count-packing stage.
#[derive(Debug)]
pub struct PositionCommitValidCountPack;

impl Position for PositionCommitValidCountPack {}

/// Tensor streamed after `commit_valid_count_pack`.
pub type CommitValidCountPackTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCommitValidCountPack, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    CommitValidCountPackTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

// ANCHOR: commit_trim_impl
// `D: MaterializableScalar` here (trim is the commit path's mandatory first stage) keeps i5/i9 uncommittable.
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitTrim,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
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
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitCast,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's type-casting stage.
    ///
    /// Folds the `f32` → `bf16` cast into the commit path, leaving the
    /// [Cast Engine](crate::engine::cast) free for sub-context Vector Engine
    /// work. See [`CommitCast`](crate::prelude::CommitCast) for why this is the
    /// only conversion, and [`commit_cast_relu`](Self::commit_cast_relu) for the
    /// variant that fuses a ReLU.
    #[primitive(TuTensor::commit_cast)]
    pub fn commit_cast<OutD: Scalar>(self) -> CommitCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: CommitCast<OutD>,
    {
        verify_commit_cast::<D, OutD, Packet>();
        CommitCastTensor::new(self.ctx, self.inner.map(|v| v.cast()))
    }

    /// The same cast with a ReLU fused in, clamping negative values to zero.
    ///
    /// A separate method rather than an argument because the two are separate
    /// hardware conversions (`CommitF32ToBf16` and `CommitF32ToBf16Relu`), and
    /// ReLU has no standalone stage to select at run time.
    #[primitive(TuTensor::commit_cast_relu)]
    pub fn commit_cast_relu<OutD: Scalar>(self) -> CommitCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: CommitCast<OutD>,
    {
        verify_commit_cast::<D, OutD, Packet>();
        CommitCastTensor::new(self.ctx, self.inner.map(|v| v.cast_relu()))
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

/// Validates the Commit Adapter's trimming stage via [`furiosa_opt_lower::config_commit_trim`]
/// (width / resize rules documented there).
pub(crate) fn verify_commit_trim<D: Scalar, Packet: M, OutPacket: M>() {
    furiosa_opt_lower::config_commit_trim(&Packet::to_value(), &OutPacket::to_value(), D::BITS)
        .unwrap_or_else(|message| panic!("{message}"));
}

/// Validates the Commit Adapter's cast via [`furiosa_opt_lower::config_commit_cast`]. `Packet` is
/// the post-`commit_trim` packet, whose width in the pre-cast type `D` is the commit input width.
fn verify_commit_cast<D: Scalar, OutD: Scalar, Packet: M>() {
    furiosa_opt_lower::config_commit_cast(&Packet::to_value(), D::BITS, OutD::BITS)
        .unwrap_or_else(|message| panic!("{message}"));
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
