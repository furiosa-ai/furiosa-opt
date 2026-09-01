//! Fetch Engine: DM → Tensor Unit stream.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::{CanApplyFetch, CanApplyFetchChipLift, CanApplyFetchClusterLift, CanApplyFetchSliceLift};
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

use furiosa_opt_lower::{
    FetchDimensionsInput, FetchInput, FetchLiftDimension, FetchLiftInput, config_fetch, config_fetch_dimensions,
    config_fetch_lift,
};

/// After the Fetch Engine (sequencer), before any adapter stage.
#[derive(Debug)]
pub struct PositionFetch;

impl Position for PositionFetch {}

/// Tensor streamed after the Fetch Engine.
pub type FetchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetch, D, Chip, Cluster, Slice, Time, Packet, B>;

/// After a lift onto `Chip`; a lift onto `Cluster` or `Slice` may still follow.
#[derive(Debug)]
pub struct PositionFetchChipLift;

impl Position for PositionFetchChipLift {}

/// After a lift onto `Cluster`; a lift onto `Slice` may still follow.
#[derive(Debug)]
pub struct PositionFetchClusterLift;

impl Position for PositionFetchClusterLift {}

/// After a lift onto `Slice`, the innermost dimension a lift can address.
#[derive(Debug)]
pub struct PositionFetchSliceLift;

impl Position for PositionFetchSliceLift {}

/// Tensor streamed after a lift onto `Chip`.
pub type FetchChipLiftTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchChipLift, D, Chip, Cluster, Slice, Time, Packet, B>;

/// Tensor streamed after a lift onto `Cluster`.
pub type FetchClusterLiftTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchClusterLift, D, Chip, Cluster, Slice, Time, Packet, B>;

/// Tensor streamed after a lift onto `Slice`.
pub type FetchSliceLiftTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchSliceLift, D, Chip, Cluster, Slice, Time, Packet, B>;

fn new_fetch_lift_tensor<
    'l,
    const T: Tu,
    P: Position,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
>(
    ctx: &'l mut TuContext<{ T }>,
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Packet>>>>, B>,
) -> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B> {
    TuTensor {
        ctx,
        inner,
        _position: PhantomData,
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchChipLiftTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        new_fetch_lift_tensor(ctx, inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchClusterLiftTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        new_fetch_lift_tensor(ctx, inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchSliceLiftTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        new_fetch_lift_tensor(ctx, inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

// ANCHOR: fetch_impl
impl<'l, const T: Tu, P: CanApplyFetch, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Sequencer.
    #[primitive(TuTensor::fetch)]
    pub fn fetch<OutTime: M, OutPacket: M>(self) -> FetchTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_fetch::<Cluster, Slice, Time, Packet, OutTime, OutPacket>();
        FetchTensor::new(self.ctx, self.inner.transpose(true))
    }
}
// ANCHOR_END: fetch_impl

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    // ANCHOR: fetch_chip_lift
    /// Gives each chip its own fetch base, lifting a DM axis onto `Chip`.
    #[primitive(TuTensor::fetch_chip_lift)]
    pub fn fetch_chip_lift<OutChip: M, OutTime: M>(
        self,
    ) -> FetchChipLiftTensor<'l, T, D, OutChip, Cluster, Slice, OutTime, Packet, B>
    where
        P: CanApplyFetchChipLift,
    {
        // ANCHOR_END: fetch_chip_lift
        constraints::assert_chip_preserved::<Chip, OutChip>();
        constraints::assert_lift_factor::<Chip, Time, OutTime>();
        verify_fetch_lift::<Chip, OutChip, Time, Packet, OutTime>(FetchLiftDimension::Chip);
        FetchChipLiftTensor::new(self.ctx, self.inner.transpose(true))
    }
}

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    // ANCHOR: fetch_cluster_lift
    /// Gives each cluster its own fetch base, lifting a DM axis onto `Cluster`.
    #[primitive(TuTensor::fetch_cluster_lift)]
    pub fn fetch_cluster_lift<OutCluster: M, OutTime: M>(
        self,
    ) -> FetchClusterLiftTensor<'l, T, D, Chip, OutCluster, Slice, OutTime, Packet, B>
    where
        P: CanApplyFetchClusterLift,
    {
        // ANCHOR_END: fetch_cluster_lift
        constraints::assert_cluster_preserved::<Cluster, OutCluster>();
        constraints::assert_lift_factor::<Cluster, Time, OutTime>();
        verify_fetch_lift::<Cluster, OutCluster, Time, Packet, OutTime>(FetchLiftDimension::Cluster);
        FetchClusterLiftTensor::new(self.ctx, self.inner.transpose(true))
    }
}

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    // ANCHOR: fetch_slice_lift
    /// Gives each slice its own fetch base, lifting a DM axis onto `Slice`.
    #[primitive(TuTensor::fetch_slice_lift)]
    pub fn fetch_slice_lift<OutSlice: M, OutTime: M>(
        self,
    ) -> FetchSliceLiftTensor<'l, T, D, Chip, Cluster, OutSlice, OutTime, Packet, B>
    where
        P: CanApplyFetchSliceLift,
    {
        // ANCHOR_END: fetch_slice_lift
        constraints::assert_slice_preserved::<Slice, OutSlice>();
        constraints::assert_lift_factor::<Slice, Time, OutTime>();
        verify_fetch_lift::<Slice, OutSlice, Time, Packet, OutTime>(FetchLiftDimension::Slice);
        FetchSliceLiftTensor::new(self.ctx, self.inner.transpose(true))
    }
}

/// Validates the Fetch dimensions and synthesizes its read descriptors.
fn verify_fetch<Cluster: M, Slice: M, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    config_fetch_dimensions(FetchDimensionsInput {
        cluster_size: Cluster::SIZE,
        slice_size: Slice::SIZE,
    })
    .unwrap_or_else(|e| panic!("{e}"));
    config_fetch(FetchInput {
        in_time: Time::to_value(),
        in_packet: Packet::to_value(),
        out_time: OutTime::to_value(),
        out_packet: OutPacket::to_value(),
        lifted: None,
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

/// Validates one placement lift and its `(time, packet, base)` descriptors.
fn verify_fetch_lift<Placement: M, OutPlacement: M, Time: M, Packet: M, OutTime: M>(dimension: FetchLiftDimension) {
    let lifted = config_fetch_lift(FetchLiftInput {
        dimension,
        in_placement: Placement::to_value(),
        out_placement: OutPlacement::to_value(),
    })
    .unwrap_or_else(|e| panic!("{e}"));
    // The packet a lift hands on is the one it was given: only `OutTime` loses the lifted axes.
    let packet = Packet::to_value();
    let _ = config_fetch(FetchInput {
        in_time: Time::to_value(),
        in_packet: packet.clone(),
        out_time: OutTime::to_value(),
        out_packet: packet,
        lifted: Some(lifted),
    })
    .unwrap_or_else(|e| panic!("{dimension} lift: {e}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_lower::{FetchError, FetchLiftError};

    axes![A = 8, B = 8, S = 64, X = 8, L = 16];

    // verify_fetch::<D, Cluster, Slice, Time, Packet, OutTime, OutPacket>; The DM form is the input
    // `Time ⊗ Packet`; `(OutTime, OutPacket)` is the requested stream.

    #[test]
    fn valid_read() {
        verify_fetch::<m![1], m![S], m![1], m![A, B], m![A], m![B]>();
    }
    /// The packet must be the DM's own innermost axis: asking for the outer one steps the DM per
    /// element, which is below the read the hardware performs.
    #[test]
    fn transposed_read_rejects() {
        assert_eq!(
            config_fetch(FetchInput {
                in_time: <m![1]>::to_value(),
                in_packet: <m![A, B]>::to_value(),
                out_time: <m![B]>::to_value(),
                out_packet: <m![A]>::to_value(),
                lifted: None,
            }),
            Err(FetchError::NonContiguousPacket {
                innermost: <m![A]>::to_value(),
                memory_stride: B::SIZE,
            })
        );
    }

    #[test]
    fn innermost_packet_broadcast_rejects() {
        assert_eq!(
            config_fetch(FetchInput {
                in_time: <m![1]>::to_value(),
                in_packet: <m![B]>::to_value(),
                out_time: <m![1]>::to_value(),
                out_packet: <m![B, X]>::to_value(),
                lifted: None,
            }),
            Err(FetchError::NonContiguousPacket {
                innermost: <m![X]>::to_value(),
                memory_stride: 0,
            })
        );
    }

    /// Splits `L`: packet reads `L % 8`, while time walks `L / 8`.
    #[test]
    fn split_axis_across_packet_and_time() {
        verify_fetch::<m![1], m![S], m![1], m![L], m![L / 8], m![L % 8]>();
    }

    /// Replicates the packet across four stride-zero time steps.
    #[test]
    fn broadcast_time_read() {
        verify_fetch::<m![1], m![S], m![1], m![A], m![4], m![A]>();
    }

    #[test]
    fn live_input_time_reads() {
        verify_fetch::<m![1], m![S], m![A], m![B], m![A], m![B]>();
    }
    /// Rejects a stream that leaves the DM's live `B` axis unread.
    #[test]
    fn unreadable_axis_rejects() {
        assert!(matches!(
            config_fetch(FetchInput {
                in_time: <m![1]>::to_value(),
                in_packet: <m![A, B]>::to_value(),
                out_time: <m![A]>::to_value(),
                out_packet: <m![X]>::to_value(),
                lifted: None,
            }),
            Err(FetchError::Unread { .. })
        ));
    }

    /// Cluster size must be 1 or 2.
    #[test]
    fn cluster_size_rejects() {
        assert_eq!(
            config_fetch_dimensions(FetchDimensionsInput {
                cluster_size: 3,
                slice_size: S::SIZE,
            }),
            Err(FetchError::ClusterSize(3))
        );
    }

    /// Slice size must be one of 64/128/256.
    #[test]
    fn slice_size_rejects() {
        assert_eq!(
            config_fetch_dimensions(FetchDimensionsInput {
                cluster_size: 1,
                slice_size: 100,
            }),
            Err(FetchError::SliceSize(100))
        );
    }

    #[test]
    fn verify_fetch_checks_dimensions() {
        assert!(
            std::panic::catch_unwind(|| { verify_fetch::<m![3], m![S], m![1], m![A, B], m![A], m![B]>() }).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { verify_fetch::<m![1], m![100], m![1], m![A, B], m![A], m![B]>() }).is_err()
        );
    }

    /// Accepts a 4-byte packet because adapters may widen it before final alignment.
    #[test]
    fn unaligned_output_packet_allowed() {
        verify_fetch::<m![1], m![S], m![1], m![L / 4], m![1], m![L / 4]>();
    }
    // verify_fetch_lift::<Placement, OutPlacement, Time, Packet, OutTime>; A lift stage moves axes
    // out of its own `Time` and into `OutPlacement`; the packet is handed on.

    #[test]
    fn valid_lift() {
        verify_fetch_lift::<m![S, 2], m![S, A / 4], m![A], m![B], m![A % 4]>(FetchLiftDimension::Slice);
    }

    #[test]
    fn valid_two_broadcast_lift() {
        verify_fetch_lift::<m![2, S, 2], m![A / 4, S, B / 4], m![A / 4, B / 4], m![X], m![1]>(
            FetchLiftDimension::Slice,
        );
    }
    /// Rejects renaming a live placement axis; only broadcasts may be filled by a lift.
    #[test]
    fn relabelled_placement_rejects() {
        assert!(matches!(
            config_fetch_lift(FetchLiftInput {
                dimension: FetchLiftDimension::Slice,
                in_placement: <m![S, A / 4]>::to_value(),
                out_placement: <m![S, B / 4]>::to_value(),
            }),
            Err(FetchLiftError::PlacementMismatch { .. })
        ));
    }

    /// A placement the stream repeats verbatim lifts nothing; that read is `fetch` alone.
    #[test]
    fn unchanged_placement_rejects() {
        assert_eq!(
            config_fetch_lift(FetchLiftInput {
                dimension: FetchLiftDimension::Slice,
                in_placement: <m![S, 2]>::to_value(),
                out_placement: <m![S, 2]>::to_value(),
            }),
            Err(FetchLiftError::NoAxisLifted {
                dimension: FetchLiftDimension::Slice,
            })
        );
    }
}
