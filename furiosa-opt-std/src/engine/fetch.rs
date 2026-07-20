//! Fetch Engine: DM → Tensor Unit stream.
//!
//! `BeginTensor::fetch` runs the Fetch Sequencer and produces a
//! `FetchTensor`. The element-wise transforms layered on top live in the
//! [Fetch Adapter](super::fetch_adapter): `fetch_mask`, `fetch_table_lookup`,
//! and `fetch_cast`. The adapter is optional; a plain `FetchTensor` can
//! flow directly into Switch or Collect.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplyFetch;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

use furiosa_opt_lower::{FetchError, config_fetch};

/// After the Fetch Engine (sequencer), before any adapter stage.
#[derive(Debug)]
pub struct PositionFetch;

impl Position for PositionFetch {}

/// Tensor streamed after the Fetch Engine.
pub type FetchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetch, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        // Packet alignment is deferred on purpose: a fetch packet may be sub-8-byte here and only
        // become 8-byte aligned after `fetch_cast` widens the element type. The final stream packet
        // is checked at the consumer (Switch/Collect) with the post-cast dtype.
    }

    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

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
    ///
    /// Configures per-slice DM reads and produces a `FetchTensor` with the
    /// chosen `OutTime` / `OutPacket`. The element type is unchanged. Type
    /// casts and other adapter transforms are applied by the per-stage
    /// `fetch_mask` / `fetch_table_lookup` / `fetch_cast` methods.
    #[primitive(TuTensor::fetch)]
    pub fn fetch<OutTime: M, OutPacket: M>(self) -> FetchTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_fetch::<Cluster, Slice, Time, Packet, OutTime, OutPacket>();
        FetchTensor::new(self.ctx, self.inner.transpose(true))
    }
}
// ANCHOR_END: fetch_impl

/// Validates the Fetch engine's type-level constraints (Cluster/Slice size, packet alignment), then
/// synthesizes the read via [`config_fetch`], panicking with the [`FetchError`] on the first
/// violation. The resolved descriptors are not consumed yet, so they're explicitly discarded. The DM
/// form is the input `Time ⊗ Packet` (`Time` may be a live axis); the read is the dual of the commit
/// write.
fn verify_fetch<Cluster: M, Slice: M, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    assert!(
        matches!(Cluster::SIZE, 1 | 2),
        "{}",
        FetchError::ClusterSize(Cluster::SIZE)
    );
    assert!(
        matches!(Slice::SIZE, 64 | 128 | 256),
        "{}",
        FetchError::SliceSize(Slice::SIZE)
    );
    // Packet 8-byte alignment is a property of the final stream packet that the
    // downstream engine reads, not of the fetch read: a sub-aligned packet here
    // can be widened by `fetch_cast` into an aligned one. Switch/Collect enforce
    // it on the post-cast packet.
    let _ = config_fetch(
        &Time::to_value(),
        &Packet::to_value(),
        &OutTime::to_value(),
        &OutPacket::to_value(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 8, B = 8, S = 64, X = 8, L = 16];

    // verify_fetch::<D, Cluster, Slice, Time, Packet, OutTime, OutPacket>;
    // The DM form is the input `Time ⊗ Packet`; `(OutTime, OutPacket)` is the requested stream.

    /// The requested stream reads the DM `Packet` form cleanly (identity read).
    #[test]
    fn valid_read() {
        verify_fetch::<m![1], m![S], m![1], m![A, B], m![A], m![B]>();
    }

    /// A transposed read of the same axes is still a valid read.
    #[test]
    fn transposed_read() {
        verify_fetch::<m![1], m![S], m![1], m![A, B], m![B], m![A]>();
    }

    /// A split axis: the packet reads the inner chunk `L % 8` (8 bytes, aligned), the time the outer
    /// `L / 8`. The two stages sequence them separately, so the shared `L` stays split across the
    /// packet and time configs instead of recombining into one straddling entry.
    #[test]
    fn split_axis_across_packet_and_time() {
        verify_fetch::<m![1], m![S], m![1], m![L], m![L / 8], m![L % 8]>();
    }

    /// A broadcast read: `OutTime` is a literal broadcast the DM lacks, replicating the packet
    /// across 4 time steps (stride-0, the same DM address re-read). The broadcast is not a live
    /// axis, so `covered(Read)` still passes, and it consumes one entry of the shared table.
    #[test]
    fn broadcast_time_read() {
        verify_fetch::<m![1], m![S], m![1], m![A], m![4], m![A]>();
    }

    /// A live input `Time` (e.g. an interleave) is a valid read: the DM form is `Time ⊗ Packet`, so
    /// the time descriptor reads `Time` and the packet descriptor reads `Packet`.
    #[test]
    fn live_input_time_reads() {
        verify_fetch::<m![1], m![S], m![A], m![B], m![A], m![B]>();
    }
}
