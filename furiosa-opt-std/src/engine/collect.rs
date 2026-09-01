//! Collect Engine: packet normalization to flit-sized chunks.
//!
//! Normalizes a `FetchTensor` or `SwitchTensor` packet to exactly one flit
//! (`FLIT_BYTES`). Pads to flit-aligned boundary, then splits: inner 32 bytes
//! become `Packet2`, outer flit portion is absorbed into `Time2`.
//!
//! Also home to the terminating moves from `CollectTensor`:
//! - `to_trf`: store to the Tensor Register File.
//! - `to_vrf`: store to the Vector Register File.

use furiosa_mapping::*;
use furiosa_opt_lower::{
    CollectInput, ToTrfError, ToTrfInput, ToVrfInput, config_collect, config_to_trf, config_to_vrf,
};
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::engine::{CanApplyCollect, CanApplyToTrf, CanApplyToVrf};
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::memory::{TrfTensor, VrfTensor};
use crate::tensor::tu::{Position, TuTensor};

/// After the switch engine's collect engine (32-byte packet normalized).
#[derive(Debug)]
pub struct PositionCollect;

impl Position for PositionCollect {}

/// Tensor after collect engine: packet is exactly 32 bytes (one flit).
pub type CollectTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCollect, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    CollectTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_one_flit::<D, Packet>();
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

// ANCHOR: collect_impl
impl<'l, const T: Tu, P: CanApplyCollect, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Normalizes packet to exactly 32 bytes (one flit).
    ///
    /// Pads to flit-aligned boundary, then splits: inner 32 bytes become
    /// `Packet2`, outer flit portion is absorbed into `Time2`. For packets
    /// already ≤ 32 bytes, only padding is added.
    #[primitive(TuTensor::collect)]
    pub fn collect<Time2: M, Packet2: M>(self) -> CollectTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2, B> {
        verify_collect::<D, Time, Packet, Time2, Packet2>();
        CollectTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: collect_impl

// ANCHOR: collect_to_trf
impl<'l, const T: Tu, P: CanApplyToTrf, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Stores to the tensor register file.
    #[primitive(TuTensor::to_trf)]
    pub fn to_trf<Lane: M, Element: M>(self) -> TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B> {
        verify_to_trf::<D, Lane, Time, Packet, Element>();
        TrfTensor::new(self.inner.transpose(false))
    }
}
// ANCHOR_END: collect_to_trf

// ANCHOR: collect_to_vrf
impl<'l, P: CanApplyToVrf, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, { Tu::Sub }, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Stores to the vector register file.
    #[primitive(TuTensor::to_vrf)]
    pub fn to_vrf<Element: M>(self) -> VrfTensor<D, Chip, Cluster, Slice, Element, B> {
        verify_to_vrf::<D, Time, Packet, Element>();
        VrfTensor::new(self.inner.transpose(false))
    }
}

impl<'l, P: CanApplyToVrf, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, { Tu::Main }, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Stores to the vector register file, occupying the sub context the write runs through.
    ///
    /// The borrow lasts the statement. How long the *register* stays occupied is the compiler's
    /// allocation to make, so the returned [`VrfTensor`] keeps no borrow.
    #[primitive(TuTensor::to_vrf_from_main)]
    pub fn to_vrf<Element: M>(
        self,
        _sub: &mut TuContext<{ Tu::Sub }>,
    ) -> VrfTensor<D, Chip, Cluster, Slice, Element, B> {
        verify_to_vrf::<D, Time, Packet, Element>();
        VrfTensor::new(self.inner.transpose(false))
    }
}
// ANCHOR_END: collect_to_vrf

/// Validates the Collect engine via [`furiosa_opt_lower::config_collect`] (packet / time rules
/// documented there).
pub(crate) fn verify_collect<D: Scalar, Time: M, Packet: M, Time2: M, Packet2: M>() {
    config_collect(CollectInput {
        in_time: Time::to_value(),
        in_packet: Packet::to_value(),
        out_time: Time2::to_value(),
        out_packet: Packet2::to_value(),
        element_bits: D::BITS,
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

/// Validates `to_vrf` via [`furiosa_opt_lower::config_to_vrf`] (element rule documented there).
pub(crate) fn verify_to_vrf<D: Scalar, Time: M, Packet: M, Element: M>() {
    config_to_vrf(ToVrfInput {
        stream_time: Time::to_value(),
        stream_packet: Packet::to_value(),
        vrf_element: Element::to_value(),
        element_bits: D::BITS,
    })
    .unwrap_or_else(|error| panic!("{error}"));
}

/// Total TRF capacity in bytes: 8 lanes x 2 banks x 128 rows x 32 bytes. Where a tensor lands in the
/// register file is the compiler's to decide, so this is the only capacity the frontend checks.
pub const TRF_CAPACITY_BYTES: usize = 65_536;

/// Validates `to_trf` via [`furiosa_opt_lower::config_to_trf`] (lane / capacity / element rules
/// documented there).
pub(crate) fn verify_to_trf<D: Scalar, Lane: M, Time: M, Packet: M, Element: M>() {
    config_to_trf(ToTrfInput {
        stream_lane: Lane::to_value(),
        stream_time: Time::to_value(),
        stream_packet: Packet::to_value(),
        trf_element: Element::to_value(),
        capacity: TRF_CAPACITY_BYTES,
        element_bits: D::BITS,
    })
    .unwrap_or_else(|error| match error {
        ToTrfError::ExceedsCapacity {
            total_bytes,
            lanes,
            per_lane_bytes,
            capacity,
        } => panic!(
            "TRF data ({total_bytes} bytes = {lanes} lanes x {per_lane_bytes} bytes) \
             exceeds the tensor register file capacity ({capacity} bytes)"
        ),
        other => panic!("{other}"),
    });
}
