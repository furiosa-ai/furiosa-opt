//! Collect Engine: packet normalization to flit-sized chunks.
//!
//! Normalizes a `FetchTensor` or `SwitchTensor` packet to exactly one flit
//! (`FLIT_BYTES`). Pads to flit-aligned boundary, then splits: inner 32 bytes
//! become `Packet2`, outer flit portion is absorbed into `Time2`.
//!
//! Also home to the terminating moves from `CollectTensor`:
//! - `to_trf`: store to the Tensor Register File.
//! - `to_vrf`: store to the Vector Register File.

use abi_stable::std_types::Tuple2;
use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::engine::{CanApplyCollect, CanApplyToTrf, CanApplyToVrf, FLIT_BYTES, align_up, exact_div};
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::memory::{Address, TrfAddress, TrfTensor, VrfTensor};
use crate::tensor::tu::{Position, TuTensor};

/// After the switch engine's collect engine (32-byte packet normalized).
#[derive(Debug)]
pub struct PositionCollect;

impl Position for PositionCollect {}

/// Tensor after collect engine: packet is exactly 32 bytes (one flit).
pub type CollectTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCollect, D, Chip, Cluster, Slice, Time, Packet, B>;

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
    pub fn to_trf<Lane: M, Element: M>(
        self,
        address: TrfAddress,
    ) -> TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B> {
        verify_to_trf::<D, Lane, Time, Packet, Element>(&address);
        TrfTensor::new(self.inner.transpose(false), address)
    }
}
// ANCHOR_END: collect_to_trf

// ANCHOR: collect_to_vrf
impl<'l, const T: Tu, P: CanApplyToVrf, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Stores to the vector register file.
    #[primitive(TuTensor::to_vrf)]
    pub fn to_vrf<Element: M>(self, address: Address) -> VrfTensor<D, Chip, Cluster, Slice, Element, B> {
        VrfTensor::new(self.inner.transpose(false), address)
    }
}
// ANCHOR_END: collect_to_vrf

/// Validates collect engine constraints: normalizes packet to exactly one flit (32 bytes).
///
/// Pads the input packet to flit-aligned boundary, then splits:
/// - Inner 32 bytes → Packet2 (one flit)
/// - Outer flit portion → absorbed into Time2
///
/// For packets already ≤ 32 bytes, only padding is added.
pub(crate) fn verify_collect<D: Scalar, Time: M, Packet: M, Time2: M, Packet2: M>() {
    let in_packet_bytes = D::size_in_bytes_from_length(Packet::SIZE);
    let aligned_bytes = align_up(in_packet_bytes, FLIT_BYTES);
    let flit_elements = D::length_from_bytes(FLIT_BYTES);

    // Output packet must be exactly one flit.
    assert_eq!(
        D::size_in_bytes_from_length(Packet2::SIZE),
        FLIT_BYTES,
        "Collect output packet must be exactly {FLIT_BYTES} bytes (one flit)."
    );

    // Pad input packet to flit-aligned boundary, then split at flit boundary.
    let in_factorized = Packet::to_value().factorize();
    let padded = in_factorized.pad(D::length_from_bytes(aligned_bytes));
    let Tuple2(in_outer, in_flit) = padded.split_at(flit_elements);

    // Output packet = inner flit.
    let expected_packet = in_flit.normalize();
    let out_packet = Packet2::to_value().factorize();
    assert_eq!(
        expected_packet, out_packet,
        "Collect packet mismatch. Expected: {expected_packet}, got: {out_packet}"
    );

    // Time2 = Time × outer flit portion.
    let expected_time = Time::to_value().factorize().mul(in_outer).normalize();
    let out_time = Time2::to_value().factorize();
    assert_eq!(
        expected_time, out_time,
        "Collect time mismatch. Expected: {expected_time}, got: {out_time}"
    );
}

/// Validates `to_trf` constraints: reshapes `[Time, Packet]` into `[Lane, Element]`.
///
/// - `Lane::SIZE` must be 1, 2, 4, or 8.
/// - Total bytes `Lane::SIZE * Element::SIZE * sizeof(D)` must fit in the chosen TRF region.
/// - `Lane::SIZE` must divide `Time::SIZE`, with the outer factors of `Time` equal to `Lane`.
/// - The remaining inner factors of `Time` concatenated with `Packet` must equal `Element`.
pub(crate) fn verify_to_trf<D: Scalar, Lane: M, Time: M, Packet: M, Element: M>(address: &TrfAddress) {
    assert!(
        [1, 2, 4, 8].contains(&Lane::SIZE),
        "Lane::SIZE must be 1, 2, 4, or 8, got {}",
        Lane::SIZE
    );

    // Trf data should fit in the register file.
    let capacity = address.capacity();
    let total_trf_bytes = D::size_in_bytes_from_length(Lane::SIZE * Element::SIZE);
    assert!(
        total_trf_bytes <= capacity,
        "TRF data ({} bytes = {} lanes x {} bytes) exceeds register file capacity ({} bytes for {})",
        total_trf_bytes,
        Lane::SIZE,
        D::size_in_bytes_from_length(Element::SIZE),
        capacity,
        address,
    );

    // [time_outer] = [Lane]
    let time = Time::to_value().factorize();
    let Tuple2(time_outer, time_inner) = time.split_at(exact_div(Time::SIZE, Lane::SIZE).unwrap_or_else(|| {
        panic!(
            "Lane::SIZE ({}) does not divide Time::SIZE ({})",
            Lane::SIZE,
            Time::SIZE
        )
    }));
    let lane = Lane::to_value().factorize();
    assert_eq!(
        time_outer, lane,
        "`to_trf` lane mismatch: time_outer != Lane: {time_outer} != {lane}",
    );

    // [time_inner, Packet] = [Element]
    let expected_element = time_inner.mul(Packet::to_value().factorize()).normalize();
    let element = Element::to_value().factorize();
    assert_eq!(
        expected_element, element,
        "`to_trf` element mismatch: [time_inner, Packet] != Element: {expected_element} != {element}",
    );
}
