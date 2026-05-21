//! Cast Engine: scalar type conversion on a single flit.
//!
//! The cast engine operates on a single 32-byte flit. The input packet must be
//! exactly one flit (32 bytes). After casting, the output packet is padded back
//! to 32 bytes. Time passes through unchanged.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::cast::Cast;
use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::engine::{CanApplyCast, FLIT_BYTES};
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::tu::{Position, TuTensor};

/// After the cast engine.
#[derive(Debug)]
pub struct PositionCast;

impl Position for PositionCast {}

/// Tensor streamed after the cast engine.
pub type CastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCast, D, Chip, Cluster, Slice, Time, Packet, B>;

// ANCHOR: cast_impl
//
// The Cast Engine accepts only `VeScalar` inputs (hardware constraint), so the
// bound lives on the impl rather than on a wider trait.
impl<'l, const T: Tu, P: CanApplyCast, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Casts each element to type `OutD` and pads the output packet back to one
    /// 32-byte flit.
    #[primitive(TuTensor::cast)]
    pub fn cast<OutD: Scalar, OutPacket: M>(self) -> CastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, OutPacket, B>
    where
        D: Cast<OutD>,
    {
        verify_cast::<D, OutD, Packet, OutPacket>();
        CastTensor::new(self.ctx, self.inner.map(|v| v.map(|v| v.cast())).transpose(false))
    }
}
// ANCHOR_END: cast_impl

/// Validates cast engine constraints.
///
/// Checks:
/// 1. Input packet must be exactly one flit (32 bytes).
/// 2. Output packet must be exactly one flit (32 bytes).
/// 3. The data terms must match (only padding differs).
fn verify_cast<D: Scalar, OutD: Scalar, InPacket: M, OutPacket: M>() {
    // Input packet must be exactly one flit.
    assert_eq!(
        D::size_in_bytes_from_length(InPacket::SIZE),
        FLIT_BYTES,
        "Cast input packet must be exactly {FLIT_BYTES} bytes (one flit): \
         {} elements = {} bytes",
        InPacket::SIZE,
        D::size_in_bytes_from_length(InPacket::SIZE),
    );

    let out_flit_elements = OutD::length_from_bytes(FLIT_BYTES);

    // Cast elements and pad to 32 bytes.
    let in_packet = InPacket::to_value().factorize();
    let expected_packet = in_packet.pad(out_flit_elements).normalize();

    // Output packet must be exactly one flit.
    let out_packet = OutPacket::to_value().factorize();
    assert_eq!(
        OutD::size_in_bytes_from_length(OutPacket::SIZE),
        FLIT_BYTES,
        "Cast output packet must be exactly {FLIT_BYTES} bytes (one flit). \
         Expected: {expected_packet}, got: {out_packet}",
    );
    assert_eq!(
        expected_packet, out_packet,
        "Cast packet mismatch. Expected: {expected_packet}, got: {out_packet}",
    );
}
