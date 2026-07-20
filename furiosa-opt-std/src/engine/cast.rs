//! Cast Engine: scalar type conversion on a single flit.
//!
//! The cast engine operates on a single 32-byte flit. The input packet must be
//! exactly one flit (32 bytes). After casting, the output packet is padded back
//! to 32 bytes. Time passes through unchanged.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::cast::Cast;
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplyCast;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// After the cast engine.
#[derive(Debug)]
pub struct PositionCast;

impl Position for PositionCast {}

/// Tensor streamed after the cast engine.
pub type CastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionCast, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    CastTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
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
        CastTensor::new(self.ctx, self.inner.map(|v| v.cast()).transpose(false))
    }
}
// ANCHOR_END: cast_impl

/// Validates the Cast engine via [`furiosa_opt_lower::config_cast`] (one-flit in / recast one-flit out
/// rules documented there).
fn verify_cast<D: Scalar, OutD: Scalar, InPacket: M, OutPacket: M>() {
    furiosa_opt_lower::config_cast(&InPacket::to_value(), &OutPacket::to_value(), D::BITS, OutD::BITS)
        .unwrap_or_else(|message| panic!("{message}"));
}
