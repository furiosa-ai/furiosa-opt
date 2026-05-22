//! Packet Reducer: reduce-add within `Packet`.
//!
//! The Outer stage has already multiplied (and widened) the operands; the Packet
//! Reducer only sums along the contracted axes inside `Packet`.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::cast::ContractionCast;
use crate::context::*;
use crate::engine::contraction::outer::ContractOuterTensor;
use crate::engine::contraction::{ContractPacketTensor, TEMPORAL_ACCUMULATOR_COLS};
use crate::runtime::Backend;
use crate::scalar::*;

// ANCHOR: contract_packet_def
impl<
    'l,
    const T: Tu,
    D: Scalar + ContractionCast,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractOuterTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Spatial reduction within `Packet`: reduce-add along the contracted axes inside `Packet`
    /// of the product already produced by the Outer stage.
    /// The product type is the widened contraction output type: `i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`.
    #[primitive(ContractOuterTensor::contract_packet)]
    pub fn contract_packet<OutPacket: M>(
        self,
    ) -> ContractPacketTensor<'l, T, <D as ContractionCast>::Output, Chip, Cluster, Slice, Lane, Time, OutPacket, B>
    {
        verify_contract_packet::<D, Packet, OutPacket>();
        ContractPacketTensor {
            ctx: self.ctx,
            inner: self.inner.reduce_add(),
        }
    }
}
// ANCHOR_END: contract_packet_def

/// Validates the Packet Reducer.
///
/// Checks:
/// 1. `Packet` should be 32 or 64 bytes (mirroring the Outer stage's `OutPacket`).
/// 2. `OutPacket::SIZE` should be a power of two and at most
///    `TEMPORAL_ACCUMULATOR_COLS` and be obtainable from `Packet` by splitting
///    at a power-of-two sized boundary.
pub(crate) fn verify_contract_packet<D: Scalar, Packet: M, OutPacket: M>() {
    let packet_size = D::size_in_bytes_from_length(Packet::SIZE);
    assert!(
        [32, 64].contains(&packet_size),
        "Packet must be 32 or 64 bytes, got {packet_size} bytes"
    );

    assert!(
        OutPacket::SIZE <= TEMPORAL_ACCUMULATOR_COLS,
        "OutPacket::SIZE must be at most {TEMPORAL_ACCUMULATOR_COLS}, got {}",
        OutPacket::SIZE
    );

    assert!(
        OutPacket::SIZE.is_power_of_two(),
        "OutPacket::SIZE must be a power of two, got {}",
        OutPacket::SIZE
    );

    let packet = Packet::to_value().factorize();
    let out_packet = OutPacket::to_value().factorize().remove_padding();

    assert!(
        (0..=Packet::SIZE.trailing_zeros()).rev().any(|depth| {
            let split = 1 << depth;
            let outer = packet.clone().stride(split);
            outer.remove_padding() == out_packet.clone()
        }),
        "OutPacket {out_packet} is not a valid contraction of Packet {packet}",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bf16;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    #[test]
    fn valid_full_reduction() {
        verify_contract_packet::<i8, m![K], m![1]>();
    }

    #[test]
    fn valid_partial_reduction() {
        // K % 4 reduced
        verify_contract_packet::<i8, m![K], m![K / 4]>();
    }

    #[test]
    fn valid_partial_reduction_multi_axis() {
        // `D / 2 % 4` is reduced, retained_packet is `[A, D / 8]`.
        verify_contract_packet::<i8, m![A, D / 2], m![A, D / 8]>();
    }

    #[test]
    fn valid_padded_packet_inner_reduction() {
        verify_contract_packet::<i8, m![A # 16, C], m![A]>();
    }

    #[test]
    fn valid_padded_packet_inner_reduction_with_padding() {
        verify_contract_packet::<i8, m![A # 16, C], m![A # 16]>();
    }

    #[test]
    fn valid_padded_packet_split() {
        verify_contract_packet::<i8, m![B # 8, N], m![B]>();
    }

    #[test]
    fn valid_no_spatial_reduction_bf16() {
        // Tree depth 0: all 32 bf16 elements pass through, no reduction.
        verify_contract_packet::<bf16, m![D], m![D]>();
    }
}
