//! Packet Reducer: reduce-add within `Packet`.
//!
//! The Outer stage has already multiplied (and widened) the operands; the Packet
//! Reducer only sums along the contracted axes inside `Packet`.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::ContractionCast;
use crate::context::*;
use crate::engine::contraction::ContractPacketTensor;
use crate::engine::contraction::outer::ContractOuterTensor;
use crate::scalar::*;

// ANCHOR: contract_packet_def
impl<
    'l,
    const T: Tu,
    D: Scalar,
    Storage: ContractionCast<Output = D>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractOuterTensor<'l, T, D, Storage, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Spatial reduction within `Packet`: validates the reduce-add along the contracted axes inside
    /// `Packet` that the fused fold at `contract_lane` will perform. `D` is the widened accumulator the
    /// deferred carrier stays keyed on; the DPE input packet is still sized in `Storage` bytes.
    #[primitive(ContractOuterTensor::contract_packet)]
    pub fn contract_packet<OutPacket: M>(
        self,
    ) -> ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, OutPacket, B> {
        verify_contract_packet::<Storage, Packet, OutPacket>();
        // Carry the deferred operands forward unreduced: the fused contraction at `contract_lane`
        // performs this Packet reduction too. This stage only re-types the carrier to `OutPacket`.
        ContractPacketTensor::new(self.ctx, self.inner)
    }
}
// ANCHOR_END: contract_packet_def

/// Validates the Packet Reducer via [`furiosa_opt_lower::config_contract_packet`] (size / power-of-two /
/// contraction rules documented there). The packet size is taken in `Storage` (pre-widen) bytes: the
/// DPE reads storage-width input flits, not accumulator-width.
pub(crate) fn verify_contract_packet<Storage: Scalar, Packet: M, OutPacket: M>() {
    furiosa_opt_lower::config_contract_packet(&Packet::to_value(), &OutPacket::to_value(), Storage::BITS)
        .unwrap_or_else(|message| panic!("{message}"));
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
