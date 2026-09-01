//! Packet Reducer: reduce-add within `Packet`.
//!
//! The Outer stage has already multiplied (and widened) the operands; the Packet
//! Reducer only sums along the contracted axes inside `Packet`.

use furiosa_mapping::*;
use furiosa_opt_lower::{ContractPacketInput, config_contract_packet};
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
        verify_contract_packet(ContractPacketInput {
            in_packet: Packet::to_value(),
            out_packet: OutPacket::to_value(),
            element_bits: Storage::BITS,
        });
        // Carry the deferred operands forward unreduced: the fused contraction at `contract_lane`
        // performs this Packet reduction too. This stage only re-types the carrier to `OutPacket`.
        ContractPacketTensor::new(self.ctx, self.inner)
    }
}
// ANCHOR_END: contract_packet_def

/// Validates the Packet Reducer via [`furiosa_opt_lower::config_contract_packet`] (size / power-of-two /
/// contraction rules documented there). The packet size is taken in `Storage` (pre-widen) bytes: the
/// DPE reads storage-width input flits, not accumulator-width.
pub(crate) fn verify_contract_packet(input: ContractPacketInput) {
    config_contract_packet(input).unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bf16;
    use furiosa_mapping::M as _;
    use furiosa_opt_lower::ContractPacketError;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    #[test]
    fn valid_full_reduction() {
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![K]>::to_value(),
            out_packet: <m![1]>::to_value(),
            element_bits: <i8 as Scalar>::BITS,
        });
    }

    #[test]
    fn valid_partial_reduction() {
        // K % 4 reduced
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![K]>::to_value(),
            out_packet: <m![K / 4]>::to_value(),
            element_bits: <i8 as Scalar>::BITS,
        });
    }
    #[test]
    fn invalid_retained_packet_size() {
        assert!(matches!(
            config_contract_packet(ContractPacketInput {
                in_packet: <m![K]>::to_value(),
                out_packet: <m![D]>::to_value(),
                element_bits: <i8 as Scalar>::BITS,
            }),
            Err(ContractPacketError::NotAContraction { .. })
        ));
    }

    #[test]
    fn invalid_no_reduction() {
        // Temporal accumulator only has 32 columns, cannot fit 64 packet
        assert_eq!(
            config_contract_packet(ContractPacketInput {
                in_packet: <m![K]>::to_value(),
                out_packet: <m![K]>::to_value(),
                element_bits: <i8 as Scalar>::BITS,
            }),
            Err(ContractPacketError::OutPacketTooWide(K::SIZE))
        );
    }

    #[test]
    fn valid_partial_reduction_multi_axis() {
        // `D / 2 % 4` is reduced, retained_packet is `[A, D / 8]`.
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![A, D / 2]>::to_value(),
            out_packet: <m![A, D / 8]>::to_value(),
            element_bits: <i8 as Scalar>::BITS,
        });
    }

    #[test]
    fn valid_padded_packet_inner_reduction() {
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![A # 16, C]>::to_value(),
            out_packet: <m![A]>::to_value(),
            element_bits: <i8 as Scalar>::BITS,
        });
    }

    #[test]
    fn valid_padded_packet_inner_reduction_with_padding() {
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![A # 16, C]>::to_value(),
            out_packet: <m![A # 16]>::to_value(),
            element_bits: <i8 as Scalar>::BITS,
        });
    }

    #[test]
    fn valid_padded_packet_split() {
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![B # 8, N]>::to_value(),
            out_packet: <m![B]>::to_value(),
            element_bits: <i8 as Scalar>::BITS,
        });
    }

    #[test]
    fn valid_no_spatial_reduction_bf16() {
        // Tree depth 0: all 32 bf16 elements pass through, no reduction.
        verify_contract_packet(ContractPacketInput {
            in_packet: <m![D]>::to_value(),
            out_packet: <m![D]>::to_value(),
            element_bits: <bf16 as Scalar>::BITS,
        });
    }
    #[test]
    fn flit_check_sizes_in_storage_not_accumulator_width() {
        // The flit check must size the packet in the pre-widen `Storage` dtype, not the widened
        // accumulator `D` the result tensor carries. `D = 32` elements is a valid 64-byte packet at
        // `bf16` storage (`valid_no_spatial_reduction_bf16` above), but 128 bytes at `f32` accumulator
        // width. The pipeline threads `Storage` (here `bf16`), never `D` (`f32`), into this check; had
        // it threaded the accumulator, this valid packet would be rejected as below.
        assert_eq!(
            config_contract_packet(ContractPacketInput {
                in_packet: <m![D]>::to_value(),
                out_packet: <m![D]>::to_value(),
                element_bits: <f32 as Scalar>::BITS,
            }),
            Err(ContractPacketError::InvalidPacketSize(128))
        );
    }
    #[test]
    fn invalid_non_power_of_two_out_packet() {
        assert_eq!(
            config_contract_packet(ContractPacketInput {
                in_packet: <m![K]>::to_value(),
                out_packet: <m![K = 3]>::to_value(),
                element_bits: <i8 as Scalar>::BITS,
            }),
            Err(ContractPacketError::OutPacketNotPow2(3))
        );
    }

    #[test]
    fn invalid_partial_inner_packet() {
        assert!(matches!(
            config_contract_packet(ContractPacketInput {
                in_packet: <m![K]>::to_value(),
                out_packet: <m![K % 4]>::to_value(),
                element_bits: <i8 as Scalar>::BITS,
            }),
            Err(ContractPacketError::NotAContraction { .. })
        ));
    }
}
