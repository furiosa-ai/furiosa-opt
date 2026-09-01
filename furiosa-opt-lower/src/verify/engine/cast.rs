//! Cast engine: a one-flit input recast to `out_bits` and repadded to a one-flit output.

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};

use crate::verify::{
    ElementSizeError, FLIT_BYTES, OneFlitPacketError, PacketSide, length_from_bytes, require_one_flit, size_in_bytes,
};

/// A conversion supported by the Cast engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    I32ToI4,
    I32ToI8,
    I32ToI16,
    F32ToF8E4M3,
    F32ToF8E5M2,
    F32ToBf16,
}

/// Why a cast is not realizable on the Cast engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CastError {
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
    /// The input or output packet is not exactly one flit.
    #[error("Cast {0}")]
    PacketSize(#[from] OneFlitPacketError),
    /// The output packet is not the input recast and repadded to one flit.
    #[error("Cast packet mismatch. Expected: {expected}, got: {got}")]
    PacketMismatch {
        /// The recast, repadded input packet.
        expected: Mapping,
        /// The declared output packet.
        got: Mapping,
    },
}

/// Inputs used to configure the Cast engine.
pub struct CastInput {
    pub in_packet: Mapping,
    pub out_packet: Mapping,
    pub kind: CastKind,
}

impl CastKind {
    fn widths(self) -> (usize, usize) {
        match self {
            Self::I32ToI4 => (32, 4),
            Self::I32ToI8 => (32, 8),
            Self::I32ToI16 | Self::F32ToBf16 => (32, 16),
            Self::F32ToF8E4M3 | Self::F32ToF8E5M2 => (32, 8),
        }
    }
}

/// Checks the one-flit input and output mappings for a supported Cast conversion.
pub fn config_cast(input: CastInput) -> Result<(), CastError> {
    let CastInput {
        in_packet,
        out_packet,
        kind,
    } = input;
    let (in_bits, out_bits) = kind.widths();
    let in_packet_bytes = size_in_bytes(in_bits, in_packet.size())?;
    require_one_flit(PacketSide::Input, in_packet.size(), in_packet_bytes)?;

    let out_flit_elements = length_from_bytes(out_bits, FLIT_BYTES)?;
    let expected_packet = in_packet.padding(out_flit_elements, PaddingKind::Top).normalize();

    let out_packet_bytes = size_in_bytes(out_bits, out_packet.size())?;
    require_one_flit(PacketSide::Output, out_packet.size(), out_packet_bytes)?;

    let out_packet = out_packet.normalize();
    if expected_packet != out_packet {
        return Err(CastError::PacketMismatch {
            expected: expected_packet,
            got: out_packet,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![Packet = 8];

    #[test]
    fn accepts_a_supported_conversion() {
        assert_eq!(
            config_cast(CastInput {
                in_packet: <m![Packet]>::to_value(),
                out_packet: <m![Packet # 64]>::to_value(),
                kind: CastKind::I32ToI4,
            }),
            Ok(())
        );
    }

    #[test]
    fn reports_the_output_element_and_byte_counts() {
        assert_eq!(
            config_cast(CastInput {
                in_packet: <m![Packet]>::to_value(),
                out_packet: <m![Packet]>::to_value(),
                kind: CastKind::F32ToBf16,
            }),
            Err(CastError::PacketSize(OneFlitPacketError {
                side: PacketSide::Output,
                elements: 8,
                bytes: 16,
            }))
        );
    }
}
