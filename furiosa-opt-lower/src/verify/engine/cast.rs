//! Cast engine: a one-flit input recast to `out_bits` and repadded to a one-flit output.

use furiosa_mapping::{Mapping, MappingExt};

use crate::verify::{FLIT_BYTES, length_from_bytes, size_in_bytes};

/// Why a cast is not realizable on the Cast engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CastError {
    /// The input packet is not exactly one flit.
    #[error("Cast input packet must be exactly {FLIT_BYTES} bytes (one flit): {elements} elements = {bytes} bytes")]
    InputNotOneFlit {
        /// Input packet element count.
        elements: usize,
        /// Input packet byte size.
        bytes: usize,
    },
    /// The output packet is not exactly one flit.
    #[error("Cast output packet must be exactly {FLIT_BYTES} bytes (one flit).")]
    OutputNotOneFlit,
    /// The output packet is not the input recast and repadded to one flit.
    #[error("Cast packet mismatch. Expected: {expected}, got: {got}")]
    PacketMismatch {
        /// The recast, repadded input packet.
        expected: Mapping,
        /// The declared output packet.
        got: Mapping,
    },
}

/// The input packet must be exactly one flit; the output packet must be the input recast to `out_bits`
/// and repadded to one flit. `in_bits`/`out_bits` are the input/output element bit-widths.
pub fn config_cast(
    in_packet: &Mapping,
    out_packet: &Mapping,
    in_bits: usize,
    out_bits: usize,
) -> Result<(), CastError> {
    let in_packet_bytes = size_in_bytes(in_bits, in_packet.size());
    if in_packet_bytes != FLIT_BYTES {
        return Err(CastError::InputNotOneFlit {
            elements: in_packet.size(),
            bytes: in_packet_bytes,
        });
    }

    let out_flit_elements = length_from_bytes(out_bits, FLIT_BYTES);
    let expected_packet = in_packet.clone().replace_padding(out_flit_elements).normalize();

    let out_packet_bytes = size_in_bytes(out_bits, out_packet.size());
    if out_packet_bytes != FLIT_BYTES {
        return Err(CastError::OutputNotOneFlit);
    }

    let out_packet = out_packet.normalize();
    if expected_packet != out_packet {
        return Err(CastError::PacketMismatch {
            expected: expected_packet,
            got: out_packet,
        });
    }
    Ok(())
}
