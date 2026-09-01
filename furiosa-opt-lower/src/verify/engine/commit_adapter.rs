//! Commit trim and commit cast: the widths one Commit Adapter write may use.

use furiosa_mapping::{Mapping, MappingExt};
use furiosa_opt_lower_types::{COMMIT_BASE_SIZE, COMMIT_VALID_PACKET_SIZES};

use crate::verify::{ElementSizeError, size_in_bytes};

/// Why a commit trim is not realizable on the Commit Adapter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommitTrimError {
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
    /// The output packet is not one of the hardware commit widths.
    #[error("commit_trim output packet must be one of {COMMIT_VALID_PACKET_SIZES:?} bytes, got {0}")]
    InvalidWidth(usize),
    /// The output packet is not a resize (trimming) of the input.
    #[error("commit_trim packet mismatch. Expected {in_packet} or a trimming of it, got {out_packet}")]
    PacketMismatch { in_packet: Mapping, out_packet: Mapping },
}

/// Inputs used to configure a commit trim.
pub struct CommitTrimInput {
    pub in_packet: Mapping,
    pub out_packet: Mapping,
    pub element_bits: usize,
}

/// The conversion performed by the Commit Adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitCastKind {
    F32ToBf16,
}

/// The post-trim packet is not a width a converting commit can write.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommitCastError {
    /// The element width cannot describe a whole byte sequence.
    #[error(transparent)]
    ElementSize(#[from] ElementSizeError),
    /// The post-trim packet is not a supported width for the conversion.
    #[error("commit_cast input packet is {commit_in_size} B, not one of {legal:?} B for {unit} B conversion units")]
    InvalidWidth {
        /// Post-trim packet width, in pre-cast bytes.
        commit_in_size: usize,
        /// Bytes covered by one conversion unit.
        unit: usize,
        /// Hardware commit widths this conversion admits.
        legal: Vec<usize>,
    },
}

/// Inputs used to configure a commit cast.
pub struct CommitCastInput {
    pub in_packet: Mapping,
    pub kind: CommitCastKind,
}

impl CommitCastKind {
    fn in_bits(self) -> usize {
        match self {
            Self::F32ToBf16 => 32,
        }
    }

    fn out_bits(self) -> usize {
        match self {
            Self::F32ToBf16 => 16,
        }
    }
}

/// The output packet must be a supported hardware width and a resize of the input.
pub fn config_commit_trim(input: CommitTrimInput) -> Result<(), CommitTrimError> {
    let out_packet_bytes = size_in_bytes(input.element_bits, input.out_packet.size())?;
    if !COMMIT_VALID_PACKET_SIZES.contains(&out_packet_bytes) {
        return Err(CommitTrimError::InvalidWidth(out_packet_bytes));
    }
    if !input.out_packet.is_resize_of(&input.in_packet) {
        return Err(CommitTrimError::PacketMismatch {
            in_packet: input.in_packet,
            out_packet: input.out_packet,
        });
    }
    Ok(())
}

/// Checks that a commit cast's input packet has a supported hardware width.
pub fn config_commit_cast(input: CommitCastInput) -> Result<(), CommitCastError> {
    let unit = COMMIT_BASE_SIZE * (input.kind.in_bits() / input.kind.out_bits());
    let commit_in_size = size_in_bytes(input.kind.in_bits(), input.in_packet.size())?;
    let legal: Vec<_> = COMMIT_VALID_PACKET_SIZES
        .into_iter()
        .filter(|size| size.is_multiple_of(unit))
        .collect();
    if legal.contains(&commit_in_size) {
        return Ok(());
    }
    Err(CommitCastError::InvalidWidth {
        commit_in_size,
        unit,
        legal,
    })
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![One = 1, Six = 6, Eight = 8];

    #[test]
    fn commit_trim_rejects_a_partial_byte() {
        assert_eq!(
            config_commit_trim(CommitTrimInput {
                in_packet: <m![One]>::to_value(),
                out_packet: <m![One]>::to_value(),
                element_bits: 4,
            }),
            Err(CommitTrimError::ElementSize(ElementSizeError::NotByteAligned {
                elements: 1,
                element_bits: 4,
            }))
        );
    }

    #[test]
    fn f32_to_bf16_accepts_a_supported_input_width() {
        assert_eq!(
            config_commit_cast(CommitCastInput {
                in_packet: <m![Eight]>::to_value(),
                kind: CommitCastKind::F32ToBf16,
            }),
            Ok(())
        );
    }

    #[test]
    fn f32_to_bf16_rejects_an_unsupported_input_width() {
        assert_eq!(
            config_commit_cast(CommitCastInput {
                in_packet: <m![Six]>::to_value(),
                kind: CommitCastKind::F32ToBf16,
            }),
            Err(CommitCastError::InvalidWidth {
                commit_in_size: 24,
                unit: 16,
                legal: vec![16, 32],
            })
        );
    }

    #[test]
    fn f32_to_bf16_rejects_an_aligned_width_beyond_the_commit_limit() {
        axes![Twelve = 12];

        assert_eq!(
            config_commit_cast(CommitCastInput {
                in_packet: <m![Twelve]>::to_value(),
                kind: CommitCastKind::F32ToBf16,
            }),
            Err(CommitCastError::InvalidWidth {
                commit_in_size: 48,
                unit: 16,
                legal: vec![16, 32],
            })
        );
    }
}
