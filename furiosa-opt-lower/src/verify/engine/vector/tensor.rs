//! Vector-engine packet reshaping and intra-slice reduce verifications.

// Glob import: the `m!` macro expands to DSL type-level structs (`Padding`, `Broadcast`, ...) that
// must all be in scope.
use furiosa_mapping::*;

use crate::verify::{
    HALF_FLIT_ELEMENTS, ONE_FLIT_ELEMENTS, VE_ELEMENT_BITS, VRF_CACHE_BYTES, axis_leaves, length_from_bytes,
};
use crate::{DivideError, DivideInput};

/// Inner `Time` cells folded into the packet by Way4 concat.
const WAY4_TIME_INNER: usize = 2;
/// Accumulator slots the intra-slice reducer holds in Way4, one per non-reduce position it keeps live.
const INTRA_SLICE_ACCUMULATOR_SLOTS: usize = 8;

/// Why a vector packet reshape or intra-slice reduce is not realizable on the Vector engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VectorError {
    /// `vector_narrow_split` input packet is not one flit.
    #[error("Split requires Packet of 8 elements (one flit).")]
    SplitRequiresOneFlit,
    /// `vector_narrow_split` time factor mismatch.
    #[error("Vector_Split time factor mismatch. Expected: {expected}, got: {got}")]
    SplitTimeMismatch {
        /// Expected time.
        expected: Mapping,
        /// Declared output time.
        got: Mapping,
    },
    /// `vector_narrow_split` output packet is not the front half of a flit.
    #[error("Vector_Split output Packet2 must have 4 elements (front half of flit), got: {0}")]
    SplitOutputSize(usize),
    /// `vector_narrow_split` packet mismatch.
    #[error("Vector_Split packet mismatch. Expected: {expected}, got: {got}")]
    SplitPacketMismatch {
        /// Expected packet.
        expected: Mapping,
        /// Declared output packet.
        got: Mapping,
    },
    /// `vector_widen_concat` input packet is not Way4.
    #[error("Concat requires Packet of 4 elements (Way4 mode).")]
    ConcatRequiresWay4,
    /// `vector_widen_concat` output packet is not one flit.
    #[error("Vector_Concat output Packet2 must have 8 elements (one flit), got: {0}")]
    ConcatOutputSize(usize),
    /// `vector_widen_concat` time factor mismatch.
    #[error("Vector_Concat time factor mismatch. Expected: {expected}, got: {got}")]
    ConcatTimeMismatch {
        /// Expected time.
        expected: Mapping,
        /// Declared output time.
        got: Mapping,
    },
    /// `vector_widen_concat` packet mismatch.
    #[error("Vector_Concat packet mismatch. Expected: {expected}, got: {got}")]
    ConcatPacketMismatch {
        /// Expected packet.
        expected: Mapping,
        /// Declared output packet.
        got: Mapping,
    },
    /// `vector_narrow_trim` input packet is not one flit.
    #[error(
        "vector_narrow_trim: input Packet must have 8 elements (one flit), got {0}. \
         vector_narrow_trim strips the back-4 dummy lanes before float operations; \
         if Packet is already 4, you don't need it."
    )]
    TrimInputSize(usize),
    /// `vector_narrow_trim` back-4 lanes are not dummy padding.
    #[error(
        "vector_narrow_trim: the back 4 lanes of the packet must be dummy (padding), but got: {0}. \
         If they contain real data, use vector_narrow_split() instead."
    )]
    TrimBackNotDummy(Mapping),
    /// `vector_narrow_trim` output packet is not the front 4.
    #[error("vector_narrow_trim: output Packet2 must have 4 elements, got {0}.")]
    TrimOutputSize(usize),
    /// `vector_narrow_trim` output packet does not match the front 4 of the input.
    #[error("vector_narrow_trim: Packet2 must match the front 4 of Packet. Expected: {expected}, got: {got}.")]
    TrimPacketMismatch {
        /// Expected packet (front 4 of input).
        expected: Mapping,
        /// Declared output packet.
        got: Mapping,
    },
    /// `vector_widen_pad` input packet is not the trimmed half.
    #[error("vector_widen_pad: input Packet must have 4 elements (after vector_narrow_trim), got {0}.")]
    PadInputSize(usize),
    /// `vector_widen_pad` output packet is not one flit.
    #[error("vector_widen_pad: output Packet2 must have 8 elements (one flit), got {0}.")]
    PadOutputSize(usize),
    /// `vector_widen_pad` output packet is not the input padded to one flit.
    #[error("vector_widen_pad: Packet2 must be Packet padded to 8. Expected: {expected}, got: {got}.")]
    PadPacketMismatch {
        /// Expected packet (input padded to 8).
        expected: Mapping,
        /// Declared output packet.
        got: Mapping,
    },
    /// `vector_intra_slice_unzip` group axis is not in the stream it unzips.
    #[error("vector_intra_slice_unzip: group axis {group_axis} is not in the stream it unzips: {stream}")]
    UnzipGroupAxisMissing {
        /// The requested group axis.
        group_axis: Ident,
        /// The `(Time, Packet)` stream searched.
        stream: Mapping,
    },
    /// `vector_intra_slice_unzip` group does not fit the VE register-file cache.
    #[error(
        "vector_intra_slice_unzip: {volume} elements sit inside the group axis {group_axis}, over the \
         {max_volume} one group may hold in the VE register-file cache. Move {group_axis} inward in the \
         fetch/collect time, so that at most {max_volume} elements follow it."
    )]
    UnzipGroupTooLarge {
        /// The group axis.
        group_axis: Ident,
        /// Elements inside the group axis.
        volume: usize,
        /// Elements one group may hold.
        max_volume: usize,
    },
    /// Intra-slice reduce: output does not divide input.
    #[error("[Intra-slice reduce] divide failed: output shape must divide input shape: {0}")]
    ReduceIndivisible(#[from] DivideError),
    /// Intra-slice reduce: a reduced axis does not carry `reduce_label`.
    #[error(
        "IntraSliceReduce: all reduced axes must match the specified reduce_label {reduce_label}, got quotient {quotient}"
    )]
    ReduceLabelMismatch {
        /// The required reduce label.
        reduce_label: Ident,
        /// The quotient (the reduced axes).
        quotient: Mapping,
    },
    /// Intra-slice reduce: a `reduce_label` axis survives into the retained division terms.
    #[error(
        "IntraSliceReduce: all the reduce axes must be fully reduced, but reduce_label {reduce_label} is still present in a retained axis"
    )]
    ReduceAxisNotFullyReduced {
        /// The reduce label found in a retained axis.
        reduce_label: Ident,
    },
    /// Intra-slice reduce: packet is neither preserved nor reduced to 4.
    #[error(
        "IntraSliceReduce: Packet should be either preserved or reduced to 4 (for partial reduction), got Packet {in_packet} -> OutPacket {out_packet}"
    )]
    ReducePacketMismatch { in_packet: Mapping, out_packet: Mapping },
    /// Intra-slice reduce: the non-reduce `Time` axes inner to the reduce outnumber the accumulators.
    #[error(
        "IntraSliceReduce: InnerTime, the non-reduce Time axes inner to the outermost {reduce_label} \
         axis, needs {slots} accumulator slots but the reducer holds {limit}. Time {time}"
    )]
    ReduceAccumulatorsExceeded {
        /// The required reduce label.
        reduce_label: Ident,
        time: Mapping,
        /// Slots the mapping asks for (`InnerTime::SIZE`).
        slots: usize,
        /// Slots the reducer holds.
        limit: usize,
    },
}

/// Inputs used to split a one-flit vector packet.
pub struct VectorNarrowSplitInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
}

/// Inputs used to concatenate a Way4 vector packet.
pub struct VectorWidenConcatInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
}

/// Inputs used to trim a vector packet to Way4 width.
pub struct VectorNarrowTrimInput {
    pub in_packet: Mapping,
    pub out_packet: Mapping,
}

/// Inputs used to pad a Way4 vector packet to one flit.
pub struct VectorWidenPadInput {
    pub in_packet: Mapping,
    pub out_packet: Mapping,
}

/// Inputs used to determine an intra-slice unzip group size.
pub struct VectorIntraSliceUnzipInput {
    pub group_axis: Ident,
    pub in_time: Mapping,
    pub in_packet: Mapping,
}

/// Inputs used to validate an intra-slice reduction label.
pub struct ReduceLabelInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
    pub reduce_label: Ident,
}

/// `vector_narrow_split`: split a one-flit (8-element) packet, folding the front 4 into `Packet2` and
/// the back 4 onto `Time2`.
pub fn config_vector_narrow_split(input: VectorNarrowSplitInput) -> Result<(), VectorError> {
    let VectorNarrowSplitInput {
        in_time,
        in_packet,
        out_time,
        out_packet,
    } = input;
    if in_packet.size() != ONE_FLIT_ELEMENTS {
        return Err(VectorError::SplitRequiresOneFlit);
    }
    let (packet_outer, packet_inner) = in_packet.split_at(HALF_FLIT_ELEMENTS);
    let expected_time = in_time.pair(packet_outer).normalize();
    let expected_packet = packet_inner.normalize();

    let out_time = out_time.normalize();
    if expected_time != out_time {
        return Err(VectorError::SplitTimeMismatch {
            expected: expected_time,
            got: out_time,
        });
    }
    if out_packet.size() != HALF_FLIT_ELEMENTS {
        return Err(VectorError::SplitOutputSize(out_packet.size()));
    }
    let out_packet = out_packet.normalize();
    if expected_packet != out_packet {
        return Err(VectorError::SplitPacketMismatch {
            expected: expected_packet,
            got: out_packet,
        });
    }
    Ok(())
}

/// `vector_widen_concat`: concatenate a 4-element packet with the inner 2 `Time` cells into a one-flit
/// (8-element) `Packet2` (Way4 mode).
pub fn config_vector_widen_concat(input: VectorWidenConcatInput) -> Result<(), VectorError> {
    let VectorWidenConcatInput {
        in_time,
        in_packet,
        out_time,
        out_packet,
    } = input;
    if in_packet.size() != HALF_FLIT_ELEMENTS {
        return Err(VectorError::ConcatRequiresWay4);
    }
    let (time_outer, time_inner) = in_time.split_at(WAY4_TIME_INNER);
    let expected_time = time_outer.normalize();
    let expected_packet = time_inner.pair(in_packet).normalize();

    if out_packet.size() != ONE_FLIT_ELEMENTS {
        return Err(VectorError::ConcatOutputSize(out_packet.size()));
    }
    let out_time = out_time.normalize();
    let out_packet = out_packet.normalize();
    if expected_time != out_time {
        return Err(VectorError::ConcatTimeMismatch {
            expected: expected_time,
            got: out_time,
        });
    }
    if expected_packet != out_packet {
        return Err(VectorError::ConcatPacketMismatch {
            expected: expected_packet,
            got: out_packet,
        });
    }
    Ok(())
}

/// `vector_narrow_trim`: strip the back-4 dummy lanes of a one-flit packet, keeping the front 4.
pub fn config_vector_narrow_trim(input: VectorNarrowTrimInput) -> Result<(), VectorError> {
    let VectorNarrowTrimInput { in_packet, out_packet } = input;
    if in_packet.size() != ONE_FLIT_ELEMENTS {
        return Err(VectorError::TrimInputSize(in_packet.size()));
    }
    let (packet_outer, packet_inner) = in_packet.split_at(HALF_FLIT_ELEMENTS);
    // The back 4 must be dummy padding (`[1 # 2]`); otherwise use vector_narrow_split.
    if packet_outer.normalize() != <m![1 # 2]>::to_value().normalize() {
        return Err(VectorError::TrimBackNotDummy(packet_outer));
    }
    if out_packet.size() != HALF_FLIT_ELEMENTS {
        return Err(VectorError::TrimOutputSize(out_packet.size()));
    }
    let out_packet = out_packet.normalize();
    if packet_inner.normalize() != out_packet {
        return Err(VectorError::TrimPacketMismatch {
            expected: packet_inner,
            got: out_packet,
        });
    }
    Ok(())
}

/// `vector_widen_pad`: restore the back-4 dummy lanes stripped by `vector_narrow_trim` (4 -> 8).
pub fn config_vector_widen_pad(input: VectorWidenPadInput) -> Result<(), VectorError> {
    let VectorWidenPadInput { in_packet, out_packet } = input;
    if in_packet.size() != HALF_FLIT_ELEMENTS {
        return Err(VectorError::PadInputSize(in_packet.size()));
    }
    if out_packet.size() != ONE_FLIT_ELEMENTS {
        return Err(VectorError::PadOutputSize(out_packet.size()));
    }
    let expected = in_packet.padding(ONE_FLIT_ELEMENTS, PaddingKind::Top).normalize();
    let out_packet = out_packet.normalize();
    if expected != out_packet {
        return Err(VectorError::PadPacketMismatch {
            expected,
            got: out_packet,
        });
    }
    Ok(())
}

/// `vector_intra_slice_unzip`: the elements inside `group_axis`, which is the branch group size.
///
/// Group 0 waits in the VE register-file cache while group 1 streams in, so one group must fit it.
pub fn config_vector_intra_slice_unzip(input: VectorIntraSliceUnzipInput) -> Result<usize, VectorError> {
    let VectorIntraSliceUnzipInput {
        group_axis,
        in_time,
        in_packet,
    } = input;
    let stream = in_time.pair(in_packet);
    let mut group_volume = 1;
    for axis in axis_leaves(&stream) {
        if axis.idents().contains(&group_axis) {
            let max_volume = length_from_bytes(VE_ELEMENT_BITS, VRF_CACHE_BYTES)
                .expect("the fixed VRF cache size contains whole VE elements");
            if group_volume > max_volume {
                return Err(VectorError::UnzipGroupTooLarge {
                    group_axis,
                    volume: group_volume,
                    max_volume,
                });
            }
            return Ok(group_volume);
        }
        // Padded extents: the stream walks the padding cells too.
        group_volume *= axis.size();
    }
    Err(VectorError::UnzipGroupAxisMissing { group_axis, stream })
}

/// Intra-slice reduce: `[OutTime, OutPacket]` must divide `[Time, Packet]` exactly, every reduced axis
/// must carry `reduce_label`, the packet is preserved or reduced to 4, and the `Time` axes inner to
/// the reduce must fit the accumulator slots.
pub fn config_reduce_label(input: ReduceLabelInput) -> Result<(), VectorError> {
    let ReduceLabelInput {
        in_time,
        in_packet,
        out_time,
        out_packet,
        reduce_label,
    } = input;
    let dividend = in_time.clone().pair(in_packet.clone());
    let divisor = out_time.clone().pair(out_packet.clone());

    let division_terms = crate::config_divide_exact(DivideInput {
        dividend: dividend.clone(),
        divisor: divisor.clone(),
    })?;

    let quotient = crate::config_divide_relaxed(DivideInput { dividend, divisor }).dividend_residue;
    if !quotient.idents().iter().all(|ident| ident == &reduce_label) {
        return Err(VectorError::ReduceLabelMismatch { reduce_label, quotient });
    }
    if !division_terms
        .iter()
        .all(|d| d.idents.iter().all(|ident| ident != &reduce_label))
    {
        return Err(VectorError::ReduceAxisNotFullyReduced { reduce_label });
    }

    // Only `Time` indexes slots: `Packet` is the spatial width of one, so it costs none.
    let slots = crate::verify::inner_reduce_extent(&in_time, &out_time)?;
    if slots > INTRA_SLICE_ACCUMULATOR_SLOTS {
        return Err(VectorError::ReduceAccumulatorsExceeded {
            reduce_label,
            time: in_time,
            slots,
            limit: INTRA_SLICE_ACCUMULATOR_SLOTS,
        });
    }

    let packet = in_packet.normalize();
    let out_packet = out_packet.normalize();
    if packet != out_packet && out_packet != <m![1 # 4]>::to_value().normalize() {
        return Err(VectorError::ReducePacketMismatch {
            in_packet: packet,
            out_packet,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![R = 19, A = 2, I = 2, G = 512, H = 264, S = 16, U = 12, V = 8, W = 4];

    /// The group axis innermost in the time: a group is the packet alone, whatever the row length.
    #[test]
    fn unzip_group_axis_inner_is_one_packet() {
        assert_eq!(
            config_vector_intra_slice_unzip(VectorIntraSliceUnzipInput {
                group_axis: Ident::I,
                in_time: <m![G / 8, I]>::to_value(),
                in_packet: <m![G % 8]>::to_value(),
            }),
            Ok(8)
        );
    }

    /// The group axis outermost: the group is the whole row, and 512 `i32` does not fit the 1 KiB cache.
    #[test]
    fn unzip_group_axis_outer_over_cache() {
        assert_eq!(
            config_vector_intra_slice_unzip(VectorIntraSliceUnzipInput {
                group_axis: Ident::I,
                in_time: <m![I, G / 8]>::to_value(),
                in_packet: <m![G % 8]>::to_value(),
            }),
            Err(VectorError::UnzipGroupTooLarge {
                group_axis: Ident::I,
                volume: 512,
                max_volume: 256,
            })
        );
    }

    /// Corner case: the cache holds exactly 256 `i32`, so 256 elements pass and 264 do not.
    #[test]
    fn unzip_group_bound_is_the_cache_in_i32_elements() {
        assert_eq!(
            config_vector_intra_slice_unzip(VectorIntraSliceUnzipInput {
                group_axis: Ident::I,
                in_time: <m![I, G / 512, G / 8 % 32]>::to_value(),
                in_packet: <m![G % 8]>::to_value(),
            }),
            Ok(256)
        );
        assert!(matches!(
            config_vector_intra_slice_unzip(VectorIntraSliceUnzipInput {
                group_axis: Ident::I,
                in_time: <m![I, H / 8]>::to_value(),
                in_packet: <m![H % 8]>::to_value(),
            }),
            Err(VectorError::UnzipGroupTooLarge { volume: 264, .. })
        ));
    }

    /// A group axis that is not in the stream is refused, rather than silently taken as the whole stream.
    #[test]
    fn unzip_group_axis_must_be_in_the_stream() {
        assert!(matches!(
            config_vector_intra_slice_unzip(VectorIntraSliceUnzipInput {
                group_axis: Ident::I,
                in_time: <m![G / 8]>::to_value(),
                in_packet: <m![G % 8]>::to_value(),
            }),
            Err(VectorError::UnzipGroupAxisMissing { .. })
        ));
    }

    /// Padded stride/modulo split through `config_vector_narrow_split`.
    ///
    /// Input  Time  = `R # 24 / 4`, Packet = `(R # 24 % 4, A)` (size 8).
    /// Output Time2 = `(R # 24 / 4, R # 24 / 2 % 2)`, Packet2 = `(R # 24 % 2, A)`.
    ///
    /// The complementary halves only line up because `R # 24 % n` factorizes to its minimal-aligned
    /// period, matching the period the `/ stride` partner produces.
    #[test]
    fn vector_narrow_split_padded_stride_modulo() {
        config_vector_narrow_split(VectorNarrowSplitInput {
            in_time: <m![R # 24 / 4]>::to_value(),
            in_packet: <m![R # 24 % 4, A]>::to_value(),
            out_time: <m![R # 24 / 4, R # 24 / 2 % 2]>::to_value(),
            out_packet: <m![R # 24 % 2, A]>::to_value(),
        })
        .unwrap();
    }

    /// A reduce on the innermost `Time` axis holds one accumulator, however wide the axes outside it.
    #[test]
    fn reduce_innermost_axis_needs_one_slot() {
        config_reduce_label(ReduceLabelInput {
            in_time: <m![U, S]>::to_value(),
            in_packet: <m![1 # 4]>::to_value(),
            out_time: <m![U]>::to_value(),
            out_packet: <m![1 # 4]>::to_value(),
            reduce_label: S::NAME,
        })
        .unwrap();
    }

    /// The `Packet`-only reduce spends no slot: `Time` passes through untouched at any size.
    #[test]
    fn reduce_in_packet_only_needs_one_slot() {
        config_reduce_label(ReduceLabelInput {
            in_time: <m![U]>::to_value(),
            in_packet: <m![W]>::to_value(),
            out_time: <m![U]>::to_value(),
            out_packet: <m![1 # 4]>::to_value(),
            reduce_label: W::NAME,
        })
        .unwrap();
    }

    /// The non-reduce axes inner to the reduce multiply into the slot count, up to the 8 available.
    #[test]
    fn reduce_inner_non_reduce_axes_fill_the_slots() {
        config_reduce_label(ReduceLabelInput {
            in_time: <m![S, U % 2, V % 4]>::to_value(),
            in_packet: <m![1 # 4]>::to_value(),
            out_time: <m![U % 2, V % 4]>::to_value(),
            out_packet: <m![1 # 4]>::to_value(),
            reduce_label: S::NAME,
        })
        .unwrap();

        // One more inner position than the reducer can hold.
        let over = config_reduce_label(ReduceLabelInput {
            in_time: <m![S, U % 3, V % 4]>::to_value(),
            in_packet: <m![1 # 4]>::to_value(),
            out_time: <m![U % 3, V % 4]>::to_value(),
            out_packet: <m![1 # 4]>::to_value(),
            reduce_label: S::NAME,
        });
        assert_eq!(
            over,
            Err(VectorError::ReduceAccumulatorsExceeded {
                reduce_label: S::NAME,
                time: <m![S, U % 3, V % 4]>::to_value(),
                slots: 12,
                limit: INTRA_SLICE_ACCUMULATOR_SLOTS,
            })
        );
    }

    /// A reduced axis inner to the outermost reduced one folds into the same slot, so only the
    /// non-reduce axis between the two is charged.
    #[test]
    fn reduce_axis_inner_to_the_reduce_costs_no_slot() {
        config_reduce_label(ReduceLabelInput {
            in_time: <m![S / 8, U % 2, S % 8]>::to_value(),
            in_packet: <m![1 # 4]>::to_value(),
            out_time: <m![U % 2]>::to_value(),
            out_packet: <m![1 # 4]>::to_value(),
            reduce_label: S::NAME,
        })
        .unwrap();
    }

    /// A padded non-reduce axis occupies a slot per padded position, not per live element.
    #[test]
    fn reduce_charges_the_padded_extent() {
        let padded = config_reduce_label(ReduceLabelInput {
            in_time: <m![S, A # 16]>::to_value(),
            in_packet: <m![1 # 4]>::to_value(),
            out_time: <m![A # 16]>::to_value(),
            out_packet: <m![1 # 4]>::to_value(),
            reduce_label: S::NAME,
        });
        assert_eq!(
            padded,
            Err(VectorError::ReduceAccumulatorsExceeded {
                reduce_label: S::NAME,
                time: <m![S, A # 16]>::to_value(),
                slots: 16,
                limit: INTRA_SLICE_ACCUMULATOR_SLOTS,
            })
        );
    }
}
