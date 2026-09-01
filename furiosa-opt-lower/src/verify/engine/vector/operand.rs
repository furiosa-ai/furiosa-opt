//! VRF-operand verification: what the VRF indexer can address for one VE access.
//!
//! One access is `Packet` elements per way (8 in `Way8`, 4 in `Way4`) and comes from ONE indexer
//! step: a broadcast (one address for every lane) or `Packet` consecutive cells (the ISA's
//! `ReadIndexerOpMode::{BroadcastRead, ContiguousRead}`). The backend's `make_vrf_sequencer` checks
//! it again, as an internal error. The second rule, that the stream match every named axis of the
//! operand, is the DSL's own policy.

use furiosa_mapping::{Mapping, PaddingKind, SequencerError, SequencerMode, sequence};

use crate::verify::{HALF_FLIT_ELEMENTS, ONE_FLIT_ELEMENTS};

/// Why the stream cannot read the operand, ahead of any indexer question.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnreadableCause {
    /// Named axes of the operand the stream never matched. A leftover broadcast or pad is not one.
    #[error(
        "the stream leaves operand axes unmatched: {}. A VE op matches every named axis of its \
         operand against `Time` x `Packet`, so either drop the axis from the operand, which reads it \
         as a broadcast, or widen the stream to cover it. This is the DSL's rule and stricter than \
         the compiler's, so that a half-read operand cannot pass for a working kernel.",
        .0.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    )]
    Unread(Vec<Mapping>),
    /// A stream segment matched no operand axis, so the read cannot carve the operand at all.
    #[error(
        "part of the stream matches no operand axis: an operand's axes must be the stream's, \
         regrouped, or absent from it entirely (which reads as a broadcast)"
    )]
    Unmatched,
    /// The operand carries a `Bottom` pad, so it is not a fully written register.
    #[error("the operand carries a write hole (`Bottom` padding), which a VE op may not read")]
    WriteHole,
}

/// Why a VRF tensor cannot be read as a VE operand.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VrfOperandError {
    /// The stream does not read the operand.
    #[error("VRF operand {vrf_element} is not readable by the stream {stream}: {cause}")]
    Unreadable {
        vrf_element: Mapping,
        stream: Mapping,
        /// Which way the read failed.
        cause: UnreadableCause,
    },
    /// The packet-sized access is neither one broadcast address nor one contiguous run.
    #[error(
        "VRF operand {vrf_element} cannot feed the {}-element packet {stream_packet}: the indexer reads one \
         access as a single broadcast address or as {} consecutive elements, and this operand gives \
         it neither. Write the operand with the packet axes innermost and contiguous (change the \
         `collect` / `to_vrf` that produced it), or drop them from it entirely (broadcast). \
         `reshape` cannot fix this: it may not permute axes.",
        stream_packet.size(),
        stream_packet.size()
    )]
    PacketNotIndexable {
        vrf_element: Mapping,
        stream_packet: Mapping,
    },
}

/// Inputs used to validate a VRF operand access.
pub struct VrfOperandInput {
    pub vrf_element: Mapping,
    pub stream_time: Mapping,
    pub stream_packet: Mapping,
}

/// Judges an operand against both rules the [module docs](self) state, one sequencing each.
///
/// Indexability asks about the packet alone, since the access IS the packet: cutting the operand at
/// the access width would assume it steps once per stream position, which `m![A / 16]` under a
/// stream over `m![A]` does not.
pub fn config_vrf_operand(input: VrfOperandInput) -> Result<(), VrfOperandError> {
    let VrfOperandInput {
        vrf_element,
        stream_time,
        stream_packet,
    } = input;
    let access = stream_packet.size();
    assert!(
        matches!(access, HALF_FLIT_ELEMENTS | ONE_FLIT_ELEMENTS),
        "the access window is the ALU width ({HALF_FLIT_ELEMENTS} or {ONE_FLIT_ELEMENTS}), \
         got a {access}-element packet {stream_packet}"
    );
    let stream = stream_time.pair(stream_packet.clone());
    let unreadable = |cause| VrfOperandError::Unreadable {
        vrf_element: vrf_element.clone(),
        stream: stream.clone(),
        cause,
    };

    // On the whole operand: a hole the carve folds into a stride goes unseen below.
    if has_write_hole(&vrf_element) {
        return Err(unreadable(UnreadableCause::WriteHole));
    }

    if !packet_reads_one_access(&vrf_element, &stream_packet) {
        return Err(VrfOperandError::PacketNotIndexable {
            vrf_element: vrf_element.clone(),
            stream_packet,
        });
    }

    // Coverage: a live operand cell no stream position reaches comes back as `Unconsumed`.
    sequence(&[&vrf_element], &[&stream], SequencerMode::Read).map_err(|error| {
        unreadable(match error {
            SequencerError::Unconsumed(memories) => UnreadableCause::Unread(memories.into_iter().collect()),
            SequencerError::StreamUnmatchedSegment => UnreadableCause::Unmatched,
            SequencerError::InputBottomPadding => UnreadableCause::WriteHole,
        })
    })?;
    Ok(())
}

/// Whether one access reads the operand in a mode the indexer has: one broadcast address, or
/// `Packet` consecutive cells.
///
/// [`SequencerMode::Locate`], as `config_commit` judges its own packet: one access is a fragment of
/// the read, covering neither the operand nor its padding.
fn packet_reads_one_access(element: &Mapping, packet: &Mapping) -> bool {
    // A packet that cannot carve the operand at all is no access either.
    let Ok(configs) = sequence(&[element], &[packet], SequencerMode::Locate) else {
        return false;
    };
    let [access] = &configs[..] else {
        unreachable!("one stream was sequenced, so the sequencer answered with one config")
    };

    // One address for every lane: no lane steps the operand.
    let broadcast = access.0.iter().all(|(_, entry)| entry.memory_stride == 0);
    // Or one run: coalescing has already merged a run the packet spells in several terms.
    let mut entries = access.0.iter().map(|(_, entry)| entry);
    let contiguous = matches!(
        (entries.next(), entries.next()),
        (Some(run), None) if run.memory_stride == 1 && run.mapping.size() == packet.size()
    );
    broadcast || contiguous
}

/// Whether the mapping carries a `Bottom` pad anywhere, which is the marker a `view_mut` write hole
/// leaves behind.
fn has_write_hole(mapping: &Mapping) -> bool {
    match mapping {
        Mapping::Padding {
            kind: PaddingKind::Bottom,
            ..
        } => true,
        Mapping::Stride { inner, .. }
        | Mapping::Modulo { inner, .. }
        | Mapping::Resize { inner, .. }
        | Mapping::Padding { inner, .. } => has_write_hole(inner),
        Mapping::Pair { left, right } => has_write_hole(left) || has_write_hole(right),
        Mapping::Symbol { .. } | Mapping::Broadcast { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    // Glob import: the `m!` macro expands to DSL type-level structs that must all be in scope.
    use furiosa_mapping::*;

    use super::*;

    axes![H = 32, G = 4, D = 8, P = 4];

    /// The logical coordinate a stream cell reads, per axis. A term `symbol // stride % modulo` at
    /// digit `d` contributes `d * stride` to that symbol, so `m![W / 2, W % 2]` and `m![W]` decode
    /// the same physical position to the same `W`. `None` is a cell that is padding or out of bounds.
    fn coords<Map: M>(offset: usize) -> Option<BTreeMap<Ident, usize>> {
        let Cell::Index(index) = Map::map(offset) else {
            return None;
        };
        let mut axes = BTreeMap::new();
        for (term, digit) in index.0.iter() {
            let Atom::Symbol { symbol, .. } = &term.inner else {
                panic!("these mappings name symbols, not composites")
            };
            *axes.entry(*symbol).or_insert(0) += digit * term.stride;
        }
        Some(axes)
    }

    /// The operand cell every stream position reads, in logical coordinates the rule never touches.
    /// `None` is a position no cell answers; a `None` lane inside is a padding stream cell.
    ///
    /// A cell answers the coordinates it COVERS: `m![A / 16]` steps `A` in 16s, so its cell at
    /// `A = 16` answers 16 through 31, the largest step at or below.
    fn simulate<Element: M, Time: M, Packet: M>() -> Option<Vec<Option<usize>>> {
        let live: Vec<(BTreeMap<Ident, usize>, usize)> = (0..Element::SIZE)
            .filter_map(|offset| coords::<Element>(offset).map(|axes| (axes, offset)))
            .collect();
        // Every coordinate the operand steps an axis to, which "covers" is read against.
        let mut steps: BTreeMap<Ident, BTreeSet<usize>> = BTreeMap::new();
        for (axes, _) in &live {
            for (axis, value) in axes {
                steps.entry(*axis).or_default().insert(*value);
            }
        }
        (0..Time::SIZE * Packet::SIZE)
            .map(|position| {
                let (Some(mut want), Some(packet)) = (
                    coords::<Time>(position / Packet::SIZE),
                    coords::<Packet>(position % Packet::SIZE),
                ) else {
                    return Some(None);
                };
                for (axis, value) in packet {
                    // Summed, not overwritten: one axis may be split across `Time` and `Packet`.
                    *want.entry(axis).or_insert(0) += value;
                }
                live.iter()
                    .find(|(axes, _)| {
                        axes.iter().all(|(axis, value)| {
                            want.get(axis)
                                .is_some_and(|coord| steps[axis].range(..=coord).next_back() == Some(value))
                        })
                    })
                    .map(|(_, offset)| Some(*offset))
            })
            .collect()
    }

    /// Checks the rule's verdict against [`simulate`], judging each access the way the hardware does.
    /// A padding lane sits out of it, so contiguity is "live lane `l` reads `base + l`" under one
    /// base. Returns the verdict, so a rejection test pins the error too.
    #[track_caller]
    fn assert_verdict<Element: M, Time: M, Packet: M>(legal: bool) -> Result<(), VrfOperandError> {
        let verdict = config_vrf_operand(VrfOperandInput {
            vrf_element: Element::to_value(),
            stream_time: Time::to_value(),
            stream_packet: Packet::to_value(),
        });
        assert_eq!(verdict.is_ok(), legal, "the rule says {verdict:?}");

        let reads = simulate::<Element, Time, Packet>();
        let simulated = reads.as_ref().is_some_and(|reads| {
            reads.chunks(Packet::SIZE).all(|access| {
                let lanes: Vec<(usize, usize)> = access
                    .iter()
                    .enumerate()
                    .filter_map(|(lane, read)| read.map(|offset| (lane, offset)))
                    .collect();
                let Some(&(first_lane, base)) = lanes.first() else {
                    return true;
                };
                lanes.iter().all(|&(_, offset)| offset == base)
                    || lanes.iter().all(|&(lane, offset)| offset + first_lane == base + lane)
            })
        });
        assert_eq!(simulated, legal, "the simulation disagrees with the rule: {reads:?}");
        verdict
    }

    /// A shape the rule accepts, agreed on by the simulation.
    #[track_caller]
    fn assert_legal<Element: M, Time: M, Packet: M>() {
        assert_verdict::<Element, Time, Packet>(true).expect("asserted above");
    }

    /// A shape the rule refuses, agreed on by the simulation, and the error it refuses with.
    #[track_caller]
    fn assert_illegal<Element: M, Time: M, Packet: M>() -> VrfOperandError {
        assert_verdict::<Element, Time, Packet>(false).expect_err("asserted above")
    }

    /// The common operand: the packet reads its innermost axis, 8 consecutive elements per access.
    #[test]
    fn contiguous_packet_reads() {
        assert_legal::<m![H], m![H / 8], m![H % 8]>();
    }

    /// A per-time scalar: the operand has no packet axis at all, so one address feeds every lane.
    #[test]
    fn broadcast_packet_reads() {
        assert_legal::<m![D], m![D, H / 8], m![H % 8]>();
    }

    /// An operand shorter than one access still reads: the window is the stream's, not its own size.
    #[test]
    fn operand_shorter_than_one_access_broadcasts() {
        assert_legal::<m![G], m![G, D / 8], m![D % 8]>();
    }

    /// An operand COARSER than the stream, a per-group scale: one cell covers 8 stream positions, so
    /// a 4-lane access sits inside one of them. Cutting the OPERAND instead would call it strided.
    #[test]
    fn coarse_operand_broadcasts_inside_one_access() {
        assert_legal::<m![H / 8], m![H / 4], m![H % 4]>();
    }

    /// Finer than the access: 4 lanes span two cells and read `[o, o, o + 1, o + 1]`, neither mode.
    #[test]
    fn coarse_operand_stepping_inside_one_access_rejects() {
        let error = assert_illegal::<m![H / 2], m![H / 4], m![H % 4]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// No axis the packet names, padding included: the absent axis is what makes it a broadcast.
    #[test]
    fn operand_without_packet_axis_broadcasts() {
        assert_legal::<m![1 # 8], m![H / 8], m![H % 8]>();
    }

    /// Two packed `P` rows in one access: the second starts where the first ends, so it is one run.
    #[test]
    fn packet_spanning_packed_rows_reads() {
        assert_legal::<m![G, P], m![G / 2], m![G % 2, P]>();
    }

    /// The same rows `P`-padded, read 4 lanes at a time: the pad sits outside the access.
    #[test]
    fn way4_packet_reads_inside_padded_row() {
        assert_legal::<m![G, P # 8], m![G], m![P]>();
    }

    /// The packet walks the OUTER axis, so consecutive lanes sit `D` apart and no access gathers them.
    #[test]
    fn strided_packet_rejects() {
        let error = assert_illegal::<m![G, D], m![D], m![G]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// Half the access broadcasts, half reads live cells. Each alone is indexable, the mix is not.
    #[test]
    fn mixed_broadcast_and_live_lanes_reject() {
        let error = assert_illegal::<m![D], m![D / 4], m![D % 4, 2]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// The same two rows under an 8-wide packet: a padded row is 8 cells, so the second row's lanes
    /// sit 8 apart, not 4. The padded stride is the one the indexer walks.
    #[test]
    fn padding_breaks_contiguity() {
        let error = assert_illegal::<m![G, P # 8], m![G / 2], m![G % 2, P]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
        // Pinned here, in `ve_vrf_strided_packet`'s answer key, and in that fixture's `snapshot.toml`
        // reason. Keep the three in step.
        assert!(
            error.to_string().contains("cannot feed the 8-element packet"),
            "{error}"
        );
    }

    /// A packet axis the operand lacks is a shape error: the packet broadcasts and the operand's own
    /// axis goes unread.
    #[test]
    fn packet_axis_the_operand_lacks_rejects() {
        let error = assert_illegal::<m![H], m![1], m![D]>();
        assert!(
            matches!(
                &error,
                VrfOperandError::Unreadable {
                    cause: UnreadableCause::Unread(_),
                    ..
                }
            ),
            "{error}"
        );
    }

    /// `vector_narrow_split` puts the two half-flits just OUTSIDE the packet, so an operand written
    /// contiguous across all 8 stays legal: the back half is the indexer's next step.
    #[test]
    fn narrow_split_half_flit_reads() {
        assert_legal::<m![D], m![D / 4], m![D % 4]>();
    }

    /// `Time` carries no indexability rule, so it may walk the operand in any order: inner axis
    /// first here, while the packet still reads eight consecutive cells.
    #[test]
    fn time_walking_the_operand_out_of_order_reads() {
        assert_legal::<m![G, H], m![H / 8, G], m![H % 8]>();
    }

    /// A packet that is ALL padding (the flit a per-time scalar fills): one live lane, one address.
    #[test]
    fn packet_of_padding_reads_one_address() {
        assert_legal::<m![H], m![H], m![1 # 8]>();
        assert_legal::<m![H], m![H], m![1 # 4]>();
    }

    /// The operand that stream writes: one live cell per flit, so the pads line up on both sides.
    #[test]
    fn scalar_per_flit_operand_reads() {
        assert_legal::<m![H, 1 # 8], m![H], m![1 # 8]>();
    }

    /// Padding OUTSIDE the live lanes: four consecutive cells, the pad lanes trailing off them.
    #[test]
    fn packet_padding_outside_the_live_lanes_reads() {
        assert_legal::<m![H, G], m![H], m![1 # 2, G]>();
    }

    /// Padding BETWEEN them: the live lanes want cells `0..4`, a contiguous access hands `0, 2, 4, 6`.
    #[test]
    fn packet_padding_between_the_live_lanes_rejects() {
        let error = assert_illegal::<m![H, G], m![H], m![G, 1 # 2]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// A `Bottom` pad marks a write hole, so no read of the operand is defined. Fails before any
    /// indexer question.
    #[test]
    fn write_hole_operand_rejects() {
        let error = config_vrf_operand(VrfOperandInput {
            vrf_element: <m![G #{!} 8]>::to_value(),
            stream_time: <m![1]>::to_value(),
            stream_packet: <m![G #{!} 8]>::to_value(),
        })
        .unwrap_err();
        assert!(
            matches!(
                &error,
                VrfOperandError::Unreadable {
                    cause: UnreadableCause::WriteHole,
                    ..
                }
            ),
            "{error}"
        );
    }

    /// An operand the stream reads only half of is refused rather than half-read.
    #[test]
    fn partially_read_operand_rejects() {
        let error = assert_illegal::<m![G, D], m![D / 8], m![D % 8]>();
        assert!(
            matches!(
                &error,
                VrfOperandError::Unreadable {
                    cause: UnreadableCause::Unread(_),
                    ..
                }
            ),
            "{error}"
        );
    }
}
