//! TensorValue streams.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use abi_stable::std_types::{RBox, Tuple2};
use furiosa_mapping::*;
use furiosa_mapping_macro::primitive;
use furiosa_opt_macro::m;

/// Size of a single flit in bytes.
///
/// Data flows through the switching network in flit-sized units.
/// Both the collect engine and cast engine normalize packets to exactly one flit.
const FETCH_ALIGN_BYTES: usize = 8;
const FLIT_BYTES: usize = 32;

/// Transpose engine input packet size in bytes.
const TRANSPOSE_INPUT_BYTES: usize = 32;

/// Transpose engine output packet size in bytes.
const TRANSPOSE_OUTPUT_BYTES: usize = 32;

/// Number of elements fetched per transpose packet for non 4-bit types.
const TRANSPOSE_ELEMENTS_PER_PACKET_NON_4BIT: usize = 8;

/// Number of elements fetched per transpose packet for 4-bit types.
const TRANSPOSE_ELEMENTS_PER_PACKET_4BIT: usize = 16;

/// Maximum size for transpose `in_rows` in bytes.
const TRANSPOSE_MAX_IN_ROWS_BYTES: usize = 8;

/// Valid `in_cols` values for the transpose engine (non 4-bit types).
const TRANSPOSE_VALID_IN_COLS: &[usize] = &[8, 16, 32];

/// Valid `in_cols` values for the transpose engine (4-bit types).
const TRANSPOSE_VALID_IN_COLS_4BIT: &[usize] = &[16, 32];

/// Number of columns in the reducer's temporal accumulator.
const TEMPORAL_ACCUMULATOR_COLS: usize = 32;

/// Number of elements in `accumulate`'s output packet (i32/f32).
const ACCUMULATE_OUT_PACKET_ELEMENTS: usize = 8;

/// Valid output packet sizes for the commit engine in bytes.
const COMMIT_OUT_PACKET_SIZES: [usize; 4] = [8, 16, 24, 32];

/// Contraction mode for accumulation.
#[primitive(AccumulationKind)]
#[derive(Clone, Debug)]
pub enum AccumulationKind {
    /// Interleaved accumulation: Outputs data element-by-element across all Rows.
    Interleaved,
    /// Sequential accumulation: Outputs reduced data in each Row sequentially.
    Sequential,
}

impl std::fmt::Display for AccumulationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccumulationKind::Interleaved => write!(f, "Interleaved"),
            AccumulationKind::Sequential => write!(f, "Sequential"),
        }
    }
}

use crate::cast::*;
use crate::context::*;
use crate::memory_tensor::*;
use crate::scalar::*;
use crate::tensor::*;
use crate::vector_engine::scalar::VeScalar;
use crate::vector_engine::tensor::VectorInitTensor;

/// Marker trait for pipeline position of stream tensors.
///
/// Position does not contain Vector Engine position: VectorTensor has its own typestate.
pub trait Position: std::fmt::Debug + 'static {}

/// After beginning the pipeline.
#[derive(Debug)]
pub struct PositionBegin;

/// After the fetch engine.
#[derive(Debug)]
pub struct PositionFetch;

/// After the switch engine.
#[derive(Debug)]
pub struct PositionSwitch;

/// After the switch engine's collect engine (32-byte packet normalized).
#[derive(Debug)]
pub struct PositionCollect;

/// After the contraction engine.
#[derive(Debug)]
pub struct PositionContraction;

/// After the vector engine (vector_final).
#[derive(Debug)]
pub struct PositionVectorFinal;

/// After the cast engine.
#[derive(Debug)]
pub struct PositionCast;

/// After the transpose engine.
#[derive(Debug)]
pub struct PositionTranspose;

impl Position for PositionBegin {}
impl Position for PositionFetch {}
impl Position for PositionSwitch {}
impl Position for PositionCollect {}
impl Position for PositionContraction {}
impl Position for PositionVectorFinal {}
impl Position for PositionCast {}
impl Position for PositionTranspose {}

/// Tensor streamed through the pipeline.
#[derive(Debug)]
pub struct StreamTensor<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Packet>>>>>,
    _position: PhantomData<P>,
}

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    StreamTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

    /// Creates a new stream tensor.
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping>) -> Self {
        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

/// Tensor streamed after the beginning.
pub type BeginTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionBegin, D, Chip, Cluster, Slice, Time, Packet>;

/// Tensor streamed after the fetch engine.
pub type FetchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionFetch, D, Chip, Cluster, Slice, Time, Packet>;

/// Tensor streamed after the switch engine.
pub type SwitchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionSwitch, D, Chip, Cluster, Slice, Time, Packet>;

/// Tensor after collect engine: packet is exactly 32 bytes (one flit).
pub type CollectTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionCollect, D, Chip, Cluster, Slice, Time, Packet>;

/// Pair of aligned tensors ready for contraction (Feed Buffer + TRF Sequencer output).
#[derive(Debug)]
pub struct AlignedPair<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Time: M, Packet: M> {
    ctx: &'l mut TuContext<{ T }>,
    lhs: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Row, Pair<Time, Packet>>>>>>,
    trf: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Row, Pair<Time, Packet>>>>>>,
}

/// Intermediate tensor after contraction (LAT reduce within Packet),
/// before accumulation (accumulator reduce across Time).
#[derive(Debug)]
pub struct ContractionTensor<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Time: M, Packet: M> {
    ctx: &'l mut TuContext<{ T }>,
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Row, Pair<Time, Packet>>>>>>,
}

/// Tensor streamed after the contraction engine.
pub type AccumulationTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionContraction, D, Chip, Cluster, Slice, Time, Packet>;

/// Tensor after the vector engine (after `vector_final()`).
pub type VectorFinalTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionVectorFinal, D, Chip, Cluster, Slice, Time, Packet>;

/// Tensor streamed after the cast engine.
pub type CastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionCast, D, Chip, Cluster, Slice, Time, Packet>;

/// Tensor streamed after the transpose engine.
pub type TransposeTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet> =
    StreamTensor<'l, { T }, PositionTranspose, D, Chip, Cluster, Slice, Time, Packet>;

// ANCHOR: fetch_impl
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    BeginTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs fetch operation to create a fetched tensor.
    #[primitive(BeginTensor::fetch)]
    pub fn fetch<D2: Scalar, Time2: M, Packet2: M>(self) -> FetchTensor<'l, T, D2, Chip, Cluster, Slice, Time2, Packet2>
    where
        D: FetchCast<D2>,
    {
        assert_eq!(Cluster::SIZE, 2, "Cluster size must be 2, got {}", Cluster::SIZE);
        assert_eq!(Slice::SIZE, 256, "Slice size must be 256, got {}", Slice::SIZE);
        let packet_bytes = D2::size_in_bytes_from_length(Packet2::SIZE);
        assert_eq!(
            packet_bytes % FETCH_ALIGN_BYTES,
            0,
            "Fetch output packet must be {FETCH_ALIGN_BYTES}-byte aligned, got {packet_bytes} bytes.",
        );
        FetchTensor::new(self.ctx, self.inner.map(|v| v.map(|v| v.cast())).transpose(true))
    }
}
// ANCHOR_END: fetch_impl

/// Creates a CastTensor after validating cast engine constraints.
///
/// The cast engine operates on a single 32-byte flit. The input packet must
/// be exactly one flit (32 bytes). After casting, the output packet is padded
/// back to 32 bytes. Time passes through unchanged.
///
/// Constraints checked:
/// 1. Input packet must be exactly one flit (32 bytes).
/// 2. Output packet must be exactly one flit (32 bytes).
/// 3. The data terms must match (only padding differs).
pub(crate) fn cast_stream<
    'l,
    const T: Tu,
    D: VeScalar + Cast<D2>,
    D2: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    InPacket: M,
    OutPacket: M,
>(
    ctx: &'l mut TuContext<{ T }>,
    inner: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Time }, { InPacket }]>,
) -> CastTensor<'l, T, D2, Chip, Cluster, Slice, Time, OutPacket> {
    // Input packet must be exactly one flit.
    assert_eq!(
        D::size_in_bytes_from_length(InPacket::SIZE),
        FLIT_BYTES,
        "Cast input packet must be exactly {FLIT_BYTES} bytes (one flit): \
         {} elements = {} bytes",
        InPacket::SIZE,
        D::size_in_bytes_from_length(InPacket::SIZE),
    );

    let out_flit_elements = D2::length_from_bytes(FLIT_BYTES);

    // Cast elements and pad to 32 bytes.
    let in_packet = InPacket::to_value().factorize();
    let expected_packet = in_packet.pad(out_flit_elements).normalize();

    // Output packet must be exactly one flit.
    let out_packet = OutPacket::to_value().factorize();
    assert_eq!(
        D2::size_in_bytes_from_length(OutPacket::SIZE),
        FLIT_BYTES,
        "Cast output packet must be exactly {FLIT_BYTES} bytes (one flit). \
         Expected: {expected_packet}, got: {out_packet}",
    );
    assert_eq!(
        expected_packet, out_packet,
        "Cast packet mismatch. Expected: {expected_packet}, got: {out_packet}",
    );

    CastTensor::new(ctx, inner.map(|v| v.map(|v| v.cast())).transpose(false))
}

/// Configuration for the `switch` operation.
#[derive(Debug, Clone)]
pub enum SwitchConfig {
    /// Replicates data across slices along slice dimensions 0 and 1.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | tile | tile\]
    /// Time:  \[time1 | time0\] to \[time1 | slice1 | time0 | slice0\]
    Broadcast01 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
        /// Time dimension 0 size.
        time0: usize,
    },
    /// Replicates data across slices along slice dimension 1.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | tile | slice0\]
    /// Time:  \[time0\] to \[time0 | slice1\]
    Broadcast1 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// Swaps slice1 and slice0 dimensions in the slice dimension. Time is unchanged.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | slice0 | slice1\]
    Transpose {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
    /// Swaps and transposes between the slice and time dimensions.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | time1 | slice0\]
    /// Time:  \[time2 | time1 | time0\] to \[time2 | time0 | slice1\]
    InterTranspose {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
        /// Time dimension 0 size.
        time0: usize,
    },
    /// Routes data across slices using a custom snoop bitmap.
    /// The bitmap is computed by the compiler from the input shape and topology
    /// parameters.
    CustomBroadcast {
        /// Ring group size for the custom routing.
        ring_size: usize,
    },
    /// Swaps slice1 and slice0 dimensions in the slice dimension and replicates
    /// data acorss slices along slice dimension 0.  The behavior is equivalent
    /// to applying `Transpose` and `Broadcast1` at once.
    /// Slice: \[slice2 | slice1 | slice0\] to \[slice2 | tile | slice1\]
    /// Time:  \[time0\] to \[time0 | slice0\]
    TransposedBroadcast1 {
        /// Slice dimension 1 size.
        slice1: usize,
        /// Slice dimension 0 size.
        slice0: usize,
    },
}

/// Gathers all symbols present in a mapping.
fn extract_symbols(mapping: &Mapping) -> HashSet<Ident> {
    let mut symbols = HashSet::new();
    let mapping = mapping.factorize();
    let mut stack = mapping.clone().into_inner();
    while let Some(factor) = stack.pop() {
        if let Factor::Term { inner, .. } = factor {
            match &inner.inner {
                Atom::Symbol { symbol, .. } => {
                    symbols.insert(*symbol);
                }
                Atom::Composite(inner) => {
                    stack.extend(RBox::into_inner(inner.clone()).into_inner());
                }
            }
        }
    }
    symbols
}

/// Validates switch engine constraints:
/// 1. Switch input and output slice sizes must match.
///
/// Delegates to [`SwitchConfig::verify`] for topology-specific checks.
fn verify_switch<InSlice: M, InTime: M, OutSlice: M, OutTime: M>(config: &SwitchConfig) {
    assert_eq!(
        InSlice::SIZE,
        OutSlice::SIZE,
        "Switch input and output slice sizes must match, got {} and {}",
        InSlice::SIZE,
        OutSlice::SIZE
    );
    config.verify::<InSlice, InTime, OutSlice, OutTime>();
}

impl SwitchConfig {
    /// Verifies that the switch configuration is compatible with the provided
    /// input and output slice/time mappings.
    pub fn verify<InSlice: M, InTime: M, OutSlice: M, OutTime: M>(&self) {
        match self {
            SwitchConfig::Broadcast01 { slice1, slice0, time0 } => {
                assert!(
                    [slice1, slice0, time0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );
                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );
                assert_eq!(InTime::SIZE % time0, 0, "InTime::SIZE must be divisible by time0");

                // m![InSlice / (slice1 * slice0)] = m![OutSlice / (slice1 * slice0)]
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );

                // Broadcast axes in OutSlice should be new.
                if *slice1 * *slice0 > 1 {
                    let out_slice_broadcast_symbols = extract_symbols(&Mapping::Modulo {
                        inner: RBox::new(OutSlice::to_value()),
                        modulo: *slice1 * *slice0,
                    });

                    let mut symbols = HashSet::new();
                    symbols.extend(extract_symbols(&InSlice::to_value()));
                    symbols.extend(extract_symbols(&InTime::to_value()));
                    assert_eq!(
                        out_slice_broadcast_symbols.intersection(&symbols).count(),
                        0,
                        "OutSlice broadcast axes must be new axes"
                    );
                }

                let mut expected_out_time = Mapping::Identity
                    // time1 = m![InTime / time0]
                    .pair(Mapping::Stride {
                        inner: RBox::new(InTime::to_value()),
                        stride: *time0,
                    })
                    // slice1 = m![InSlice / slice0 % slice1]
                    .pair(Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(InSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    });
                if *time0 > 1 {
                    // time0 = m![InTime % time0]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(InTime::to_value()),
                        modulo: *time0,
                    })
                }
                if *slice0 > 1 {
                    // slice0 = m![InSlice % slice0]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(InSlice::to_value()),
                        modulo: *slice0,
                    })
                }

                // OutTime must match [time1, slice1, time0, slice0]
                assert_eq!(
                    expected_out_time.factorize(),
                    OutTime::to_value().factorize(),
                    "OutTime does not match expected layout: [time1, slice1, time0, slice0]"
                );
            }

            SwitchConfig::Broadcast1 { slice1, slice0 } => {
                assert!(
                    [slice1, slice0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );
                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );

                // m![InSlice / (slice1 * slice0)] = m![OutSlice / (tile * slice0)]
                // Since tile has the same size as slice1, this is: m![InSlice / (slice1 * slice0)] = m![OutSlice / (slice1 * slice0)]
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );

                // Broadcast axes in OutSlice should be new.
                if *slice1 > 1 {
                    let out_slice_broadcast_symbols = extract_symbols(&Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(OutSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    });

                    let mut symbols = HashSet::new();
                    symbols.extend(extract_symbols(&InSlice::to_value()));
                    symbols.extend(extract_symbols(&InTime::to_value()));
                    assert_eq!(
                        out_slice_broadcast_symbols.intersection(&symbols).count(),
                        0,
                        "OutSlice broadcast axes must be new axes"
                    );
                }

                // slice0 is preserved at the innermost part of OutSlice
                if *slice0 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(InSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        "OutSlice must preserve slice0 from InSlice",
                    );
                }

                // OutTime must match [time0, slice1]
                let mut expected_out_time = Mapping::Identity.pair(InTime::to_value());
                if *slice1 > 1 {
                    // slice1 = m![InSlice / slice0 % slice1]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(InSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    });
                }

                assert_eq!(
                    expected_out_time.factorize(),
                    OutTime::to_value().factorize(),
                    "OutTime does not match expected layout: [time0, slice1]"
                );
            }

            SwitchConfig::Transpose { slice1, slice0 } => {
                assert!(
                    [slice1, slice0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );

                // Time can be padded, but padding must not change the size.
                assert_eq!(
                    InTime::SIZE,
                    OutTime::SIZE,
                    "Input and output time dimensions must have the same size"
                );
                assert_eq!(
                    InTime::to_value().factorize().remove_padding(),
                    OutTime::to_value().factorize().remove_padding(),
                    "Input and output time dimensions must match (excluding padding)"
                );

                // OutSlice must match [slice2, slice0, slice1] (slice1 and slice0 are swapped)
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .pair(Mapping::Modulo {
                        inner: RBox::new(InSlice::to_value()),
                        modulo: *slice0,
                    })
                    .pair(Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(InSlice::to_value()),
                            stride: *slice0,
                        }),
                        modulo: *slice1,
                    })
                    .factorize(),
                    OutSlice::to_value().factorize(),
                    "OutSlice does not match expected layout: [slice2, slice0, slice1]"
                );
            }

            SwitchConfig::InterTranspose { slice1, slice0, time0 } => {
                assert!(
                    [slice1, slice0, time0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );

                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );
                assert_eq!(
                    InTime::SIZE % (slice1 * time0),
                    0,
                    "InTime::SIZE must be divisible by (slice1 * time0)"
                );

                let slice2 = InSlice::SIZE / (slice1 * slice0);
                let time2 = InTime::SIZE / (slice1 * time0);

                assert_eq!(slice2 * slice1 * slice0, 256, "All dimensions should multiply to 256");
                assert_eq!(
                    time2 * slice1 * time0,
                    InTime::SIZE,
                    "time2 * slice1 * time0 must equal InTime::SIZE"
                );

                // InSlice  = [slice2, slice1,  slice0]
                // OutSlice = [slice2, time1, slice0]
                // slice2 and slice0 are preserved
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );
                if *slice0 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(InSlice::to_value()),
                            modulo: *slice0,
                        }
                        .factorize(),
                        "OutSlice must preserve slice0 from InSlice",
                    );
                }

                // InTime   = [time2,  time1, time0]
                // OutSlice = [slice2, time1, slice0]
                // time1 comes from InTime
                if *slice1 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(OutSlice::to_value()),
                                stride: *slice0,
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(InTime::to_value()),
                                stride: *time0,
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        "OutSlice time1 must come from InTime"
                    );
                }

                // InTime  = [time2, time1, time0]
                // OutTime = [time2, time0,  slice1]
                // time2 is preserved
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(OutTime::to_value()),
                        stride: *slice1 * time0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(InTime::to_value()),
                        stride: *time0 * slice1,
                    }
                    .factorize(),
                    "OutTime must preserve 'time2' from InTime",
                );
                // time0 is moved from InTime to OutTime
                if *time0 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(OutTime::to_value()),
                                stride: *slice1,
                            }),
                            modulo: *time0,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(InTime::to_value()),
                            modulo: *time0,
                        }
                        .factorize(),
                        "OutTime must preserve 'time0' from InTime"
                    );
                }
                // InSlice = [slice2, slice1, slice0]
                // OutTime = [time2,  time0,   slice1]
                // slice1 in OutTime matches the one in InSlice
                if *slice1 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutTime::to_value()),
                            modulo: *slice1,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(InSlice::to_value()),
                                stride: *slice0,
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        "OutTime must preserve 'slice1' from InSlice"
                    );
                }
            }

            // Custom topologies allow:
            // 1. Arbitrary slice permutations in Slice.
            // 2. Slice to Time broadcasts with slicing.

            // Constraints checked:
            // 1. Broadcast axes must use new unique axes (not in input `Slice` or `Time`).
            // 2. Broadcast axes should not be padded.
            // 3. Outer portion of output `Time` must match input `Time`.
            // 4. Axes moving from `Slice` to `Time` appear at the innermost output `Time` positions.
            // 5. Axes moving from `Slice` to `Time` must preserve their relative order from input `Slice`.
            // 6. Ring size must match the outermost non-directcast boundary and be a power of 2.
            SwitchConfig::CustomBroadcast { ring_size } => {
                assert!(
                    ring_size.is_power_of_two(),
                    "Switch ring size must be a power of 2, got {ring_size}"
                );

                let slice = InSlice::to_value().factorize();
                let out_slice = OutSlice::to_value().factorize();
                let time = InTime::to_value().factorize();
                let out_time = OutTime::to_value().factorize();

                // Identify broadcast axes. Broadcast axes use new axes, not
                // present in InSlice.
                let furiosa_mapping::Division {
                    dividend_residue,
                    divisor_residue,
                    division_terms,
                    ..
                } = out_slice.clone().divide_strict(slice.clone());
                assert!(
                    dividend_residue
                        .clone()
                        .divide_strict(slice.clone().mul(time.clone()))
                        .division_terms()
                        .is_empty(),
                    "Switch broadcast axes must be new axes (not present in input Slice or Time)."
                );

                // Each broadcast axis must be used exactly once in OutSlice.
                // `terms_with_stride` counts all terms (including duplicates),
                // `idents` returns unique axes. A mismatch means an axis
                // appears more than once.
                assert_eq!(
                    dividend_residue.terms_with_stride().len(),
                    dividend_residue.idents().len(),
                    "Switch broadcast axes must each be used exactly once in OutSlice"
                );

                // Broadcast axes must not have padding.
                let factors = out_slice.factors();
                for (i, factor) in factors.iter().enumerate() {
                    if let Factor::Term { inner, .. } = factor
                        && !division_terms.iter().any(|d| d.term.inner == inner.inner)
                        && matches!(factors.get(i + 1), Some(Factor::Padding { .. }))
                    {
                        panic!("Switch broadcast axis {inner} in output Slice must not be padded.");
                    }
                }

                // OutTime = [InTime (outer) | moved axes (inner)].
                let Tuple2(outer, inner) =
                    out_time.split_at(exact_div(OutTime::SIZE, InTime::SIZE).unwrap_or_else(|| {
                        panic!(
                            "Input Time size ({}) does not divide Output Time size ({})",
                            InTime::SIZE,
                            OutTime::SIZE
                        )
                    }));
                assert_eq!(
                    outer, time,
                    "Switch axes moving from input slice to output time must be at the output time innermost positions. \
                     Expected outer portion of output time to be {time}, got {outer}"
                );

                let broadcast_divisions = divisor_residue
                    .clone()
                    .divide_relaxed(inner.clone())
                    .exact()
                    .unwrap_or_else(|_| {
                        panic!(
                            "Switch broadcast axes in output time must come from input slice. \
                            Input Slice: {slice}, inner part of output Time: {inner}"
                        )
                    })
                    .division_terms;

                // Axes moving from Slice to Time must preserve their relative
                // order from input Slice.
                for window in broadcast_divisions.windows(2) {
                    assert!(
                        window[0].divisor_stride > window[1].divisor_stride,
                        "Switch axes moving from input Slice to output Time must preserve their relative order from input Slice. \
                         {} (outer in input Slice) appears inner to {} in output Time",
                        window[0].term,
                        window[1].term
                    );
                }

                // Find the span of the outermost non-directcast axis in the
                // input Slice.
                let mut max_non_dc = 0;
                for d in &division_terms {
                    if d.dividend_stride != d.divisor_stride {
                        max_non_dc = max_non_dc.max(d.divisor_stride * d.resize);
                    }
                }
                for t in &divisor_residue.terms_with_stride() {
                    max_non_dc = max_non_dc.max(t.stride * t.resize);
                }

                // Find the directcast axis with the smallest stride that is at
                // or above `max_non_dc`.
                let mut expected_ring_size = InSlice::SIZE;
                for d in &division_terms {
                    if d.dividend_stride == d.divisor_stride && d.divisor_stride >= max_non_dc {
                        expected_ring_size = expected_ring_size.min(d.divisor_stride);
                    }
                }
                assert_eq!(
                    *ring_size, expected_ring_size,
                    "Switch ring size mismatch. Expected {expected_ring_size}, got {ring_size}"
                );
            }

            SwitchConfig::TransposedBroadcast1 { slice1, slice0 } => {
                assert!(
                    [slice1, slice0].iter().all(|&x| *x > 0),
                    "All dimensions must be greater than 0"
                );

                assert_eq!(
                    InSlice::SIZE % (slice1 * slice0),
                    0,
                    "InSlice::SIZE must be divisible by (slice1 * slice0)"
                );

                // m![InSlice / (slice1 * slice0)] = m![OutSlice / (tile * slice1)]
                // Since tile has the same size as slice0, this is: m![InSlice / (slice1 * slice0)] = m![OutSlice / (tile * slice1)]
                assert_eq!(
                    Mapping::Stride {
                        inner: RBox::new(InSlice::to_value()),
                        stride: *slice1 * *slice0,
                    }
                    .factorize(),
                    Mapping::Stride {
                        inner: RBox::new(OutSlice::to_value()),
                        stride: *slice0 * *slice1,
                    }
                    .factorize(),
                    "OutSlice must preserve slice2 from InSlice",
                );

                // Broadcast axes in OutSlice should be new.
                if *slice0 > 1 {
                    let out_slice_broadcast_symbols = extract_symbols(&Mapping::Modulo {
                        inner: RBox::new(Mapping::Stride {
                            inner: RBox::new(OutSlice::to_value()),
                            stride: *slice1,
                        }),
                        modulo: *slice0,
                    });

                    let mut symbols = HashSet::new();
                    symbols.extend(extract_symbols(&InSlice::to_value()));
                    symbols.extend(extract_symbols(&InTime::to_value()));
                    assert_eq!(
                        out_slice_broadcast_symbols.intersection(&symbols).count(),
                        0,
                        "OutSlice broadcast axes must be new axes"
                    );
                }

                // slice1 is preserved at the innermost part of OutSlice
                if *slice1 > 1 {
                    assert_eq!(
                        Mapping::Modulo {
                            inner: RBox::new(OutSlice::to_value()),
                            modulo: *slice1,
                        }
                        .factorize(),
                        Mapping::Modulo {
                            inner: RBox::new(Mapping::Stride {
                                inner: RBox::new(InSlice::to_value()),
                                stride: *slice0
                            }),
                            modulo: *slice1,
                        }
                        .factorize(),
                        "OutSlice must preserve slice1 from InSlice",
                    );
                }

                // OutTime must match [time0, slice0]
                let mut expected_out_time = Mapping::Identity.pair(InTime::to_value());
                if *slice0 > 1 {
                    // slice0 = m![InSlice % slice0]
                    expected_out_time = expected_out_time.pair(Mapping::Modulo {
                        inner: RBox::new(InSlice::to_value()),
                        modulo: *slice0,
                    });
                }

                assert_eq!(
                    expected_out_time.factorize(),
                    OutTime::to_value().factorize(),
                    "OutTime does not match expected layout: [time0, slice0]"
                );
            }
        }
    }
}

/// Validates collect engine constraints: normalizes packet to exactly one flit (32 bytes).
///
/// Pads the input packet to flit-aligned boundary, then splits:
/// - Inner 32 bytes → Packet2 (one flit)
/// - Outer flit portion → absorbed into Time2
///
/// For packets already ≤ 32 bytes, only padding is added.
fn verify_collect<D: Scalar, Time: M, Packet: M, Time2: M, Packet2: M>() {
    let in_packet_bytes = D::size_in_bytes_from_length(Packet::SIZE);
    let aligned_bytes = align_up(in_packet_bytes, FLIT_BYTES);
    let flit_elements = D::length_from_bytes(FLIT_BYTES);

    // Output packet must be exactly one flit.
    assert_eq!(
        D::size_in_bytes_from_length(Packet2::SIZE),
        FLIT_BYTES,
        "Collect output packet must be exactly {FLIT_BYTES} bytes (one flit)."
    );

    // Pad input packet to flit-aligned boundary, then split at flit boundary.
    let in_factorized = Packet::to_value().factorize();
    let padded = in_factorized.pad(D::length_from_bytes(aligned_bytes));
    let Tuple2(in_outer, in_flit) = padded.split_at(flit_elements);

    // Output packet = inner flit.
    let expected_packet = in_flit.normalize();
    let out_packet = Packet2::to_value().factorize();
    assert_eq!(
        expected_packet, out_packet,
        "Collect packet mismatch. Expected: {expected_packet}, got: {out_packet}"
    );

    // Time2 = Time × outer flit portion.
    let expected_time = Time::to_value().factorize().mul(in_outer).normalize();
    let out_time = Time2::to_value().factorize();
    assert_eq!(
        expected_time, out_time,
        "Collect time mismatch. Expected: {expected_time}, got: {out_time}"
    );
}

/// Validates hardware constraints for the transpose engine.
///
/// Constraints checked:
/// 1. `Packet` and `OutPacket` must be 32 bytes
/// 2. `in_rows` * sizeof(D) <= 8 bytes
/// 3. `in_cols` must be 8, 16, or 32 (4-bit: 16 or 32 only)
/// 4. `out_rows` <= `in_cols`
fn verify_transpose<D: Scalar, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    // Packet must be TRANSPOSE_INPUT_BYTES bytes.
    let packet_bytes = D::size_in_bytes_from_length(Packet::SIZE);
    assert_eq!(
        packet_bytes, TRANSPOSE_INPUT_BYTES,
        "Transpose input packet must be {TRANSPOSE_INPUT_BYTES} bytes, got {packet_bytes}"
    );

    // OutPacket must be TRANSPOSE_OUTPUT_BYTES bytes.
    let out_packet_bytes = D::size_in_bytes_from_length(OutPacket::SIZE);
    assert_eq!(
        out_packet_bytes, TRANSPOSE_OUTPUT_BYTES,
        "Transpose output packet must be {TRANSPOSE_OUTPUT_BYTES} bytes, got {out_packet_bytes}",
    );

    // OutPacket is `[in_rows # padding]`.
    // `in_rows` * sizeof(D) <= 8 bytes
    let in_rows = OutPacket::to_value().factorize().remove_padding();
    let in_rows_bytes = D::size_in_bytes_from_length(in_rows.size());
    assert!(
        in_rows_bytes <= TRANSPOSE_MAX_IN_ROWS_BYTES,
        "Transpose `in_rows` must be <= {TRANSPOSE_MAX_IN_ROWS_BYTES} bytes, got {in_rows_bytes}"
    );

    // `Time = [..., in_rows, packets_per_col]`
    // Check that `in_rows` matches OutPacket `in_rows`.
    let time = Time::to_value().factorize();
    let division_terms = time
        .clone()
        .divide_relaxed(in_rows.clone())
        .exact()
        .unwrap_or_else(|_| panic!("Transpose `in_rows` ({in_rows}) must be present in the input Time ({time})"))
        .division_terms;

    // `in_cols` = `packets_per_col` * `elements_per_packet` must be in {8, 16, 32} (4-bit: {16, 32})
    let Tuple2(time_outer, packets_per_col) = time.split_at(division_terms[0].dividend_stride);
    let Tuple2(time_outer, _) = time_outer.split_at(in_rows.size());
    let (elements_per_packet, valid_in_cols) = match D::BITS {
        4 => (TRANSPOSE_ELEMENTS_PER_PACKET_4BIT, TRANSPOSE_VALID_IN_COLS_4BIT),
        _ => (TRANSPOSE_ELEMENTS_PER_PACKET_NON_4BIT, TRANSPOSE_VALID_IN_COLS),
    };
    let in_cols = packets_per_col.size() * elements_per_packet;
    assert!(
        valid_in_cols.contains(&in_cols),
        "Transpose `in_cols` size ({in_cols}) must be one of {valid_in_cols:?} for {}-bit type",
        D::BITS,
    );

    let elements_per_packet = Packet::to_value().factorize().pad(elements_per_packet);
    let out_time = OutTime::to_value().factorize();

    // `OutTime = [time_outer, packets_per_col, elements_per_packet]`.
    // `elements_per_packet` may be sliced: `[in_cols x in_rows] → [out_rows x in_rows]`.
    // Make sure only padding is removed.
    let Tuple2(out_time_outer, out_time_elements_per_packet) = if packets_per_col.size() > 1 {
        let division_terms = out_time
            .clone()
            .divide_relaxed(packets_per_col.clone())
            .exact()
            .unwrap_or_else(|_| {
                panic!("Transpose `packets_per_col` ({packets_per_col}) not found in OutTime ({out_time})")
            })
            .division_terms;
        let Tuple2(outer, num_elems) = out_time.split_at(division_terms[0].dividend_stride);
        let Tuple2(outer, _) = outer.split_at(packets_per_col.size());
        Tuple2(outer, num_elems)
    } else {
        let out_rows = OutTime::SIZE / time_outer.size();
        out_time.split_at(out_rows)
    };
    let out_rows = packets_per_col
        .clone()
        .mul(out_time_elements_per_packet.clone())
        .normalize();
    let in_cols = packets_per_col.mul(elements_per_packet.clone()).normalize();
    assert!(
        out_rows.size() <= in_cols.size(),
        "Transpose `out_rows` ({}) must be <= `in_cols` ({})",
        out_rows.size(),
        in_cols.size(),
    );
    assert_eq!(
        out_time_elements_per_packet.remove_padding(),
        elements_per_packet.remove_padding(),
        "Transpose `out_rows` ({out_rows}) must match `in_cols` ({in_cols}) (excluding padding)",
    );

    // Outer Time axes should match outer OutTime axes.
    assert_eq!(
        time_outer, out_time_outer,
        "Transpose time mismatch: expected outer OutTime to be {time_outer}, got ({out_time_outer})"
    );
}

// ANCHOR: switch_impl
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    FetchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs switching operation to create a switched tensor.
    ///
    /// Applies switching network routing only. The packet passes through
    /// unchanged — no padding, no reshaping. Use [`SwitchTensor::collect`]
    /// afterwards to normalize the packet to flit-sized chunks.
    #[primitive(FetchTensor::switch)]
    pub fn switch<Slice2: M, Time2: M>(
        self,
        config: SwitchConfig,
    ) -> SwitchTensor<'l, T, D, Chip, Cluster, Slice2, Time2, Packet> {
        verify_switch::<Slice, Time, Slice2, Time2>(&config);
        SwitchTensor::new(self.ctx, self.inner.transpose(true))
    }

    /// Skips the switching network and goes directly to collect.
    ///
    /// Slice and Time are preserved from fetch; only the packet is normalized
    /// to flit-sized chunks.
    #[primitive(FetchTensor::collect)]
    pub fn collect<Time2: M, Packet2: M>(self) -> CollectTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2> {
        verify_collect::<D, Time, Packet, Time2, Packet2>();
        CollectTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: switch_impl

// ANCHOR: collect_impl
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    SwitchTensor<'l, { T }, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Normalizes packet to exactly 32 bytes (one flit).
    ///
    /// Pads to flit-aligned boundary, then splits: inner 32 bytes become Packet2,
    /// outer flit portion is absorbed into Time2.
    /// For packets already ≤ 32 bytes, only padding is added.
    #[primitive(SwitchTensor::collect)]
    pub fn collect<Time2: M, Packet2: M>(self) -> CollectTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2> {
        verify_collect::<D, Time, Packet, Time2, Packet2>();
        CollectTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: collect_impl

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    CollectTensor<'l, { T }, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Stores to the tensor register file.
    #[primitive(CollectTensor::to_trf)]
    pub fn to_trf<Row: M, Element: M>(self, address: TrfAddress) -> TrfTensor<D, Chip, Cluster, Slice, Row, Element> {
        assert!(
            [1, 2, 4, 8].contains(&Row::SIZE),
            "Row::SIZE must be 1, 2, 4, or 8, got {}",
            Row::SIZE
        );

        // Trf data should fit in the register file.
        let capacity = address.capacity();
        let total_trf_bytes = D::size_in_bytes_from_length(Row::SIZE * Element::SIZE);
        assert!(
            total_trf_bytes <= capacity,
            "TRF data ({} bytes = {} rows x {} bytes) exceeds register file capacity ({} bytes for {})",
            total_trf_bytes,
            Row::SIZE,
            D::size_in_bytes_from_length(Element::SIZE),
            capacity,
            address,
        );

        // |row| <= |time|
        assert!(
            Row::SIZE <= Time::SIZE,
            "Row::SIZE must be <= Time::SIZE, got {} > {}",
            Row::SIZE,
            Time::SIZE,
        );

        // [time_outer] = [Row]
        let time = Time::to_value().factorize();
        let Tuple2(time_outer, time_inner) = time.split_at(
            exact_div(Time::SIZE, Row::SIZE)
                .unwrap_or_else(|| panic!("Row::SIZE ({}) does not divide Time::SIZE ({})", Row::SIZE, Time::SIZE)),
        );
        let row = Row::to_value().factorize();
        assert_eq!(
            time_outer, row,
            "`to_trf` row mismatch: time_outer != Row: {time_outer} != {row}",
        );

        // [time_inner, Packet] = [Element]
        let expected_element = time_inner.mul(Packet::to_value().factorize()).normalize();
        let element = Element::to_value().factorize();
        assert_eq!(
            expected_element, element,
            "`to_trf` element mismatch: [time_inner, Packet] != Element: {expected_element} != {element}",
        );

        TrfTensor::new(self.inner.transpose(false), address)
    }

    /// Stores to the vector register file.
    #[primitive(CollectTensor::to_vrf)]
    pub fn to_vrf<Element2: M>(self, address: Address) -> VrfTensor<D, Chip, Cluster, Slice, Element2>
    where
        D: VeScalar,
    {
        VrfTensor::new(self.inner.transpose(false), address)
    }

    // ANCHOR: collect_vector_init
    /// Initializes Vector Engine processing for this tensor.
    #[primitive(CollectTensor::vector_init)]
    pub fn vector_init(self) -> VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
    where
        D: VeScalar,
        // ANCHOR_END: collect_vector_init
    {
        VectorInitTensor::new(self.ctx, self.inner)
    }

    /// Aligns LHS stream (via Feed Buffer) and RHS TRF (via TRF Sequencer) to computation shape.
    #[primitive(CollectTensor::align)]
    pub fn align<OutTime: M, OutPacket: M, Row: M, TrfElement: M>(
        self,
        trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Row, TrfElement>,
    ) -> AlignedPair<'l, { T }, D, Chip, Cluster, Slice, Row, OutTime, OutPacket>
    where
        Chip: M,
        Cluster: M,
        Slice: M,
    {
        assert!([1, 2, 4, 8].contains(&Row::SIZE), "Row::SIZE should be 1, 2, 4, or 8");

        let out_packet_size = D::size_in_bytes_from_length(OutPacket::SIZE);
        assert_eq!(
            out_packet_size, 64,
            "OutPacket must be 64 bytes, got {out_packet_size} bytes"
        );

        // Inner flit of OutPacket must match input Packet.
        let flit_elements = D::length_from_bytes(FLIT_BYTES);
        let Tuple2(out_packet_outer, out_packet_inner) = OutPacket::to_value().factorize().split_at(flit_elements);
        let out_packet_inner = out_packet_inner.normalize();
        let expected_packet = Packet::to_value().factorize();
        assert_eq!(
            out_packet_inner, expected_packet,
            "`align` packet mismatch: inner flit of OutPacket != Packet: {out_packet_inner} != {expected_packet}",
        );

        // Time must equal OutTime * outer flit portion of OutPacket.
        // Padding is stripped for the collect_flits = 1 case.
        let expected_time = OutTime::to_value()
            .factorize()
            .mul(out_packet_outer.remove_padding())
            .normalize();
        let input_time = Time::to_value().factorize();

        // Time broadcast axes are the innermost in `OutTime`.
        // These are present in TRF, but absent in the input data.
        let tiling_size = expected_time.size() / input_time.size();
        let align_div = expected_time
            .divide_relaxed(input_time.clone())
            .exact()
            .expect("`align`: Time does not divide OutTime");
        let tiling = align_div.dividend_residue;
        let division_terms = align_div.division_terms;

        // Non-tiling axes must follow the same order in both mappings.
        assert!(
            division_terms
                .windows(2)
                .all(|w| w[0].divisor_stride > w[1].divisor_stride),
            "`align`: Time axes are reordered in OutTime"
        );

        // Tiling axes are the innermost axes in `OutTime`.
        assert!(
            division_terms.iter().all(|d| d.dividend_stride >= tiling_size),
            "`align`: tiling axes must be innermost in OutTime"
        );

        if tiling.size() > 1 {
            let trf_element = TrfElement::to_value().factorize();
            assert!(
                trf_element.clone().divide_relaxed(tiling.clone()).exact().is_ok(),
                "tiling axes must be present in TRF",
            );
        }

        let lhs = self.inner.transpose(true);
        let trf = trf_tensor.inner.transpose(true);

        AlignedPair {
            ctx: self.ctx,
            lhs,
            trf,
        }
    }

    /// Performs transpose operation.
    #[primitive(CollectTensor::transpose)]
    pub fn transpose<OutTime: M, OutPacket: M>(
        self,
    ) -> TransposeTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket> {
        verify_transpose::<D, Time, Packet, OutTime, OutPacket>();
        TransposeTensor::new(self.ctx, self.inner.transpose(false))
    }
}

// ANCHOR: cast_impl
impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M> StreamCast<D>
    for CollectTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    type CastOutput<D2: Scalar, OutPacket: M>
        = CastTensor<'l, T, D2, Chip, Cluster, Slice, Time, OutPacket>
    where
        D: Cast<D2>;

    #[primitive(CollectTensor::cast)]
    fn cast<D2: Scalar, OutPacket: M>(self) -> Self::CastOutput<D2, OutPacket>
    where
        D: Cast<D2>,
    {
        cast_stream(self.ctx, self.inner)
    }
}
// ANCHOR_END: cast_impl

/// Validates contraction.
///
/// Checks:
/// 1. `Packet` should be 64 bytes.
/// 2. `OutPacket::SIZE` should be a power of two and at most
///    `TEMPORAL_ACCUMULATOR_COLS` and be obtainable from `Packet` by splitting
///    at a power-of-two sized boundary.
fn verify_contract<D: Scalar, Packet: M, OutPacket: M>() {
    let packet_size = D::size_in_bytes_from_length(Packet::SIZE);
    assert_eq!(packet_size, 64, "Packet must be 64 bytes, got {packet_size} bytes");

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

/// Validates accumulation constraints on output shape and accumulator size.
///
/// Checks:
/// 1. `Packet::SIZE` must be at most 32 (`i32`/`f32`).
/// 2. `OutPacket::SIZE` must be 8 (`i32`/`f32`).
/// 3. Output factor composition matches retained (non-reduced) axes:
///    - Interleaved: `OutTime = [retained Time, retained Packet (may be sliced)]`, `OutPacket = [Row # 8]`.
///    - Sequential: `OutTime = [retained Time, Row, packet_outer]`, `OutPacket = [packet_inner # 8]`.
/// 4. Accumulator fits hardware limit (1024 elements):
///    - Interleaved: `inner_time * ReducedPacket <= 128` (since `align_up(Row, 8) = 8`).
///    - Sequential: `inner_time * Row * packet_outer <= 32` (since `align_up(ReducedPacket, 32) = 32`).
fn verify_accumulate<Row: M, Time: M, Packet: M, OutTime: M, OutPacket: M>(kind: AccumulationKind) {
    assert!(
        Packet::SIZE <= TEMPORAL_ACCUMULATOR_COLS,
        "accumulate: Packet::SIZE must be at most {TEMPORAL_ACCUMULATOR_COLS}, got {}",
        Packet::SIZE
    );
    assert_eq!(
        OutPacket::SIZE,
        ACCUMULATE_OUT_PACKET_ELEMENTS,
        "accumulate: OutPacket::SIZE must be {ACCUMULATE_OUT_PACKET_ELEMENTS}, got {}",
        OutPacket::SIZE
    );

    let time = Time::to_value().factorize();
    let packet = Packet::to_value().factorize().remove_padding();
    let out_time = OutTime::to_value().factorize();
    let out_packet = OutPacket::to_value().factorize();

    // Determine `outer_time` and `packet_outer_size` based on contraction kind.
    // `packet_outer_size` is 1 for Interleaved (no packet split into OutTime),
    // and the size of the outer packet portion for Sequential.
    let (outer_time, packet_outer_size) = match kind {
        AccumulationKind::Interleaved => {
            // `OutPacket = [Row # 8]`
            let expected_out_packet = Row::to_value().factorize().pad(ACCUMULATE_OUT_PACKET_ELEMENTS);
            assert_eq!(
                out_packet, expected_out_packet,
                "accumulate ({kind}): OutPacket mismatch. Expected: {expected_out_packet}, got: {out_packet}"
            );

            // `OutTime = [Time', Packet (may be sliced)]`
            // Search for the `Packet / Time'` boundary.
            let outer_time = (1..=out_time.size().min(packet.size()).min(TEMPORAL_ACCUMULATOR_COLS))
                .filter(|&split| {
                    out_time.size() % split == 0
                        // If `packet` is not the identity mapping, require it
                        // to be present in `OutTime`.
                        && (split > 1 || packet.size() == 1)
                        // `Time'::SIZE` <= `Time::SIZE`.
                        && out_time.size() / split <= time.size()
                })
                .find_map(|split| {
                    let Tuple2(outer_time, sliced_packet) = out_time.split_at(split);

                    // Slicing may only remove padding.
                    if sliced_packet != packet {
                        return None;
                    }

                    Some(outer_time)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "accumulate ({kind}): OutTime mismatch. \
                         Could not decompose OutTime {out_time} into [Time', Packet'] \
                         where Time' is {time} after temporal accumulation and Packet' is a truncation of {packet}"
                    )
                });
            (outer_time, 1)
        }
        AccumulationKind::Sequential => {
            // `OutTime   = [Time', Row, packet_outer]`
            // `OutPacket = [packet_inner # ACCUMULATE_OUT_PACKET_ELEMENTS]`
            //
            // `packet` is padded to the next multiple of ACCUMULATE_OUT_PACKET_ELEMENTS,
            // then split at ACCUMULATE_OUT_PACKET_ELEMENTS:
            // - `packet_inner` (ACCUMULATE_OUT_PACKET_ELEMENTS elements): becomes `OutPacket`
            // - `packet_outer`                                          : absorbed into `OutTime`
            let padded = packet
                .clone()
                .pad(align_up(packet.size(), ACCUMULATE_OUT_PACKET_ELEMENTS));
            let Tuple2(packet_outer, packet_inner) = padded.split_at(ACCUMULATE_OUT_PACKET_ELEMENTS);
            let packet_outer_size = packet_outer.size();

            // `OutPacket = [packet_inner # ACCUMULATE_OUT_PACKET_ELEMENTS]`.
            assert_eq!(
                packet_inner, out_packet,
                "accumulate ({kind}): OutPacket mismatch. Expected: {packet_inner}, got: {out_packet}"
            );

            // `OutTime` ends with `[Row, packet_outer]`
            let row_packet = Row::to_value().factorize().mul(packet_outer);
            let Tuple2(outer_time, inner_time) = out_time.split_at(row_packet.size());
            assert_eq!(
                inner_time, row_packet,
                "accumulate ({kind}): OutTime mismatch. Expected {row_packet}, got {inner_time}"
            );

            (outer_time, packet_outer_size)
        }
    };

    // The outer portion of `Time` should divide `Time`.
    // Some axes can be reduced in temporal accumulation.
    let division_terms = time
        .clone()
        .divide_relaxed(outer_time.clone())
        .exact()
        .unwrap_or_else(|_| {
            panic!("accumulate ({kind}): OutTime mismatch. Some axes present in Time are not present in Time': {time}, {outer_time}")
        })
        .division_terms;
    // Non-reduced axes must have their order preserved in `OutTime`.
    assert!(
        division_terms
            .windows(2)
            .all(|w| w[0].divisor_stride > w[1].divisor_stride),
        "accumulate ({kind}): OutTime axes must follow the same order as the Time axes"
    );

    // Each retained axis in `outer_time` must preserve its padding from `time`.
    // We store `padding_size / stride` per term (always exact since padding
    // aligns to stride boundaries) and verify that the stride boundaries between
    // consecutive retained axes in `outer_time` match the gaps produced by
    // padding in `time`.
    let mut time_padding_per_stride: HashMap<usize, usize> = HashMap::new();
    let factors = time.factors();
    let mut stride = 1;
    // Build `cumulative_stride` : `padding_per_stride` table for each term in `time`.
    // `m![A # 8, B # 4]` with `axes![A = 4, B = 2]` -> { 1: 8, 8: 4 }
    for (i, factor) in factors.iter().enumerate() {
        match factor {
            Factor::Term { resize, .. } => {
                time_padding_per_stride.insert(
                    stride,
                    if let Some(Factor::Padding { size, .. }) = factors.get(i + 1) {
                        size / stride
                    } else {
                        *resize
                    },
                );
                stride *= resize;
            }
            Factor::Padding { size, .. } => {
                stride = *size;
            }
        }
    }

    // Sort retained axes inner-to-outer by their position in `outer_time` (divisor).
    let mut sorted_divisions: Vec<&DivisionTerm> = division_terms.iter().collect();
    sorted_divisions.sort_by_key(|d| d.divisor_stride);

    // The first divisor should have a stride of 1.
    // This catches unexpected padding that creeps into `outer_time` when
    // `split_at` absorbs padding from an adjacent axis (e.g., `M # 8` in
    // `row_packet` leaking a leading padding of 2 into `outer_time`).
    if let Some(first) = sorted_divisions.first() {
        assert_eq!(
            first.divisor_stride, 1,
            "accumulate ({kind}): Padding mismatch. \
             OutTime {outer_time} has unexpected leading padding not present in Time {time}"
        );
    }

    // For each retained axis, its padding end must equal the start of the next retained axis.
    for (pos, d) in sorted_divisions.iter().enumerate() {
        let expected_end = d.divisor_stride
            * time_padding_per_stride
                .get(&d.dividend_stride)
                .copied()
                .unwrap_or(d.resize);
        let end = sorted_divisions
            .get(pos + 1)
            // The last term's padding ends at `outer_time.size()`.
            .map_or(outer_time.size(), |next| next.divisor_stride);
        assert_eq!(
            expected_end, end,
            "accumulate ({kind}): Padding mismatch. \
             Non-reduced axes in OutTime {outer_time} do not preserve padding from Time {time}"
        );
    }

    // Calculate axis size inner to outermost reduce.
    let padding_end = |d: &DivisionTerm| {
        d.dividend_stride
            * time_padding_per_stride
                .get(&d.dividend_stride)
                .copied()
                .unwrap_or(d.resize)
    };
    let inner_time = if division_terms.is_empty() {
        // Case 1: All axes reduced.
        1
    } else if padding_end(&division_terms[0]) < time.size() {
        // Case 2: The outermost axis was reduced.
        outer_time.size()
    } else {
        // Case 3: The outermost retained factor reaches the top of `Time`.
        // Walk outer-to-inner looking for the first gap between adjacent division terms.
        division_terms
            .windows(2)
            .find(|w| padding_end(&w[1]) != w[0].dividend_stride)
            .map_or(1, |w| w[0].divisor_stride)
    };

    // Check buffer limits
    match kind {
        AccumulationKind::Interleaved => {
            let buffer = inner_time * packet.size();
            assert!(
                buffer <= 1024 / ACCUMULATE_OUT_PACKET_ELEMENTS,
                "accumulate ({}): axes inner to reduce must be <= {} in size, got {}",
                kind,
                1024 / ACCUMULATE_OUT_PACKET_ELEMENTS,
                buffer
            );
        }
        AccumulationKind::Sequential => {
            let buffer = inner_time * Row::SIZE * packet_outer_size;
            assert!(
                buffer <= 1024 / TEMPORAL_ACCUMULATOR_COLS,
                "accumulate ({}): axes inner to reduce must be <= {} in size, got {}",
                kind,
                1024 / TEMPORAL_ACCUMULATOR_COLS,
                buffer
            );
        }
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Time: M, Packet: M>
    AlignedPair<'l, { T }, D, Chip, Cluster, Slice, Row, Time, Packet>
{
    /// Performs contraction (LAT reduce within Packet).
    /// Data type is widened during contraction: i4/i8 -> i32, f8/bf16 -> f32.
    #[primitive(AlignedPair::contract)]
    pub fn contract<OutPacket: M>(
        self,
    ) -> ContractionTensor<'l, { T }, <D as ContractionCast>::Output, Chip, Cluster, Slice, Row, Time, OutPacket>
    where
        D: ContractionCast + Cast<<D as ContractionCast>::Output>,
    {
        verify_contract::<D, Packet, OutPacket>();
        let lhs = self.lhs.map(|v| v.map(|v| v.cast()));
        let trf = self.trf.map(|v| v.map(|v| v.cast()));
        ContractionTensor {
            ctx: self.ctx,
            inner: lhs.zip_with(&trf, |a, b| a * b).reduce_add(),
        }
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Time: M, Packet: M>
    ContractionTensor<'l, { T }, D, Chip, Cluster, Slice, Row, Time, Packet>
{
    /// Performs accumulation (accumulator reduce across Time).
    #[primitive(ContractionTensor::accumulate)]
    pub fn accumulate<OutTime: M, OutPacket: M>(
        self,
        kind: AccumulationKind,
    ) -> AccumulationTensor<'l, { T }, D, Chip, Cluster, Slice, OutTime, OutPacket> {
        verify_accumulate::<Row, Time, Packet, OutTime, OutPacket>(kind);
        AccumulationTensor::new(self.ctx, self.inner.reduce_add())
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    AccumulationTensor<'l, { T }, D, Chip, Cluster, Slice, Time, Packet>
{
    // ANCHOR: accumulation_vector_init
    /// Initializes Vector Engine processing from contraction output.
    #[primitive(AccumulationTensor::vector_init)]
    pub fn vector_init(self) -> VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
    where
        D: VeScalar,
        // ANCHOR_END: accumulation_vector_init
    {
        VectorInitTensor::new(self.ctx, self.inner)
    }
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M> StreamCast<D>
    for AccumulationTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    type CastOutput<D2: Scalar, OutPacket: M>
        = CastTensor<'l, T, D2, Chip, Cluster, Slice, Time, OutPacket>
    where
        D: Cast<D2>;

    #[primitive(AccumulationTensor::cast)]
    fn cast<D2: Scalar, OutPacket: M>(self) -> Self::CastOutput<D2, OutPacket>
    where
        D: Cast<D2>,
    {
        cast_stream(self.ctx, self.inner)
    }
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M> StreamCast<D>
    for VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    type CastOutput<D2: Scalar, OutPacket: M>
        = CastTensor<'l, T, D2, Chip, Cluster, Slice, Time, OutPacket>
    where
        D: Cast<D2>;

    #[primitive(VectorFinalTensor::cast)]
    fn cast<D2: Scalar, OutPacket: M>(self) -> Self::CastOutput<D2, OutPacket>
    where
        D: Cast<D2>,
    {
        cast_stream(self.ctx, self.inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Stores to the vector register file after VE pipeline.
    #[primitive(VectorFinalTensor::to_vrf)]
    pub fn to_vrf<Element: M>(self, address: Address) -> VrfTensor<D, Chip, Cluster, Slice, Element>
    where
        D: VeScalar,
    {
        VrfTensor::new(self.inner.transpose(false), address)
    }

    /// Performs transpose operation (transitions to Transpose engine).
    #[primitive(VectorFinalTensor::transpose)]
    pub fn transpose<Time2: M, Packet2: M>(self) -> TransposeTensor<'l, T, D, Chip, Cluster, Slice, Time2, Packet2> {
        TransposeTensor::new(self.ctx, self.inner.transpose(false))
    }
}

/// Verifies commit engine constraints.
///
/// Constraints checked:
/// 1. Input packet must be exactly one flit (32 bytes).
/// 2. Output packet must be 8, 16, 24, or 32 bytes.
/// 3. Truncation may only remove elements from Packet.
fn verify_commit<D: Scalar, Time: M, Packet: M, Element: M>() {
    // Input packet must be exactly one flit.
    let packet_bytes = D::size_in_bytes_from_length(Packet::SIZE);
    assert_eq!(
        packet_bytes, FLIT_BYTES,
        "Commit input packet must be exactly {FLIT_BYTES} bytes (one flit), got {packet_bytes}",
    );

    // Time can be transposed.
    let Tuple2(time, packet) = Element::to_value()
        .factorize()
        .split_at(exact_div(Element::SIZE, Time::SIZE).expect("Commit element size does not divide time size"));
    let input_time = Time::to_value().factorize().normalize();
    if input_time.clone().divide_relaxed(time.clone()).exact().is_err()
        || time.clone().divide_relaxed(input_time.clone()).exact().is_err()
    {
        panic!("Commit output Time ({time}) is not a valid transpose of the input Time ({input_time})");
    }

    // Output packet must be 8, 16, 24, or 32 bytes.
    let out_packet_elements = Element::SIZE / Time::SIZE;
    let out_packet_bytes = D::size_in_bytes_from_length(out_packet_elements);
    assert!(
        COMMIT_OUT_PACKET_SIZES.contains(&out_packet_bytes),
        "Commit output packet must be one of {COMMIT_OUT_PACKET_SIZES:?} bytes, got {out_packet_bytes}",
    );

    // The resulting packet can be a slice of Packet by `commit_in_size`.
    let expected_packet = Packet::to_value().factorize();
    assert!(
        packet.is_resize_of(&expected_packet),
        "Commit packet mismatch. Expected {expected_packet} or a truncation of it, got {packet}",
    );
}

// ANCHOR: commit_impl
impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    StreamTensor<'l, { T }, P, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Commits to the data memory.
    #[primitive(StreamTensor::commit)]
    pub fn commit<Element: M>(self, address: Address) -> DmTensor<D, Chip, Cluster, Slice, Element> {
        verify_commit::<D, Time, Packet, Element>();
        DmTensor::new(self.inner.transpose(false), address)
    }

    /// Commits to mutable tensor view in the data memory.
    #[primitive(StreamTensor::commit_view)]
    pub fn commit_view<Element: M>(self, mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element>) {
        verify_commit::<D, Time, Packet, Element>();
        dst.inner.write_transpose(self.inner.view(), false);
    }
}
// ANCHOR_END: commit_impl

fn align_up(a: usize, b: usize) -> usize {
    assert_ne!(b, 0);
    a.div_ceil(b) * b
}

fn exact_div(a: usize, b: usize) -> Option<usize> {
    if a.is_multiple_of(b) { Some(a / b) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod transpose {
        use super::*;
        use crate::scalar::bf16;

        mod valid {
            use super::*;
            axes![A = 4, B = 2, C = 8, D = 4, E = 8, F = 8, G = 2, X = 64, Y = 512];

            #[test]
            fn basic() {
                verify_transpose::<i8, m![C, F], m![E # 32], m![C, E], m![F # 32]>();
            }

            #[test]
            fn small() {
                // `elements_per_packet = B # 8` is sliced to `B`
                verify_transpose::<i8, m![A], m![B # 32], m![B], m![A # 32]>();
            }

            #[test]
            fn small_no_slicing() {
                verify_transpose::<i8, m![A], m![B # 32], m![B # 8], m![A # 32]>();
            }

            #[test]
            fn large_col() {
                verify_transpose::<i8, m![B, C, D], m![E # 32], m![B, D, E], m![C # 32]>();
            }

            #[test]
            fn bf16() {
                verify_transpose::<bf16, m![C, D], m![E # 16], m![C, E], m![D # 16]>();
            }
        }

        mod input_packet {
            use super::*;
            axes![C = 8, D = 8, E = 4, F = 16];

            #[test]
            #[should_panic(expected = "Transpose input packet must be 32 bytes, got 16")]
            fn invalid() {
                verify_transpose::<i8, m![C, D], m![E # 16], m![C, E], m![D # 32]>();
            }
        }

        mod output_packet {
            use super::*;
            axes![C = 8, D = 8, E = 8];

            #[test]
            #[should_panic(expected = "Transpose output packet must be 32 bytes, got 16")]
            fn invalid() {
                verify_transpose::<i8, m![C, D], m![E # 32], m![C, E], m![D # 16]>();
            }
        }

        mod in_rows {
            use super::*;
            axes![A = 4, C = 8, D = 8, E = 8, F = 32];

            #[test]
            #[should_panic(expected = "Transpose `in_rows` must be <= 8 bytes, got 16")]
            fn invalid_i4() {
                verify_transpose::<i4, m![C, D], m![E # 64], m![C, E], m![F # 64]>();
            }

            #[test]
            #[should_panic(expected = "Transpose `in_rows` must be <= 8 bytes, got 32")]
            fn invalid_i8() {
                verify_transpose::<i8, m![C, D], m![E # 32], m![C, E], m![A, D]>();
            }

            #[test]
            #[should_panic(expected = "Transpose `in_rows` must be <= 8 bytes, got 16")]
            fn invalid_bf16() {
                verify_transpose::<bf16, m![C, D], m![E # 16], m![C, E], m![D # 16]>();
            }

            #[test]
            #[should_panic(expected = "must be present in the input Time")]
            fn invalid_in_rows_not_in_time() {
                verify_transpose::<i8, m![C, D], m![E # 32], m![C, E], m![A # 32]>();
            }
        }

        mod in_cols {
            use super::*;
            axes![C = 8, D = 8, E = 8, F = 16, G = 4];

            #[test]
            #[should_panic(expected = "Transpose `in_cols` size (64) must be one of [16, 32] for 4-bit type")]
            fn invalid_i4() {
                verify_transpose::<i4, m![F, G], m![E # 64], m![G, E], m![F # 64]>();
            }

            #[test]
            #[should_panic(expected = "Transpose `in_cols` size (64) must be one of [8, 16, 32] for 8-bit type")]
            fn invalid_i8() {
                verify_transpose::<i8, m![C, D], m![E # 32], m![D, E], m![C # 32]>();
            }
        }

        mod out_time {
            use super::*;
            axes![A = 2, B = 2, C = 4, D = 2, E = 8, F = 8, G = 16];

            #[test]
            #[should_panic(expected = "Transpose time mismatch")]
            fn invalid_outer_mismatch() {
                verify_transpose::<i8, m![B, C, D], m![E # 32], m![A, D, E], m![C # 32]>();
            }

            #[test]
            #[should_panic(expected = "not found in OutTime")]
            fn invalid_missing_packets_per_col() {
                verify_transpose::<i8, m![C, D], m![E # 32], m![E], m![C # 32]>();
            }

            #[test]
            #[should_panic(expected = "must match")]
            fn invalid_wrong_axis() {
                verify_transpose::<i8, m![C, D], m![E # 32], m![D, F], m![C # 32]>();
            }

            #[test]
            #[should_panic(expected = "must match")]
            fn invalid_non_padding_resize() {
                // E = 4 discards non-padded elements
                verify_transpose::<i8, m![C], m![E # 32], m![E = 4], m![C # 32]>();
            }

            #[test]
            #[should_panic(expected = "must be <=")]
            fn invalid_out_rows_exceeds_in_cols() {
                verify_transpose::<i8, m![A], m![B # 32], m![B # 16], m![A # 32]>();
            }
        }
    }

    mod commit {
        use super::*;

        mod valid {
            use super::*;

            axes![M = 4, N = 8, A = 4, B = 3, C = 4];

            #[test]
            fn full_truncation() {
                verify_commit::<i8, m![A, B, C], m![N # 32], m![A, B, C, N]>();
            }

            #[test]
            fn partial_truncation() {
                verify_commit::<i8, m![M], m![N # 32], m![M, N # 16]>();
            }

            #[test]
            fn no_truncation() {
                verify_commit::<i8, m![M], m![N # 32], m![M, N # 32]>();
            }

            #[test]
            fn bf16() {
                verify_commit::<bf16, m![M], m![N # 16], m![M, N]>();
            }

            #[test]
            fn f32() {
                verify_commit::<f32, m![M], m![N # 8], m![M, N]>();
            }

            #[test]
            fn single_time_step() {
                verify_commit::<i8, m![1], m![N # 32], m![N # 8]>();
            }

            #[test]
            fn non_padding_resize() {
                verify_commit::<bf16, m![1], m![N # 16], m![N = 4]>();
            }

            #[test]
            fn time_transpose() {
                verify_commit::<bf16, m![A # 32, B], m![N # 16], m![B, A # 32, N = 4]>();
            }
        }

        mod invalid {
            use super::*;

            axes![M = 4, N = 8, X = 8, Y = 4, Z = 2];

            #[test]
            #[should_panic(expected = "Commit input packet must be exactly 32 bytes (one flit), got 16")]
            fn input_packet_not_flit() {
                verify_commit::<i8, m![M], m![N # 16], m![M, N]>();
            }

            #[test]
            #[should_panic(expected = "Commit output packet must be one of [8, 16, 24, 32] bytes, got 13")]
            fn out_packet_invalid_size() {
                verify_commit::<i8, m![M], m![N # 32], m![M, N # 13]>();
            }

            #[test]
            #[should_panic(expected = "Commit output packet must be one of [8, 16, 24, 32] bytes, got 48")]
            fn extra_padding() {
                verify_commit::<i8, m![M], m![N # 32], m![M, N # 48]>();
            }

            #[test]
            #[should_panic(expected = "Commit packet mismatch")]
            fn different_packet_axes() {
                verify_commit::<i8, m![M], m![N # 32], m![M, X]>();
            }

            #[test]
            #[should_panic(expected = "not a valid transpose of the input Time")]
            fn different_time_axes() {
                verify_commit::<i8, m![M], m![N # 32], m![Y, N # 16]>();
            }

            #[test]
            #[should_panic(expected = "not a valid transpose of the input Time")]
            fn time_transpose_padding_mismatch() {
                verify_commit::<bf16, m![M # 32, X], m![N # 16], m![X, M # 16, N = 4]>();
            }

            #[test]
            #[should_panic(expected = "not a valid transpose of the input Time")]
            fn time_axis_dropped_with_padding() {
                verify_commit::<i8, m![M, Z], m![N # 32], m![M # 8, N]>();
            }

            #[test]
            #[should_panic(expected = "not a valid transpose of the input Time")]
            fn time_axis_resized_with_padding() {
                verify_commit::<i8, m![M # 8], m![N # 32], m![M = 2 # 8, N]>();
            }
        }
    }

    mod contract {
        use super::*;

        axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

        #[test]
        fn valid_full_reduction() {
            verify_contract::<i8, m![K], m![1]>();
        }

        #[test]
        fn valid_partial_reduction() {
            // K % 4 reduced
            verify_contract::<i8, m![K], m![K / 4]>();
        }

        #[test]
        #[should_panic(expected = "not a valid contraction")]
        fn invalid_retained_packet_size() {
            verify_contract::<i8, m![K], m![D]>();
        }

        #[test]
        #[should_panic(expected = "OutPacket::SIZE must be at most 32, got 64")]
        fn invalid_no_reduction() {
            // Temporal accumulator only has 32 columns, cannot fit 64 packet
            verify_contract::<i8, m![K], m![K]>();
        }

        #[test]
        fn valid_partial_reduction_multi_axis() {
            // `D / 2 % 4` is reduced, retained_packet is `[A, D / 8]`.
            verify_contract::<i8, m![A, D / 2], m![A, D / 8]>();
        }

        #[test]
        fn valid_padded_packet_inner_reduction() {
            verify_contract::<i8, m![A # 16, C], m![A]>();
        }

        #[test]
        fn valid_padded_packet_inner_reduction_with_padding() {
            verify_contract::<i8, m![A # 16, C], m![A # 16]>();
        }

        #[test]
        fn valid_padded_packet_split() {
            verify_contract::<i8, m![B # 8, N], m![B]>();
        }

        #[test]
        fn valid_no_spatial_reduction_bf16() {
            // Tree depth 0: all 32 bf16 elements pass through, no reduction.
            verify_contract::<bf16, m![D], m![D]>();
        }

        #[test]
        #[should_panic(expected = "OutPacket::SIZE must be a power of two, got 3")]
        fn invalid_non_power_of_two_out_packet() {
            verify_contract::<i8, m![K], m![K = 3]>();
        }

        #[test]
        #[should_panic(expected = "not a valid contraction")]
        fn invalid_partial_inner_packet() {
            verify_contract::<i8, m![K], m![K % 4]>();
        }
    }

    mod accumulate {
        use super::*;

        axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

        mod out_packet_size {
            use super::*;

            #[test]
            fn valid() {
                verify_accumulate::<m![1], m![A], m![1], m![A], m![1 # 8]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "OutPacket::SIZE must be 8, got 32")]
            fn invalid() {
                verify_accumulate::<m![1], m![A], m![1], m![A], m![D]>(AccumulationKind::Interleaved);
            }
        }

        mod interleaved {
            use super::*;

            #[test]
            fn valid() {
                verify_accumulate::<m![1], m![A, B], m![1], m![B], m![1 # 8]>(AccumulationKind::Interleaved);
            }

            #[test]
            fn valid_padding() {
                verify_accumulate::<m![1], m![A # 8, B # 4], m![1], m![B # 4], m![1 # 8]>(
                    AccumulationKind::Interleaved,
                );
            }

            #[test]
            fn valid_no_reduction_with_padding() {
                verify_accumulate::<m![1], m![A # 8, B], m![D], m![A # 8, B, D], m![1 # 8]>(
                    AccumulationKind::Interleaved,
                );
            }

            #[test]
            fn valid_non_outermost() {
                verify_accumulate::<m![N], m![C, A, B], m![1], m![C, B], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            fn valid_four_rows() {
                verify_accumulate::<m![M], m![C, A, B], m![1], m![C, B], m![M # 8]>(AccumulationKind::Interleaved);
            }

            #[test]
            fn valid_all_time_reduced() {
                verify_accumulate::<m![N], m![A], m![1], m![1], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "OutTime mismatch")]
            fn invalid_sliced_no_padding() {
                // Packet truncation is only allowed on padded elements.
                // Otherwise, input data would be silently discarded.
                verify_accumulate::<m![N], m![M], m![D], m![M, D = 16], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "Could not decompose OutTime")]
            fn invalid_packet_size_out_time() {
                verify_accumulate::<m![N], m![M], m![D], m![M, K], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "OutTime mismatch")]
            fn invalid_resize() {
                verify_accumulate::<m![1], m![A], m![D], m![A, D / 4 % 4], m![1 # 8]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "OutTime mismatch")]
            fn invalid_out_time() {
                verify_accumulate::<m![N], m![A, B], m![1], m![C], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "Padding mismatch")]
            fn invalid_out_time_padding() {
                verify_accumulate::<m![1], m![A, B # 4], m![1], m![B # 2], m![1 # 8]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "OutTime axes must follow the same order as the Time axes")]
            fn invalid_out_time_reorder() {
                verify_accumulate::<m![N], m![A, B], m![1], m![B, A], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "Padding mismatch")]
            fn invalid_out_time_no_padding() {
                verify_accumulate::<m![N], m![A, B # 32], m![1], m![A, B], m![N]>(AccumulationKind::Interleaved);
            }

            #[test]
            #[should_panic(expected = "axes inner to reduce must be <= 128 in size, got 256")]
            fn invalid_buffer() {
                verify_accumulate::<m![N], m![A, D # 64, C], m![1], m![D # 64, C], m![N]>(
                    AccumulationKind::Interleaved,
                );
            }

            #[test]
            #[should_panic(expected = "axes inner to reduce must be <= 128 in size, got 256")]
            fn invalid_buffer_multiple_reduce_axes() {
                verify_accumulate::<m![N], m![A, B # 64, M # 8, C], m![1], m![B # 64, C], m![N]>(
                    AccumulationKind::Interleaved,
                );
            }
        }

        mod sequential {
            use super::*;

            #[test]
            fn valid() {
                verify_accumulate::<m![N], m![A, B], m![1], m![B, N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            fn valid_padded_row() {
                verify_accumulate::<m![N], m![A, B], m![1], m![B, N # 8], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            fn valid_all_time_reduced() {
                verify_accumulate::<m![N], m![A], m![1], m![N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            fn valid_no_reduction_with_padding() {
                verify_accumulate::<m![N], m![A # 8, B], m![1], m![A # 8, B, N], m![1 # 8]>(
                    AccumulationKind::Sequential,
                );
            }

            #[test]
            fn valid_padded_packet() {
                verify_accumulate::<m![N], m![M], m![B], m![M, N], m![B # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            fn valid_full_temporal_reduction() {
                verify_accumulate::<m![N], m![M], m![D], m![N, D / 8], m![D % 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "OutPacket mismatch")]
            fn invalid_packet_axis() {
                verify_accumulate::<m![N], m![M], m![B], m![M, N], m![A # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "OutTime mismatch")]
            fn invalid_out_time() {
                verify_accumulate::<m![N], m![A, B], m![1], m![C, N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "OutTime mismatch")]
            fn invalid_out_time_row() {
                verify_accumulate::<m![N], m![A, B], m![1], m![B, M], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "OutTime axes must follow the same order as the Time axes")]
            fn invalid_out_time_reorder() {
                verify_accumulate::<m![N], m![A, B], m![1], m![B, A, N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "Padding mismatch")]
            fn invalid_out_time_padding() {
                verify_accumulate::<m![N], m![A, B # 32], m![1], m![B, N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "Padding mismatch")]
            fn invalid_out_time_padded_row() {
                verify_accumulate::<m![M], m![A, B], m![1], m![B, M # 8], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            fn valid_multi_axis_reduction() {
                verify_accumulate::<m![N], m![A, B, C], m![1], m![B, N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "axes inner to reduce must be <= 32 in size, got 64")]
            fn invalid_buffer() {
                verify_accumulate::<m![N], m![A, P], m![1], m![P, N], m![1 # 8]>(AccumulationKind::Sequential);
            }

            #[test]
            #[should_panic(expected = "axes inner to reduce must be <= 32 in size, got 64")]
            fn invalid_buffer_packet_outer() {
                verify_accumulate::<m![1], m![A, N, B], m![D], m![N, B, D / 8], m![D % 8]>(
                    AccumulationKind::Sequential,
                );
            }
        }
    }

    mod switch {
        use super::super::*;

        mod custom_broadcast {
            use super::*;

            axes![
                A = 16,
                B = 16,
                C = 8,
                D = 2,
                E = 2,
                P = 4,
                Q = 8,
                R = 8,
                S = 256,
                X = 4,
                Y = 2,
                Z = 2,
            ];

            mod permutation {
                use super::*;

                #[test]
                fn identity() {
                    verify_switch::<m![S], m![C], m![S], m![C]>(&SwitchConfig::CustomBroadcast { ring_size: 1 });
                }

                #[test]
                fn full_permutation() {
                    verify_switch::<m![A, B], m![C], m![B % 4, B / 4, A % 4, A / 4], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 256 },
                    );
                }

                #[test]
                fn partial_permutation() {
                    verify_switch::<m![A, B], m![C], m![A, B % 4, B / 4], m![C]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 16,
                    });
                }

                #[test]
                fn three_axis_inner_swap() {
                    verify_switch::<m![R, Q, P], m![C], m![R, P, Q], m![C]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 32,
                    });
                }

                #[test]
                fn three_axis_outer_swap() {
                    verify_switch::<m![R, Q, P], m![C], m![Q, R, P], m![C]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 256,
                    });
                }

                #[test]
                fn padded_identity() {
                    verify_switch::<m![P # 16, Q # 16], m![C], m![P # 16, Q # 16], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 1 },
                    );
                }

                #[test]
                fn padded_full_swap() {
                    verify_switch::<m![R # 16, Q # 16], m![C], m![Q # 16, R # 16], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 256 },
                    );
                }

                #[test]
                fn padded_full_swap_different_padding() {
                    verify_switch::<m![R # 16, Q # 16], m![C], m![Q # 32, R # 8], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 256 },
                    );
                }

                #[test]
                fn padded_partial_permutation() {
                    verify_switch::<m![R # 16, Q # 16], m![C], m![R # 16, Q # 16 % 4, Q # 16 / 4], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 16 },
                    );
                }

                #[test]
                #[should_panic(
                    expected = "Switch axes moving from input slice to output time must be at the output time innermost positions."
                )]
                fn permutation_time_change() {
                    verify_switch::<m![A, B], m![C], m![B, A], m![R]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 256,
                    });
                }
            }

            mod broadcast {
                use super::*;

                #[test]
                fn broadcast() {
                    verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 4,
                    });
                }

                #[test]
                fn multi_axis_broadcast() {
                    verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, A % 2, B % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                fn broadcast_with_permutation() {
                    verify_switch::<m![A, B], m![C], m![A % 4, A / 4, B / 4, X], m![C, B % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 256 },
                    );
                }

                #[test]
                fn broadcast_with_inner_permutation() {
                    verify_switch::<m![R, Q, P], m![C], m![R, P / 2, Q, Y], m![C, P % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                fn broadcast_innermost_axis() {
                    verify_switch::<m![R, Q, P], m![C], m![R, Q, P / 2, Y], m![C, P % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 2 },
                    );
                }

                #[test]
                fn non_contiguous_broadcast() {
                    // Move R % 2 and P % 2 to Time (skipping Q).
                    verify_switch::<m![R, Q, P], m![C], m![R / 2, Y, Q, P / 2, Z], m![C, R % 2, P % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 64 },
                    );
                }

                #[test]
                fn full_broadcast() {
                    verify_switch::<m![A, B], m![C], m![S], m![C, A, B]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 256,
                    });
                }

                #[test]
                fn padded_outer_time() {
                    verify_switch::<m![A, B], m![C # 32], m![A, B / 4, X], m![C # 32, B % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                fn padded_inner_axis_broadcast() {
                    verify_switch::<m![P # 8, Q # 32], m![C], m![P # 8, Q # 32 / 4, X], m![C, Q # 32 % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                fn broadcast_with_padded_outer_axis() {
                    verify_switch::<m![P # 32, Q], m![C], m![P # 32, Q / 4, X], m![C, Q % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                fn padded_both_axes_broadcast() {
                    verify_switch::<m![P # 16, Q # 16], m![C], m![P # 16, Q # 16 / 4, X], m![C, Q # 16 % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                fn padded_time_broadcast() {
                    verify_switch::<m![P # 8, Q # 32], m![C # 16], m![P # 8, Q # 32 / 4, X], m![C # 16, Q # 32 % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch broadcast axes must each be used exactly once in OutSlice")]
                fn duplicate_broadcast_symbol() {
                    verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Y], m![C, A % 2, B % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                #[should_panic(
                    expected = "Switch broadcast axes must be new axes (not present in input Slice or Time)."
                )]
                fn broadcast_axis_from_time() {
                    verify_switch::<m![A, B], m![C], m![A, B / 8, C], m![C, B % 8]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 8,
                    });
                }

                #[test]
                #[should_panic(
                    expected = "Switch broadcast axes must be new axes (not present in input Slice or Time)."
                )]
                fn inter_transpose() {
                    verify_switch::<m![A, B], m![C], m![A, C, B / 8], m![B % 8]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 256,
                    });
                }

                #[test]
                fn partial_broadcast_replacement() {
                    // A % 2 replaced by broadcast
                    verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, B % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch broadcast axis X in output Slice must not be padded.")]
                fn padded_broadcast_axis() {
                    verify_switch::<m![A, B], m![C], m![A / 2, X # 8, B / 8, Y], m![C, A % 2, B % 8]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                #[should_panic(
                    expected = "Switch axes moving from input slice to output time must be at the output time innermost positions."
                )]
                fn moved_axes_not_innermost() {
                    verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![A % 2, C, B % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                #[should_panic(
                    expected = "Switch axes moving from input Slice to output Time must preserve their relative order from input Slice."
                )]
                fn reversed_order_in_time() {
                    // A % 2 and B % 2 are reversed in output Time.
                    verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 2, Z], m![C, B % 2, A % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                #[should_panic(
                    expected = "Switch axes moving from input slice to output time must be at the output time innermost positions."
                )]
                fn outer_time_padding_mismatch() {
                    verify_switch::<m![A, B], m![C # 32], m![A, B / 4, X], m![C # 16, B % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                fn broadcast_replace_in_place() {
                    verify_switch::<m![R, P], m![C], m![R, X], m![C]>(&SwitchConfig::CustomBroadcast { ring_size: 4 });
                }

                #[test]
                fn broadcast_with_moved_axis() {
                    axes![A = 2, B = 2, C = 2, D = 2, E = 2, X = 2];
                    verify_switch::<m![A, B, C, D, E], m![1], m![E, B, X, A, D], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    )
                }
            }

            mod slicing {
                use super::*;

                #[test]
                fn slicing() {
                    verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4 = 3]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                fn slicing_with_broadcast() {
                    verify_switch::<m![A, B], m![C], m![A / 2, Y, B / 4, X], m![C, A % 2, B % 4 = 3]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 32 },
                    );
                }

                #[test]
                fn single_axis_slicing() {
                    verify_switch::<m![S], m![C], m![S / 4, X], m![C, S % 4 = 3]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 4,
                    });
                }

                #[test]
                fn padded_broadcast_slicing() {
                    verify_switch::<m![P # 8, Q # 32], m![C], m![P # 8, Q # 32 / 4, X], m![C, Q # 32 % 4 = 3]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch broadcast axes in output time must come from input slice.")]
                fn wrong_axis_in_slicing() {
                    verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B / 4 = 3]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }
            }

            mod ring_size {
                use super::*;

                #[test]
                #[should_panic(expected = "Switch ring size must be a power of 2, got 3")]
                fn non_power_of_two() {
                    verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 3,
                    });
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 256, got 4")]
                fn wrong_full_permutation() {
                    verify_switch::<m![A, B], m![C], m![B % 4, B / 4, A % 4, A / 4], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 16, got 256")]
                fn wrong_partial_permutation() {
                    verify_switch::<m![A, B], m![C], m![A, B % 4, B / 4], m![C]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 256,
                    });
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 4, got 32")]
                fn wrong_broadcast() {
                    verify_switch::<m![A, B], m![C], m![A, B / 4, X], m![C, B % 4]>(&SwitchConfig::CustomBroadcast {
                        ring_size: 32,
                    });
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 256, got 2")]
                fn wrong_permutation_above_broadcast() {
                    verify_switch::<m![R, Q, P], m![C], m![Q, R, P / 2, Y], m![C, P % 2]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 2 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 4, got 16")]
                fn wrong_padded_broadcast() {
                    verify_switch::<m![A # 16, B # 16], m![C], m![A # 16, B # 16 / 4, X], m![C, B # 16 % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 16 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 16, got 256")]
                fn wrong_padded_permutation() {
                    verify_switch::<m![A # 16, B # 16], m![C], m![A # 16, B # 16 % 4, B # 16 / 4], m![C]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 256 },
                    );
                }

                #[test]
                #[should_panic(expected = "Switch ring size mismatch. Expected 256, got 4")]
                fn wrong_padded_permutation_above_broadcast() {
                    verify_switch::<m![P # 8, Q # 32], m![C], m![Q # 32 / 4, P # 8, X], m![C, Q # 32 % 4]>(
                        &SwitchConfig::CustomBroadcast { ring_size: 4 },
                    );
                }
            }
        }
    }
}
