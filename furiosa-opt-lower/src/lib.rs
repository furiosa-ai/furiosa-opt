//! Furiosa-opt lowering engines.
//!
//! Two kinds of function live here. The lowering *algorithms* (`config_transpose`, `config_fetch`, …)
//! are thin safe wrappers over the prebuilt, hidden `furiosa-opt-lower-impl` static library: the
//! resolved results cross the `extern "C-unwind"` boundary as [`furiosa_opt_lower_types`] StableAbi
//! values, and each wrapper converts back to a plain `Result`. The engine *verifications* (the
//! `config_*` in [`verify`]) are published in full: they state the DSL contract in pure
//! `furiosa_mapping` terms, with no hidden algorithm. Both the frontend (`furiosa-opt-std`) and the IR
//! backend (`tu-ops` / `npu-visa-translate`) call all of these.

pub use furiosa_opt_lower_types::{
    COMMIT_VALID_PACKET_SIZES, CommitError, DivideError, DivideTerm, FETCH_BASE_BYTES, FETCH_VALID_CLUSTER_SIZES,
    FETCH_VALID_SLICE_SIZES, FactorLeaf, FetchBaseError, FetchError, MAX_SEQUENCER_ENTRIES, PadError, RelaxedDivision,
    StreamSequencerConfig, SwitchAxis, SwitchConfig, SwitchError, SwitchFrame, TileError, TransposeConfig,
    TransposeError,
};

mod lowering;
mod sys;
mod verify;
pub use lowering::*;
pub use verify::{
    BITS_PER_BYTE, CastError, CastInput, CastKind, ClusterPlacement, CollectError, CollectInput, CommitCastError,
    CommitCastInput, CommitCastKind, CommitTrimError, CommitTrimInput, ContractLaneError, ContractLaneInput,
    ContractPacketError, ContractPacketInput, ContractTimeError, ContractTimeInput, DM_WRITE_ALIGN_BYTES,
    ElementSizeError, FLIT_BYTES, FetchContext, FetchDimensionsInput, FetchLiftDimension, FetchLiftError,
    FetchLiftInput, FetchVolumeError, LaneMode, ReduceLabelInput, SliceError, SlicePatternError, SlicePlan,
    SliceRequest, SramRedistributeEntry, StreamAdapterError, StreamAdapterInput, TEMPORAL_ACCUMULATOR_COLS, ToTrfError,
    ToTrfInput, ToVrfError, ToVrfInput, UnreadableCause, VRF_BYTES, VRF_CACHE_BYTES, VectorError,
    VectorIntraSliceUnzipInput, VectorNarrowSplitInput, VectorNarrowTrimInput, VectorWidenConcatInput,
    VectorWidenPadInput, VrfOperandError, VrfOperandInput, config_cast, config_collect, config_commit_cast,
    config_commit_trim, config_contract_lane, config_contract_packet, config_contract_time, config_fetch_dimensions,
    config_fetch_lift, config_fetch_volume, config_outermost_dm_slice, config_reduce_label, config_slice,
    config_stream_adapter, config_to_trf, config_to_vrf, config_vector_intra_slice_unzip, config_vector_narrow_split,
    config_vector_narrow_trim, config_vector_widen_concat, config_vector_widen_pad, config_vrf_operand,
    validate_chip_shuffle, validate_slice_indices, validate_sram_redistribution,
};
