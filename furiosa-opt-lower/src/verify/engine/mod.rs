//! Per-engine verifications, mirroring `furiosa-opt-std/src/engine`.

mod cast;
mod collect;
mod commit_adapter;
mod contraction;
mod fetch;
mod vector;

pub use cast::{CastError, CastInput, CastKind, config_cast};
pub use collect::{
    CollectError, CollectInput, ToTrfError, ToTrfInput, ToVrfError, ToVrfInput, config_collect, config_to_trf,
    config_to_vrf,
};
pub use commit_adapter::{
    CommitCastError, CommitCastInput, CommitCastKind, CommitTrimError, CommitTrimInput, config_commit_cast,
    config_commit_trim,
};
pub use contraction::{
    ContractLaneError, ContractLaneInput, ContractPacketError, ContractPacketInput, ContractTimeError,
    ContractTimeInput, LaneMode, StreamAdapterError, StreamAdapterInput, config_contract_lane, config_contract_packet,
    config_contract_time, config_stream_adapter,
};
pub use fetch::{
    FetchDimensionsInput, FetchLiftDimension, FetchLiftError, FetchLiftInput, config_fetch_dimensions,
    config_fetch_lift,
};
pub use vector::{
    ReduceLabelInput, UnreadableCause, VectorError, VectorIntraSliceUnzipInput, VectorNarrowSplitInput,
    VectorNarrowTrimInput, VectorWidenConcatInput, VectorWidenPadInput, VrfOperandError, VrfOperandInput,
    config_reduce_label, config_vector_intra_slice_unzip, config_vector_narrow_split, config_vector_narrow_trim,
    config_vector_widen_concat, config_vector_widen_pad, config_vrf_operand,
};
