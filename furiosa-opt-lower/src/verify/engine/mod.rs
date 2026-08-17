//! Per-engine verifications, mirroring `furiosa-opt-std/src/engine`.

mod cast;
mod collect;
mod commit_adapter;
mod contraction;
mod vector;

pub use cast::{CastError, config_cast};
pub use collect::{CollectError, ToTrfError, ToVrfError, config_collect, config_to_trf, config_to_vrf};
pub use commit_adapter::{CommitCastError, CommitTrimError, config_commit_cast, config_commit_trim};
pub use contraction::{
    ContractLaneError, ContractPacketError, ContractTimeError, LaneMode, StreamAdapterError, config_contract_lane,
    config_contract_packet, config_contract_time, config_stream_adapter,
};
pub use vector::{
    VectorError, config_reduce_label, config_vector_narrow_split, config_vector_narrow_trim,
    config_vector_widen_concat, config_vector_widen_pad,
};
