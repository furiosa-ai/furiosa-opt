//! Contraction-engine verifications, mirroring `furiosa-opt-std/src/engine/contraction`.

mod lane;
mod outer;
mod packet;
mod time;

pub use lane::{ContractLaneError, ContractLaneInput, LaneMode, config_contract_lane};
pub use outer::{StreamAdapterError, StreamAdapterInput, config_stream_adapter};
pub use packet::{ContractPacketError, ContractPacketInput, config_contract_packet};
pub use time::{ContractTimeError, ContractTimeInput, config_contract_time};
