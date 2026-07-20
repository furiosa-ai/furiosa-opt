//! Contraction-engine verifications, mirroring `furiosa-opt-std/src/engine/contraction`.

mod lane;
mod outer;
mod packet;
mod time;

pub use lane::{ContractLaneError, LaneMode, config_contract_lane};
pub use outer::{StreamAdapterError, config_stream_adapter};
pub use packet::{ContractPacketError, config_contract_packet};
pub use time::{ContractTimeError, config_contract_time};
