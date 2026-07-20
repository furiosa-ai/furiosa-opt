//! Type-level wrappers over the shared `furiosa_opt_lower::config_vector_*` verifications.

use furiosa_mapping::*;

pub(crate) fn verify_vector_narrow_split<Time: M, Packet: M, Time2: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_narrow_split(
        &Time::to_value(),
        &Packet::to_value(),
        &Time2::to_value(),
        &Packet2::to_value(),
    )
    .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_widen_concat<Time: M, Packet: M, Time2: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_widen_concat(
        &Time::to_value(),
        &Packet::to_value(),
        &Time2::to_value(),
        &Packet2::to_value(),
    )
    .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_narrow_trim<Packet: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_narrow_trim(&Packet::to_value(), &Packet2::to_value())
        .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_widen_pad<Packet: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_widen_pad(&Packet::to_value(), &Packet2::to_value())
        .unwrap_or_else(|message| panic!("{message}"));
}
