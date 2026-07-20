//! Stream Adapter: stream-side validation for the [`super`] `contract_outer` deferred fold.
//!
//! [`verify_stream_adapter`] checks that the input `Time`/`Packet` can be expanded into the joint
//! computation mapping (inner-flit packet match, Time-divides-OutTime, broadcast axes confined to
//! innermost positions): the validity contract the deferred stash must still honor even though it no
//! longer materializes the expansion.

use furiosa_mapping::*;

use crate::scalar::Scalar;

/// Validates the stream-side mapping transformation for the [`super`] `contract_outer`.
///
/// `OutPacket` collects `PackSize` flits (`∈ {1, 2}`), and packing absorbs the
/// innermost `PackSize` cells of `Time` into the packet. The joint mapping is then:
/// - `OutPacket = [packed, inner flit]`, where the inner flit equals the input
///   `Packet` and `packed` equals the innermost `PackSize` cells of `Time`,
///   padding included (a padded packed cell must have come from padding in `Time`).
/// - `OutTime = [outer, broadcast]`, where `outer` is the rest of `Time` and the
///   broadcast (tiling) axes occupy the innermost positions.
pub(super) fn verify_stream_adapter<D: Scalar, Lane: M, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    furiosa_opt_lower::config_stream_adapter(
        &Lane::to_value(),
        &Time::to_value(),
        &Packet::to_value(),
        &OutTime::to_value(),
        &OutPacket::to_value(),
        D::BITS,
    )
    .unwrap_or_else(|message| panic!("{message}"));
}
