use abi_stable::std_types::Tuple2;
use furiosa_mapping::{FMappingExt, Identity, M, MappingExt, Padding};
use furiosa_opt_macro::m;

pub(crate) fn verify_vector_split<Time: M, Packet: M, Time2: M, Packet2: M>() {
    assert_eq!(Packet::SIZE, 8, "Split requires Packet of 8 elements (one flit).");
    let Tuple2(packet_outer, packet_inner) = Packet::to_value().factorize().split_at(4);
    let expected_time = Time::to_value().factorize().mul(packet_outer);
    let expected_packet = packet_inner;

    let out_time = Time2::to_value().factorize();
    let out_packet = Packet2::to_value().factorize();

    assert_eq!(
        expected_time, out_time,
        "Vector_Split time factor mismatch. Expected: {expected_time}, got: {out_time}",
    );

    assert_eq!(
        Packet2::SIZE,
        4,
        "Vector_Split output Packet2 must have 4 elements (front half of flit), got: {}",
        Packet2::SIZE,
    );

    assert_eq!(
        expected_packet, out_packet,
        "Vector_Split packet mismatch. Expected: {expected_packet}, got: {out_packet}",
    );
}

pub(crate) fn verify_vector_concat<Time: M, Packet: M, Time2: M, Packet2: M>() {
    assert_eq!(Packet::SIZE, 4, "Concat requires Packet of 4 elements (Way4 mode).");
    let Tuple2(time_outer, time_inner) = Time::to_value().factorize().split_at(2);
    let expected_time = time_outer;
    let expected_packet = time_inner.mul(Packet::to_value().factorize()).normalize();

    let out_time = Time2::to_value().factorize();
    let out_packet = Packet2::to_value().factorize();

    assert_eq!(
        Packet2::SIZE,
        8,
        "Vector_Concat output Packet2 must have 8 elements (one flit), got: {}",
        Packet2::SIZE,
    );

    assert_eq!(
        expected_time, out_time,
        "Vector_Concat time factor mismatch. Expected: {expected_time}, got: {out_time}",
    );

    assert_eq!(
        expected_packet, out_packet,
        "Vector_Concat packet mismatch. Expected: {expected_packet}, got: {out_packet}",
    );
}

/// Verify vector_trim_way4: Packet (size 8) → Packet2 (size 4, front half only).
///
/// Checks that the back 4 of Packet are dummy,
/// and that Packet2 matches the front 4 of Packet.
pub(crate) fn verify_vector_trim_way4<Packet: M, Packet2: M>() {
    assert_eq!(
        Packet::SIZE,
        8,
        "vector_trim_way4: input Packet must have 8 elements (one flit), got {}. \
         vector_trim_way4 is used to strip the back-4 dummy lanes before float operations. \
         If Packet is already 4, you don't need vector_trim_way4.",
        Packet::SIZE,
    );
    let Tuple2(packet_outer, packet_inner) = Packet::to_value().factorize().split_at(4);
    // Back 4 must be dummy — i.e., the outer half must be [1 # 2] (uninit padding).
    // If this fails, the back 4 lanes contain real data and you need vector_split() instead.
    assert_eq!(
        packet_outer.clone().normalize(),
        <m![1 # 2]>::to_value().factorize(),
        "vector_trim_way4: the back 4 lanes of the packet must be dummy (padding), \
         but got: {packet_outer}. \
         If the back 4 lanes contain real data, use vector_split() instead of vector_trim_way4()."
    );
    // Output must be the front 4
    assert_eq!(
        Packet2::SIZE,
        4,
        "vector_trim_way4: output Packet2 must have 4 elements, got {}.",
        Packet2::SIZE,
    );
    assert_eq!(
        packet_inner,
        Packet2::to_value().factorize(),
        "vector_trim_way4: Packet2 must match the front 4 of Packet. \
         Expected: {packet_inner}, got: {}.",
        Packet2::to_value().factorize(),
    );
}

/// Verify vector_pad_way8: Packet (size 4) → Packet2 (size 8, padded with dummy).
///
/// Checks that Packet2 is Packet padded to 8.
pub(crate) fn verify_vector_pad_way8<Packet: M, Packet2: M>() {
    assert_eq!(
        Packet::SIZE,
        4,
        "vector_pad_way8: input Packet must have 4 elements (after vector_trim_way4), got {}. \
         vector_pad_way8 restores the back-4 dummy lanes stripped by vector_trim_way4.",
        Packet::SIZE,
    );
    assert_eq!(
        Packet2::SIZE,
        8,
        "vector_pad_way8: output Packet2 must have 8 elements (one flit), got {}.",
        Packet2::SIZE,
    );
    let expected = Packet::to_value().factorize().pad(8);
    assert_eq!(
        expected,
        Packet2::to_value().factorize(),
        "vector_pad_way8: Packet2 must be Packet padded to 8. \
         Expected: {expected}, got: {}.",
        Packet2::to_value().factorize(),
    );
}
