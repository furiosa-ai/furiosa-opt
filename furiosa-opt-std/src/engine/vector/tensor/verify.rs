use furiosa_mapping::*;

pub(crate) fn verify_vector_narrow_split<Time: M, Packet: M, Time2: M, Packet2: M>() {
    assert_eq!(Packet::SIZE, 8, "Split requires Packet of 8 elements (one flit).");
    let (packet_outer, packet_inner) = Packet::to_value().split_at(4);
    let expected_time = Time::to_value().pair(packet_outer).normalize();
    let expected_packet = packet_inner.normalize();

    let out_time = Time2::to_value().normalize();
    let out_packet = Packet2::to_value().normalize();

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

pub(crate) fn verify_vector_widen_concat<Time: M, Packet: M, Time2: M, Packet2: M>() {
    assert_eq!(Packet::SIZE, 4, "Concat requires Packet of 4 elements (Way4 mode).");
    let (time_outer, time_inner) = Time::to_value().split_at(2);
    let expected_time = time_outer.normalize();
    let expected_packet = time_inner.pair(Packet::to_value()).normalize();

    let out_time = Time2::to_value().normalize();
    let out_packet = Packet2::to_value().normalize();

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

/// Verify vector_narrow_trim: Packet (size 8) → Packet2 (size 4, front half only).
///
/// Checks that the back 4 of Packet are dummy,
/// and that Packet2 matches the front 4 of Packet.
pub(crate) fn verify_vector_narrow_trim<Packet: M, Packet2: M>() {
    assert_eq!(
        Packet::SIZE,
        8,
        "vector_narrow_trim: input Packet must have 8 elements (one flit), got {}. \
         vector_narrow_trim is used to strip the back-4 dummy lanes before float operations. \
         If Packet is already 4, you don't need vector_narrow_trim.",
        Packet::SIZE,
    );
    let (packet_outer, packet_inner) = Packet::to_value().split_at(4);
    // Back 4 must be dummy — i.e., the outer half must be [1 # 2] (uninit padding).
    // If this fails, the back 4 lanes contain real data and you need vector_narrow_split() instead.
    assert_eq!(
        packet_outer.clone().normalize(),
        <m![1 # 2]>::to_value().normalize(),
        "vector_narrow_trim: the back 4 lanes of the packet must be dummy (padding), \
         but got: {packet_outer}. \
         If the back 4 lanes contain real data, use vector_narrow_split() instead of vector_narrow_trim()."
    );
    // Output must be the front 4
    assert_eq!(
        Packet2::SIZE,
        4,
        "vector_narrow_trim: output Packet2 must have 4 elements, got {}.",
        Packet2::SIZE,
    );
    assert_eq!(
        packet_inner.clone().normalize(),
        Packet2::to_value().normalize(),
        "vector_narrow_trim: Packet2 must match the front 4 of Packet. \
         Expected: {packet_inner}, got: {}.",
        Packet2::to_value(),
    );
}

/// Verify vector_widen_pad: Packet (size 4) → Packet2 (size 8, padded with dummy).
///
/// Checks that Packet2 is Packet padded to 8.
pub(crate) fn verify_vector_widen_pad<Packet: M, Packet2: M>() {
    assert_eq!(
        Packet::SIZE,
        4,
        "vector_widen_pad: input Packet must have 4 elements (after vector_narrow_trim), got {}. \
         vector_widen_pad restores the back-4 dummy lanes stripped by vector_narrow_trim.",
        Packet::SIZE,
    );
    assert_eq!(
        Packet2::SIZE,
        8,
        "vector_widen_pad: output Packet2 must have 8 elements (one flit), got {}.",
        Packet2::SIZE,
    );
    let expected = Packet::to_value().replace_padding(8).normalize();
    assert_eq!(
        expected,
        Packet2::to_value().normalize(),
        "vector_widen_pad: Packet2 must be Packet padded to 8. \
         Expected: {expected}, got: {}.",
        Packet2::to_value(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![R = 19, A = 2];

    /// Padded stride/modulo split through `vector_narrow_split`.
    ///
    /// Input  Time  = `R # 24 / 4`, Packet = `(R # 24 % 4, A)` (size 8).
    /// Output Time2 = `(R # 24 / 4, R # 24 / 2 % 2)`, Packet2 = `(R # 24 % 2, A)`.
    ///
    /// The complementary halves only line up because `R # 24 % n` factorizes to
    /// its minimal-aligned period, matching the period the `/ stride` partner
    /// produces.
    #[test]
    fn vector_narrow_split_padded_stride_modulo() {
        verify_vector_narrow_split::<
            m![R # 24 / 4],
            m![R # 24 % 4, A],
            m![R # 24 / 4, R # 24 / 2 % 2],
            m![R # 24 % 2, A],
        >();
    }
}
