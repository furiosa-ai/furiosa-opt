//! Stream Adapter: stream-side validation and transpose for [`super::contract_outer`].
//!
//! - [`verify_stream_adapter`]: checks that the input `Time`/`Packet` can be
//!   expanded into the joint computation mapping (inner-flit packet match,
//!   Time-divides-OutTime, broadcast axes confined to innermost positions).
//! - [`contract_outer`]: transposes the stream tensor into the joint shape.

use abi_stable::std_types::Tuple2;
use furiosa_mapping::*;

use crate::engine::FLIT_BYTES;
use crate::runtime::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

pub(super) fn contract_outer<
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    OutTime: M,
    OutPacket: M,
    B: Backend,
>(
    inner: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }], B>,
) -> Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { OutTime }, { OutPacket }], B> {
    verify_stream_adapter::<D, Lane, Time, Packet, OutTime, OutPacket>();
    inner.transpose(true)
}

/// Validates that the input `Time`/`Packet` can be expanded into the joint
/// computation mapping:
/// - The inner flit of `OutPacket` matches the input `Packet`.
/// - `Time × outer_flit(OutPacket)` divides `OutTime`, with broadcast axes
///   confined to the innermost positions of `OutTime` (same relative order as
///   in `Time`).
pub(super) fn verify_stream_adapter<D: Scalar, Lane: M, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    // Inner flit of OutPacket must match input Packet.
    assert!(
        [1, 2, 4, 8].contains(&Lane::SIZE),
        "Lane::SIZE must be 1, 2, 4, or 8, got {}",
        Lane::SIZE
    );

    let out_packet_size = D::size_in_bytes_from_length(OutPacket::SIZE);
    assert!(
        [32, 64].contains(&out_packet_size),
        "OutPacket must be 32 or 64 bytes (matching PackSize ∈ {{1, 2}}), got {out_packet_size} bytes"
    );

    let flit_elements = D::length_from_bytes(FLIT_BYTES);
    let Tuple2(out_packet_outer, out_packet_inner) = OutPacket::to_value().factorize().split_at(flit_elements);
    let out_packet_inner = out_packet_inner.normalize();
    let expected_packet = Packet::to_value().factorize();
    assert_eq!(
        out_packet_inner, expected_packet,
        "`contract_outer` packet mismatch: inner flit of OutPacket != Packet: {out_packet_inner} != {expected_packet}",
    );

    // Time must equal OutTime * outer flit portion of OutPacket.
    // Padding is stripped for the collect_flits = 1 case.
    let expected_time = OutTime::to_value()
        .factorize()
        .mul(out_packet_outer.remove_padding())
        .normalize();
    let input_time = Time::to_value().factorize();

    let tiling_size = expected_time.size() / input_time.size();
    let division_terms = expected_time
        .divide(input_time.clone())
        .exact_checked()
        .expect("`contract_outer`: Time does not divide OutTime")
        .division_terms;

    // Non-broadcast axes must follow the same order in both mappings.
    assert!(
        division_terms
            .windows(2)
            .all(|w| w[0].divisor_stride > w[1].divisor_stride),
        "`contract_outer`: Time axes are reordered in OutTime"
    );

    // Broadcast axes are the innermost in `OutTime`.
    assert!(
        division_terms.iter().all(|d| d.dividend_stride >= tiling_size),
        "`contract_outer`: tiling axes must be innermost in OutTime"
    );
}
