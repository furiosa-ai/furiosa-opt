//! Transpose Engine: packet-level transpose.

use abi_stable::std_types::Tuple2;
use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::CanApplyTranspose;
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::tu::{Position, TuTensor};

/// Transpose engine input packet size in bytes.
const TRANSPOSE_INPUT_BYTES: usize = 32;

/// Transpose engine output packet size in bytes.
const TRANSPOSE_OUTPUT_BYTES: usize = 32;

/// Number of elements fetched per transpose packet for non 4-bit types.
const TRANSPOSE_ELEMENTS_PER_PACKET_NON_4BIT: usize = 8;

/// Number of elements fetched per transpose packet for 4-bit types.
const TRANSPOSE_ELEMENTS_PER_PACKET_4BIT: usize = 16;

/// Maximum size for transpose `in_rows` in bytes.
const TRANSPOSE_MAX_IN_ROWS_BYTES: usize = 8;

/// Valid `in_cols` values for the transpose engine (non 4-bit types).
const TRANSPOSE_VALID_IN_COLS: &[usize] = &[8, 16, 32];

/// Valid `in_cols` values for the transpose engine (4-bit types).
const TRANSPOSE_VALID_IN_COLS_4BIT: &[usize] = &[16, 32];

/// After the transpose engine.
#[derive(Debug)]
pub struct PositionTranspose;

impl Position for PositionTranspose {}

/// Tensor streamed after the transpose engine.
pub type TransposeTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionTranspose, D, Chip, Cluster, Slice, Time, Packet, B>;

// ANCHOR: transpose_impl
impl<'l, const T: Tu, P: CanApplyTranspose, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Performs the transpose operation.
    #[primitive(TuTensor::transpose)]
    pub fn transpose<OutTime: M, OutPacket: M>(
        self,
    ) -> TransposeTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_transpose::<D, Time, Packet, OutTime, OutPacket>();
        TransposeTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: transpose_impl

/// Validates hardware constraints for the transpose engine.
///
/// Constraints checked:
/// 1. `Packet` and `OutPacket` must be 32 bytes
/// 2. `in_rows` * sizeof(D) <= 8 bytes
/// 3. `in_cols` must be 8, 16, or 32 (4-bit: 16 or 32 only)
/// 4. `out_rows` <= `in_cols`
pub(crate) fn verify_transpose<D: Scalar, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    // Packet must be TRANSPOSE_INPUT_BYTES bytes.
    let packet_bytes = D::size_in_bytes_from_length(Packet::SIZE);
    assert_eq!(
        packet_bytes, TRANSPOSE_INPUT_BYTES,
        "Transpose input packet must be {TRANSPOSE_INPUT_BYTES} bytes, got {packet_bytes}"
    );

    // OutPacket must be TRANSPOSE_OUTPUT_BYTES bytes.
    let out_packet_bytes = D::size_in_bytes_from_length(OutPacket::SIZE);
    assert_eq!(
        out_packet_bytes, TRANSPOSE_OUTPUT_BYTES,
        "Transpose output packet must be {TRANSPOSE_OUTPUT_BYTES} bytes, got {out_packet_bytes}",
    );

    // OutPacket is `[in_rows # padding]`.
    // `in_rows` * sizeof(D) <= 8 bytes
    let in_rows = OutPacket::to_value().factorize().remove_padding();
    let in_rows_bytes = D::size_in_bytes_from_length(in_rows.size());
    assert!(
        in_rows_bytes <= TRANSPOSE_MAX_IN_ROWS_BYTES,
        "Transpose `in_rows` must be <= {TRANSPOSE_MAX_IN_ROWS_BYTES} bytes, got {in_rows_bytes}"
    );

    // `Time = [..., in_rows, packets_per_col]`
    // Check that `in_rows` matches OutPacket `in_rows`.
    let time = Time::to_value().factorize();
    let in_rows_division = time
        .clone()
        .divide(in_rows.clone())
        .exact_checked()
        .unwrap_or_else(|_| panic!("Transpose `in_rows` ({in_rows}) must be present in the input Time ({time})"));

    // `in_cols` = `packets_per_col` * `elements_per_packet` must be in {8, 16, 32} (4-bit: {16, 32})
    let Tuple2(time_outer, packets_per_col) = time.split_at(in_rows_division.division_terms[0].dividend_stride);
    let Tuple2(time_outer, _) = time_outer.split_at(in_rows.size());
    let (elements_per_packet, valid_in_cols) = match D::BITS {
        4 => (TRANSPOSE_ELEMENTS_PER_PACKET_4BIT, TRANSPOSE_VALID_IN_COLS_4BIT),
        _ => (TRANSPOSE_ELEMENTS_PER_PACKET_NON_4BIT, TRANSPOSE_VALID_IN_COLS),
    };
    let in_cols = packets_per_col.size() * elements_per_packet;
    assert!(
        valid_in_cols.contains(&in_cols),
        "Transpose `in_cols` size ({in_cols}) must be one of {valid_in_cols:?} for {}-bit type",
        D::BITS,
    );

    let elements_per_packet = Packet::to_value().factorize().pad(elements_per_packet);
    let out_time = OutTime::to_value().factorize();

    // `OutTime = [time_outer, packets_per_col, elements_per_packet]`.
    // `elements_per_packet` may be sliced: `[in_cols x in_rows] → [out_rows x in_rows]`.
    // Make sure only padding is removed.
    let Tuple2(out_time_outer, out_time_elements_per_packet) = if packets_per_col.size() > 1 {
        let ppc_division = out_time
            .clone()
            .divide(packets_per_col.clone())
            .exact_checked()
            .unwrap_or_else(|_| {
                panic!("Transpose `packets_per_col` ({packets_per_col}) not found in OutTime ({out_time})")
            });
        let Tuple2(outer, num_elems) = out_time.split_at(ppc_division.division_terms[0].dividend_stride);
        let Tuple2(outer, _) = outer.split_at(packets_per_col.size());
        Tuple2(outer, num_elems)
    } else {
        let out_rows = OutTime::SIZE / time_outer.size();
        out_time.split_at(out_rows)
    };
    let out_rows = packets_per_col
        .clone()
        .mul(out_time_elements_per_packet.clone())
        .normalize();
    let in_cols = packets_per_col.mul(elements_per_packet.clone()).normalize();
    assert!(
        out_rows.size() <= in_cols.size(),
        "Transpose `out_rows` ({}) must be <= `in_cols` ({})",
        out_rows.size(),
        in_cols.size(),
    );
    assert_eq!(
        out_time_elements_per_packet.remove_padding(),
        elements_per_packet.remove_padding(),
        "Transpose `out_rows` ({out_rows}) must match `in_cols` ({in_cols}) (excluding padding)",
    );

    // Outer Time axes should match outer OutTime axes.
    assert_eq!(
        time_outer, out_time_outer,
        "Transpose time mismatch: expected outer OutTime to be {time_outer}, got ({out_time_outer})"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bf16;

    mod valid {
        use super::*;
        axes![A = 4, B = 2, C = 8, D = 4, E = 8, F = 8, G = 2, X = 64, Y = 512];

        #[test]
        fn basic() {
            verify_transpose::<i8, m![C, F], m![E # 32], m![C, E], m![F # 32]>();
        }

        #[test]
        fn small() {
            // `elements_per_packet = B # 8` is sliced to `B`
            verify_transpose::<i8, m![A], m![B # 32], m![B], m![A # 32]>();
        }

        #[test]
        fn small_no_slicing() {
            verify_transpose::<i8, m![A], m![B # 32], m![B # 8], m![A # 32]>();
        }

        #[test]
        fn large_col() {
            verify_transpose::<i8, m![B, C, D], m![E # 32], m![B, D, E], m![C # 32]>();
        }

        #[test]
        fn bf16() {
            verify_transpose::<bf16, m![C, D], m![E # 16], m![C, E], m![D # 16]>();
        }
    }

    mod input_packet {
        use super::*;
        axes![C = 8, D = 8, E = 4, F = 16];

        #[test]
        #[should_panic(expected = "Transpose input packet must be 32 bytes, got 16")]
        fn invalid() {
            verify_transpose::<i8, m![C, D], m![E # 16], m![C, E], m![D # 32]>();
        }
    }

    mod output_packet {
        use super::*;
        axes![C = 8, D = 8, E = 8];

        #[test]
        #[should_panic(expected = "Transpose output packet must be 32 bytes, got 16")]
        fn invalid() {
            verify_transpose::<i8, m![C, D], m![E # 32], m![C, E], m![D # 16]>();
        }
    }

    mod in_rows {
        use super::*;
        axes![A = 4, C = 8, D = 8, E = 8, F = 32];

        #[test]
        #[should_panic(expected = "Transpose `in_rows` must be <= 8 bytes, got 16")]
        fn invalid_i4() {
            verify_transpose::<i4, m![C, D], m![E # 64], m![C, E], m![F # 64]>();
        }

        #[test]
        #[should_panic(expected = "Transpose `in_rows` must be <= 8 bytes, got 32")]
        fn invalid_i8() {
            verify_transpose::<i8, m![C, D], m![E # 32], m![C, E], m![A, D]>();
        }

        #[test]
        #[should_panic(expected = "Transpose `in_rows` must be <= 8 bytes, got 16")]
        fn invalid_bf16() {
            verify_transpose::<bf16, m![C, D], m![E # 16], m![C, E], m![D # 16]>();
        }

        #[test]
        #[should_panic(expected = "must be present in the input Time")]
        fn invalid_in_rows_not_in_time() {
            verify_transpose::<i8, m![C, D], m![E # 32], m![C, E], m![A # 32]>();
        }
    }

    mod in_cols {
        use super::*;
        axes![C = 8, D = 8, E = 8, F = 16, G = 4];

        #[test]
        #[should_panic(expected = "Transpose `in_cols` size (64) must be one of [16, 32] for 4-bit type")]
        fn invalid_i4() {
            verify_transpose::<i4, m![F, G], m![E # 64], m![G, E], m![F # 64]>();
        }

        #[test]
        #[should_panic(expected = "Transpose `in_cols` size (64) must be one of [8, 16, 32] for 8-bit type")]
        fn invalid_i8() {
            verify_transpose::<i8, m![C, D], m![E # 32], m![D, E], m![C # 32]>();
        }
    }

    mod out_time {
        use super::*;
        axes![A = 2, B = 2, C = 4, D = 2, E = 8, F = 8, G = 16];

        #[test]
        #[should_panic(expected = "Transpose time mismatch")]
        fn invalid_outer_mismatch() {
            verify_transpose::<i8, m![B, C, D], m![E # 32], m![A, D, E], m![C # 32]>();
        }

        #[test]
        #[should_panic(expected = "not found in OutTime")]
        fn invalid_missing_packets_per_col() {
            verify_transpose::<i8, m![C, D], m![E # 32], m![E], m![C # 32]>();
        }

        #[test]
        #[should_panic(expected = "must match")]
        fn invalid_wrong_axis() {
            verify_transpose::<i8, m![C, D], m![E # 32], m![D, F], m![C # 32]>();
        }

        #[test]
        #[should_panic(expected = "must match")]
        fn invalid_non_padding_resize() {
            // E = 4 discards non-padded elements
            verify_transpose::<i8, m![C], m![E # 32], m![E = 4], m![C # 32]>();
        }

        #[test]
        #[should_panic(expected = "must be <=")]
        fn invalid_out_rows_exceeds_in_cols() {
            verify_transpose::<i8, m![A], m![B # 32], m![B # 16], m![A # 32]>();
        }
    }
}
