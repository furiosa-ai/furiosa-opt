//! Commit Engine: Tensor Unit stream to DM.
//!
//! Drains any flit-normalized `TuTensor` (positions starting from
//! [`super::collect::PositionCollect`]) into a [`DmTensor`] (or into an
//! existing mutable view).

use abi_stable::std_types::Tuple2;
use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::{CanApplyCommit, FLIT_BYTES, exact_div};
use crate::runtime::Backend;
use crate::scalar::*;
use crate::tensor::memory::{Address, DmTensor, DmTensorViewMut};
use crate::tensor::tu::TuTensor;

/// Valid output packet sizes for the commit engine in bytes.
const COMMIT_OUT_PACKET_SIZES: [usize; 4] = [8, 16, 24, 32];

// ANCHOR: commit_impl
impl<'l, const T: Tu, P: CanApplyCommit, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Commits to data memory at `address`.
    #[primitive(TuTensor::commit)]
    pub fn commit<Element: M>(self, address: Address) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        verify_commit::<D, Time, Packet, Element>();
        DmTensor::new(self.inner.transpose(false), address)
    }

    /// Commits to a mutable tensor view in data memory.
    #[primitive(TuTensor::commit_view)]
    pub fn commit_view<Element: M>(self, mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>) {
        verify_commit::<D, Time, Packet, Element>();
        dst.inner.write_transpose(self.inner.view(), false);
    }
}
// ANCHOR_END: commit_impl

/// Verifies commit engine constraints.
///
/// Constraints checked:
/// 1. Input packet must be exactly one flit (32 bytes).
/// 2. Output packet must be 8, 16, 24, or 32 bytes.
/// 3. Truncation may only remove elements from Packet.
pub(crate) fn verify_commit<D: Scalar, Time: M, Packet: M, Element: M>() {
    // Input packet must be exactly one flit.
    let packet_bytes = D::size_in_bytes_from_length(Packet::SIZE);
    assert_eq!(
        packet_bytes, FLIT_BYTES,
        "Commit input packet must be exactly {FLIT_BYTES} bytes (one flit), got {packet_bytes}",
    );

    // Time can be transposed.
    let Tuple2(time, packet) = Element::to_value()
        .factorize()
        .split_at(exact_div(Element::SIZE, Time::SIZE).expect("Commit element size does not divide time size"));
    let input_time = Time::to_value().factorize().normalize();
    if input_time.clone().divide(time.clone()).exact_checked().is_err()
        || time.clone().divide(input_time.clone()).exact_checked().is_err()
    {
        panic!("Commit output Time ({time}) is not a valid transpose of the input Time ({input_time})");
    }

    // Output packet must be 8, 16, 24, or 32 bytes.
    let out_packet_elements = Element::SIZE / Time::SIZE;
    let out_packet_bytes = D::size_in_bytes_from_length(out_packet_elements);
    assert!(
        COMMIT_OUT_PACKET_SIZES.contains(&out_packet_bytes),
        "Commit output packet must be one of {COMMIT_OUT_PACKET_SIZES:?} bytes, got {out_packet_bytes}",
    );

    // The resulting packet can be a slice of Packet by `commit_in_size`.
    let expected_packet = Packet::to_value().factorize();
    assert!(
        packet.is_resize_of(&expected_packet),
        "Commit packet mismatch. Expected {expected_packet} or a truncation of it, got {packet}",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bf16;

    mod valid {
        use super::*;

        axes![M = 4, N = 8, A = 4, B = 3, C = 4];

        #[test]
        fn full_truncation() {
            verify_commit::<i8, m![A, B, C], m![N # 32], m![A, B, C, N]>();
        }

        #[test]
        fn partial_truncation() {
            verify_commit::<i8, m![M], m![N # 32], m![M, N # 16]>();
        }

        #[test]
        fn no_truncation() {
            verify_commit::<i8, m![M], m![N # 32], m![M, N # 32]>();
        }

        #[test]
        fn bf16() {
            verify_commit::<bf16, m![M], m![N # 16], m![M, N]>();
        }

        #[test]
        fn f32() {
            verify_commit::<f32, m![M], m![N # 8], m![M, N]>();
        }

        #[test]
        fn single_time_step() {
            verify_commit::<i8, m![1], m![N # 32], m![N # 8]>();
        }

        #[test]
        fn non_padding_resize() {
            verify_commit::<bf16, m![1], m![N # 16], m![N = 4]>();
        }

        #[test]
        fn time_transpose() {
            verify_commit::<bf16, m![A # 32, B], m![N # 16], m![B, A # 32, N = 4]>();
        }
    }

    mod invalid {
        use super::*;

        axes![M = 4, N = 8, X = 8, Y = 4, Z = 2];
    }
}
