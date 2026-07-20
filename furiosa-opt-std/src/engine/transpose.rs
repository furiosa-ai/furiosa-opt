//! Transpose Engine: packet-level transpose.
//!
//! Shape verification runs through the `verify_transpose` FFI entry below; this
//! module only carries the engine entry point and its typestate.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplyTranspose;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// After the transpose engine.
#[derive(Debug)]
pub struct PositionTranspose;

impl Position for PositionTranspose {}

/// Tensor streamed after the transpose engine.
pub type TransposeTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionTranspose, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TransposeTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_one_flit::<D, Packet>();
    }

    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

// ANCHOR: transpose_impl
// `D: MaterializableScalar` (see its doc) excludes i5/i9 stagings from transpose.
impl<
    'l,
    const T: Tu,
    P: CanApplyTranspose,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
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

/// Verifies a transpose op, panicking if the requested `(OutTime, OutPacket)`
/// is not realizable from `(Time, Packet)` on the transpose engine. The
/// implementation is linked through `furiosa-opt-lower`'s archive; the resolved
/// parameters are discarded here, only success or the rendered error matters.
pub(crate) fn verify_transpose<D: Scalar, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    furiosa_opt_lower::config_transpose(
        &Time::to_value(),
        &Packet::to_value(),
        &OutTime::to_value(),
        &OutPacket::to_value(),
        D::BITS,
    )
    .unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    //! These exercise the engine entry point through the verify FFI
    //! (accept = no panic, reject = panic with the verifier's message). The
    //! parameter-level forward-simulation oracle is kept with the verifier
    //! implementation.
    use super::*;
    use crate::scalar::bf16;

    mod valid {
        use super::*;
        axes![
            A = 4,
            B = 2,
            C = 8,
            D = 4,
            E = 8,
            F = 8,
            G = 2,
            X = 64,
            Y = 512,
            Esplit = 16,
            PPC = 2,
            IR = 8,
            EP = 8,
            EPHalf = 4,
            Dummy1 = 1,
            Esplit32 = 32,
            BB = 4,
            Cbig = 384,
            Bbig = 96,
            Dummy8 = 8
        ];

        #[test]
        fn transpose_basic() {
            verify_transpose::<i8, m![C, F], m![E # 32], m![C, E], m![F # 32]>();
        }

        #[test]
        fn transpose_small() {
            verify_transpose::<i8, m![A], m![B # 32], m![B], m![A # 32]>();
        }

        #[test]
        fn transpose_small_no_slicing() {
            verify_transpose::<i8, m![A], m![B # 32], m![B # 8], m![A # 32]>();
        }

        #[test]
        fn transpose_large_col() {
            verify_transpose::<i8, m![B, C, D], m![E # 32], m![B, D, E], m![C # 32]>();
        }

        #[test]
        fn transpose_bf16() {
            verify_transpose::<bf16, m![C, D], m![E # 16], m![C, E], m![D # 16]>();
        }

        #[test]
        fn transpose_padding_only_in_rows() {
            verify_transpose::<bf16, m![1], m![C % 8 # 16], m![C % 8], m![1 # 16]>();
        }

        #[test]
        fn transpose_split_symbol_time() {
            verify_transpose::<i8, m![C, D, Esplit / 8], m![Esplit % 8 # 32], m![C, Esplit], m![D # 32]>();
        }

        #[test]
        fn transpose_packets_per_col_2_no_slice() {
            verify_transpose::<i8, m![IR, PPC], m![EP # 32], m![PPC, EP], m![IR # 32]>();
        }

        #[test]
        fn transpose_padding_only_packets_per_col() {
            verify_transpose::<i8, m![IR, 1 # 2], m![EP # 32], m![1 # 2, EP], m![IR # 32]>();
        }

        #[test]
        fn transpose_size1_dummy_axis_as_in_rows() {
            verify_transpose::<i8, m![D, Dummy1], m![A # 32], m![D, A], m![Dummy1 # 32]>();
        }

        #[test]
        fn transpose_i4_basic() {
            verify_transpose::<i4, m![A], m![BB # 64], m![BB], m![A # 64]>();
        }

        #[test]
        fn transpose_f32_basic() {
            verify_transpose::<f32, m![B], m![C # 8], m![C], m![B # 8]>();
        }

        #[test]
        fn transpose_split_symbol_packets_per_col_4() {
            verify_transpose::<i8, m![C, D, Esplit32 / 8], m![Esplit32 % 8 # 32], m![C, Esplit32], m![D # 32]>();
        }

        #[test]
        fn transpose_packed_pair_in_rows() {
            verify_transpose::<i8, m![(A, B) # 16], m![C # 32], m![1 # 2, C], m![(A, B) # 32]>();
        }

        #[test]
        fn transpose_padded_strided_packets_per_col() {
            verify_transpose::<i8, m![D, A # 16 / 8], m![A # 16 % 8 # 32], m![1 # 2, A # 8], m![D # 32]>();
        }

        #[test]
        fn transpose_packed_pair_strided_packets_per_col() {
            verify_transpose::<i8, m![D, (A, B) # 16 / 8], m![(A, B) # 16 % 8 # 32], m![1 # 2, (A, B)], m![D # 32]>();
        }

        #[test]
        fn transpose_packed_pair_strided_bf16() {
            verify_transpose::<bf16, m![D, (A, B) # 16 / 8], m![(A, B) # 16 % 8 # 16], m![1 # 2, (A, B)], m![D # 16]>();
        }

        #[test]
        fn transpose_all_padding_no_terms() {
            verify_transpose::<i8, m![1 # 4], m![1 # 32], m![1 # 32], m![1 # 32]>();
        }

        #[test]
        fn transpose_mixed_padding_size_arithmetic() {
            verify_transpose::<i8, m![A, 1 # 4], m![EP # 32], m![1 # 4, EP], m![A # 32]>();
        }

        #[test]
        fn transpose_fc1_bias_prepared_split_dummy8() {
            verify_transpose::<bf16, m![Dummy8], m![1 # 16], m![Dummy8 / 4], m![Dummy8 % 4 # 16]>();
        }

        #[test]
        fn transpose_engine_14_bf16_multi_split() {
            verify_transpose::<
                bf16,
                m![Cbig / 4 % 24, Cbig / 96, Bbig / 24, Bbig / 8 % 3, Cbig % 4],
                m![Bbig % 8 # 16],
                m![Cbig / 4 % 24, Cbig / 96, Bbig],
                m![Cbig % 4 # 16],
            >();
        }
    }
}
