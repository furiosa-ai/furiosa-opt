//! Lane Folder (`contract_lane`): folds `Lane` into the output stream,
//! producing the contraction pipeline's final [`ContractTensor`].

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::ContractionCast;
use crate::context::*;
use crate::engine::contraction::{ContractTensor, ContractTimeTensor};
use crate::scalar::*;
use crate::tensor::Tensor;

/// Contraction mode for the Lane Folder.
#[primitive(LaneMode)]
#[derive(Clone, Debug)]
pub enum LaneMode {
    /// Interleaved: outputs data element-by-element across all `Lane`s.
    Interleaved,
    /// Sequential: outputs reduced data in each `Lane` sequentially.
    Sequential,
}

impl std::fmt::Display for LaneMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaneMode::Interleaved => write!(f, "Interleaved"),
            LaneMode::Sequential => write!(f, "Sequential"),
        }
    }
}

// ANCHOR: contract_lane_def
impl<
    'l,
    const T: Tu,
    D: ContractionCast + MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Folds the `Lane` dimension into the output stream.
    /// `LaneMode::Interleaved` relocates `Lane` into `OutPacket`;
    /// `LaneMode::Sequential` relocates `Lane` into `OutTime`.
    #[primitive(ContractTimeTensor::contract_lane)]
    pub fn contract_lane<OutTime: M, OutPacket: M>(
        self,
        mode: LaneMode,
    ) -> ContractTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_contract_lane(
            Lane::to_value(),
            Time::to_value(),
            Packet::to_value(),
            OutTime::to_value(),
            OutPacket::to_value(),
            self.pre_reduce_time,
            mode,
        );
        // Finalize the carried operands with ONE fused contraction onto this stage's input mapping
        // `[Chip, Cluster, Slice, Lane, Time, Packet]` (Packet/Time were never actually reduced by the
        // earlier stages, only relabeled to their post-stage extents). The Lane fold relayout
        // (`transpose(false)`) then runs on the result, exactly as it always has.
        //
        // `out` is rebuilt here from this stage's own type params, independently of the `pre_reduce`
        // stashed by `contract_outer`; `Backend::contraction` reduces `pre_reduce` onto `out` via
        // `pre_reduce.carve(out)`, the same mapping-algebra carve `reduce` uses elsewhere, which is the
        // authority on whether `out` is a valid restriction of `pre_reduce` -- NOT a manual re-check here.
        // A naive per-symbol `.axes()` comparison is unsound for that: a contracted symbol can split
        // across a spatial slot this fold never touches (e.g. `Cluster`, carrying a `K`-fragment that
        // survives to `out` unreduced) and the slots that actually get contracted (`Time`/`Packet`,
        // carrying the rest of `K`); `pre_reduce`'s canonical `K` term then legitimately has a different
        // shape (wider modulo) than `out`'s, even though `out` is a correct restriction of `pre_reduce`.
        let contraction = self.inner;
        let out = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }]>::to_value();
        let reduced: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }], B> =
            Tensor::from_inner(B::contraction(
                &contraction.lhs,
                &contraction.rhs,
                &contraction.lhs_map,
                &contraction.rhs_map,
                &contraction.pre_reduce,
                &out,
            ));
        ContractTensor::new(self.ctx, reduced.transpose(false))
    }
}
// ANCHOR_END: contract_lane_def

/// Validates the Lane Folder.
///
/// Checks:
/// Validates the Lane Folder via [`furiosa_opt_lower::config_contract_lane`] (packet / time /
/// accumulator rules documented there).
pub(crate) fn verify_contract_lane(
    lane: Mapping,
    time: Mapping,
    packet: Mapping,
    out_time: Mapping,
    out_packet: Mapping,
    pre_reduce_time: Mapping,
    kind: LaneMode,
) {
    let mode = match kind {
        LaneMode::Interleaved => furiosa_opt_lower::LaneMode::Interleaved,
        LaneMode::Sequential => furiosa_opt_lower::LaneMode::Sequential,
    };
    furiosa_opt_lower::config_contract_lane(&lane, &time, &packet, &out_time, &out_packet, &pre_reduce_time, mode)
        .unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    mod out_packet_size {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![A]>::to_value(),
                <m![1]>::to_value(),
                <m![A]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![A]>::to_value(),
                LaneMode::Interleaved,
            );
        }
    }

    mod interleaved {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_padding() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![B # 4]>::to_value(),
                <m![1]>::to_value(),
                <m![B # 4]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B # 4]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![A # 8, B]>::to_value(),
                <m![D]>::to_value(),
                <m![A # 8, B, D]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![A # 8, B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_non_outermost() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![C, B]>::to_value(),
                <m![1]>::to_value(),
                <m![C, B]>::to_value(),
                <m![N]>::to_value(),
                <m![C, B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_four_rows() {
            verify_contract_lane(
                <m![M]>::to_value(),
                <m![C, B]>::to_value(),
                <m![1]>::to_value(),
                <m![C, B]>::to_value(),
                <m![M # 8]>::to_value(),
                <m![C, B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_all_time_reduced() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                <m![1]>::to_value(),
                <m![1]>::to_value(),
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                LaneMode::Interleaved,
            );
        }
    }

    mod sequential {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B, N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_padded_row() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B, N # 8]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_all_time_reduced() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                <m![1]>::to_value(),
                <m![N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![1]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![A # 8, B]>::to_value(),
                <m![1]>::to_value(),
                <m![A # 8, B, N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![A # 8, B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_padded_packet() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![M]>::to_value(),
                <m![B]>::to_value(),
                <m![M, N]>::to_value(),
                <m![B # 8]>::to_value(),
                <m![M]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_full_temporal_reduction() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                <m![D]>::to_value(),
                <m![N, D / 8]>::to_value(),
                <m![D % 8]>::to_value(),
                <m![1]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_multi_axis_reduction() {
            // Reduce A and C. post=[B].
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B, N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_wide_packet_reduced() {
            // pre=[A,N,B], post=[N,B] (A reduced). inner_time = N*B = 16. Lane = 1, Packet pads to 32,
            // so buffer = 1 * 16 * 32 = 512 <= 1024.
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![N, B]>::to_value(),
                <m![D]>::to_value(),
                <m![N, B, D / 8]>::to_value(),
                <m![D % 8]>::to_value(),
                <m![A, N, B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_wide_packet_at_capacity() {
            // pre=[A,M], post=[M] (A reduced). inner_time = M = 4. Lane = 8, Packet pads to 32,
            // so buffer = 8 * 4 * 32 = 1024 = accumulator capacity.
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![M]>::to_value(),
                <m![D]>::to_value(),
                <m![M, N, D / 8]>::to_value(),
                <m![D % 8]>::to_value(),
                <m![A, M]>::to_value(),
                LaneMode::Sequential,
            );
        }
    }
}
