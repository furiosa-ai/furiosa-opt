//! Time Reducer (`contract_time`): accumulator reduce across `Time`, shrinking
//! `Time` to `OutTime`.

use furiosa_mapping::*;
use furiosa_opt_lower::{DivideTerm, config_divide_exact};
use furiosa_opt_macro::primitive;

use crate::context::*;
use crate::engine::contraction::{ContractPacketTensor, ContractTimeTensor, padding_per_stride};
use crate::runtime::Backend;
use crate::scalar::*;

// ANCHOR: contract_time_def
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Accumulates per-cycle contractions over the `Time` dimension via the shared
    /// accumulator buffer, shrinking input `Time` to `OutTime`. The axes present in
    /// `Time` but absent from `OutTime` are reduce-added.
    #[primitive(ContractPacketTensor::contract_time)]
    pub fn contract_time<OutTime: M>(
        self,
    ) -> ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, OutTime, Packet, B> {
        verify_contract_time(Time::to_value(), OutTime::to_value());
        ContractTimeTensor {
            ctx: self.ctx,
            inner: self.inner.reduce_add(),
            pre_reduce_time: Time::to_value(),
        }
    }
}
// ANCHOR_END: contract_time_def

/// Validates the `OutTime ⊆ Time` constraint for `.contract_time()`.
///
/// The axes in `OutTime` must be a subset of the axes in `Time`, with their
/// relative order preserved. Axes present in `Time` but absent from `OutTime`
/// are summed away by `reduce_add`. Retained axes must also preserve their
/// padding from `Time`.
pub(crate) fn verify_contract_time(time: Mapping, out_time: Mapping) {
    // The outer portion of `Time` should divide `Time`.
    // Some axes can be reduced in temporal accumulation.
    let division_terms = config_divide_exact(&time, &out_time).unwrap_or_else(|_| {
        panic!(
            "contract_time: OutTime mismatch. Some axes present in Time are not present in OutTime: {time}, {out_time}"
        )
    });
    // Non-reduced axes must have their order preserved in `OutTime`.
    assert!(
        division_terms
            .windows(2)
            .all(|w| w[0].divisor_stride > w[1].divisor_stride),
        "contract_time: OutTime axes must follow the same order as the Time axes"
    );

    // Each retained axis in `out_time` must preserve its padding from `time`: the padded extent at
    // each cumulative stride (`padding_per_stride`) must line the retained axes up edge-to-edge.
    let time_padding_per_stride = padding_per_stride(&time);

    // Sort retained axes inner-to-outer by their position in `out_time` (divisor).
    let mut sorted_divisions: Vec<&DivideTerm> = division_terms.iter().collect();
    sorted_divisions.sort_by_key(|d| d.divisor_stride);

    // The first divisor should have a stride of 1.
    if let Some(first) = sorted_divisions.first() {
        assert_eq!(
            first.divisor_stride, 1,
            "contract_time: Padding mismatch. \
             OutTime {out_time} has unexpected leading padding not present in Time {time}"
        );
    }

    // For each retained axis, its padding end must equal the start of the next retained axis.
    for (pos, d) in sorted_divisions.iter().enumerate() {
        let expected_end = d.divisor_stride
            * time_padding_per_stride
                .get(&d.dividend_stride)
                .copied()
                .unwrap_or(d.resize);
        let end = sorted_divisions
            .get(pos + 1)
            .map_or(out_time.size(), |next| next.divisor_stride);
        assert_eq!(
            expected_end, end,
            "contract_time: Padding mismatch. \
             Non-reduced axes in OutTime {out_time} do not preserve padding from Time {time}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    mod contract_time_subset {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid_identity() {
            verify_contract_time(<m![A, B]>::to_value(), <m![A, B]>::to_value());
        }

        #[test]
        fn valid_reduce_inner() {
            verify_contract_time(<m![A, B]>::to_value(), <m![A]>::to_value());
        }

        #[test]
        fn valid_reduce_outer() {
            // Outer axis can be reduced too: verify_contract_lane handles cross-stage
            // checks; here we only verify the order/padding contract.
            verify_contract_time(<m![A, B]>::to_value(), <m![B]>::to_value());
        }
    }
}
