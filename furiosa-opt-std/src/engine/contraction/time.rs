//! Time Reducer (`contract_time`): accumulator reduce across `Time`, shrinking
//! `Time` to `OutTime`.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::context::*;
use crate::engine::contraction::{ContractPacketTensor, ContractTimeTensor};
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
        // Carry the deferred operands forward unreduced: the fused contraction at `contract_lane`
        // performs this Time reduction too. This stage only re-types the carrier to `OutTime`.
        ContractTimeTensor::new(self.ctx, self.inner, Time::to_value())
    }
}
// ANCHOR_END: contract_time_def

/// Validates `.contract_time()` via [`furiosa_opt_lower::config_contract_time`] (order / padding
/// preservation rules documented there).
pub(crate) fn verify_contract_time(time: Mapping, out_time: Mapping) {
    furiosa_opt_lower::config_contract_time(&time, &out_time).unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];
    axes![L1 = 192, Bat = 4096, W = 12];

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

        /// Reducing an axis that sits BETWEEN two digits of a padded axis. The division then splits
        /// the padded axis into intra-axis sub-terms, and a sub-term carries no padding of its own,
        /// so checking every term against the per-axis padding rejects a contraction the engine
        /// performs. Only the terms that land on an axis boundary carry padding to compare.
        #[test]
        fn valid_reduce_between_the_digits_of_a_padded_axis() {
            verify_contract_time(
                <m![L1 # 256 / 32 % 4, Bat / 32 % 2, Bat / 64 % 2, W % 6, W / 6, L1 # 256 / 8 % 4]>::to_value(),
                <m![L1 # 256 / 32 % 4, Bat / 32 % 2, Bat / 64 % 2, L1 # 256 / 8 % 4]>::to_value(),
            );
        }
    }
}
