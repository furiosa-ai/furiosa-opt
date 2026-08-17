//! Time reducer (`contract_time`): the retained (non-reduced) axes of `Time` must survive into
//! `OutTime` in order and with their padding preserved.

use furiosa_mapping::Mapping;

use crate::DivideTerm;
use crate::verify::{padded_extent_at, padding_per_stride};

/// Why a time contraction is not realizable on the Time Reducer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractTimeError {
    /// `OutTime` does not divide `Time` (some `Time` axis is missing from `OutTime`).
    #[error("contract_time: OutTime {out_time} must divide Time {time} (some axis is missing): {reason}")]
    Indivisible {
        /// The input time.
        time: Mapping,
        /// The declared output time.
        out_time: Mapping,
        /// The underlying division failure.
        reason: String,
    },
    /// The retained axes do not keep their `Time` order in `OutTime`.
    #[error("contract_time: OutTime axes must follow the same order as the Time axes")]
    OrderNotPreserved,
    /// `OutTime` carries leading padding absent from `Time`.
    #[error(
        "contract_time: Padding mismatch. OutTime {out_time} has unexpected leading padding not present in Time {time}"
    )]
    LeadingPadding {
        /// The input time.
        time: Mapping,
        /// The declared output time.
        out_time: Mapping,
    },
    /// A retained axis does not preserve its `Time` padding in `OutTime`.
    #[error(
        "contract_time: Padding mismatch. Non-reduced axes in OutTime {out_time} do not preserve padding from Time {time}"
    )]
    PaddingNotPreserved {
        /// The input time.
        time: Mapping,
        /// The declared output time.
        out_time: Mapping,
    },
}

/// `OutTime` must divide `Time` exactly (reduced axes are the quotient); retained axes keep their
/// order and edge-to-edge padding.
pub fn config_contract_time(time: &Mapping, out_time: &Mapping) -> Result<(), ContractTimeError> {
    // Retained axes; the reduced axes are what the division drops.
    let division_terms =
        crate::config_divide_exact(time, out_time).map_err(|reason| ContractTimeError::Indivisible {
            time: time.clone(),
            out_time: out_time.clone(),
            reason,
        })?;

    // Non-reduced axes must preserve their order in `OutTime`.
    if !division_terms
        .windows(2)
        .all(|w| w[0].divisor_stride > w[1].divisor_stride)
    {
        return Err(ContractTimeError::OrderNotPreserved);
    }

    // Each retained axis in `out_time` must preserve its padding: the padded extent at each cumulative
    // stride must line the retained axes up edge-to-edge.
    let time_padding_per_stride = padding_per_stride(time);

    let mut boundaries: Vec<(&DivideTerm, usize)> = division_terms
        .iter()
        .filter_map(|term| padded_extent_at(&time_padding_per_stride, term).map(|extent| (term, extent)))
        .collect();
    // Edge-to-edge is a claim about the OUTPUT layout, so walk it in divisor order.
    boundaries.sort_by_key(|(term, _)| term.divisor_stride);

    if let Some((first, _)) = boundaries.first()
        && first.divisor_stride != 1
    {
        return Err(ContractTimeError::LeadingPadding {
            time: time.clone(),
            out_time: out_time.clone(),
        });
    }

    for (pos, (term, extent)) in boundaries.iter().enumerate() {
        let end = boundaries
            .get(pos + 1)
            .map_or(out_time.size(), |(next, _)| next.divisor_stride);
        if term.divisor_stride * extent != end {
            return Err(ContractTimeError::PaddingNotPreserved {
                time: time.clone(),
                out_time: out_time.clone(),
            });
        }
    }
    Ok(())
}
