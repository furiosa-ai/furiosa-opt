//! Time reducer (`contract_time`): the retained (non-reduced) axes of `Time` must survive into
//! `OutTime` in order and with their padding preserved.

use furiosa_mapping::Mapping;

use crate::verify::{padded_extent_at, padding_per_stride};
use crate::{DivideError, DivideInput, DivideTerm};

/// Why a time contraction is not realizable on the Time Reducer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractTimeError {
    /// `OutTime` does not divide `Time` (some `Time` axis is missing from `OutTime`).
    #[error("contract_time: OutTime {out_time} must divide Time {in_time} (some axis is missing): {source}")]
    Indivisible {
        in_time: Mapping,
        out_time: Mapping,
        #[source]
        source: DivideError,
    },
    /// The retained axes do not keep their `Time` order in `OutTime`.
    #[error("contract_time: OutTime axes must follow the same order as the Time axes")]
    OrderNotPreserved,
    /// `OutTime` carries leading padding absent from `Time`.
    #[error(
        "contract_time: Padding mismatch. OutTime {out_time} has unexpected leading padding not present in Time {in_time}"
    )]
    LeadingPadding { in_time: Mapping, out_time: Mapping },
    /// A retained axis does not preserve its `Time` padding in `OutTime`.
    #[error(
        "contract_time: Padding mismatch. Non-reduced axes in OutTime {out_time} do not preserve padding from Time {in_time}"
    )]
    PaddingNotPreserved { in_time: Mapping, out_time: Mapping },
}

/// Inputs used to configure a time contraction.
pub struct ContractTimeInput {
    pub in_time: Mapping,
    pub out_time: Mapping,
}

/// `OutTime` must divide `Time` exactly (reduced axes are the quotient); retained axes keep their
/// order and edge-to-edge padding.
pub fn config_contract_time(input: ContractTimeInput) -> Result<(), ContractTimeError> {
    let ContractTimeInput { in_time, out_time } = input;
    // Retained axes; the reduced axes are what the division drops.
    let division_terms = crate::config_divide_exact(DivideInput {
        dividend: in_time.clone(),
        divisor: out_time.clone(),
    })
    .map_err(|source| ContractTimeError::Indivisible {
        in_time: in_time.clone(),
        out_time: out_time.clone(),
        source,
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
    let time_padding_per_stride = padding_per_stride(&in_time);

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
            in_time: in_time.clone(),
            out_time: out_time.clone(),
        });
    }

    for (pos, (term, extent)) in boundaries.iter().enumerate() {
        let end = boundaries
            .get(pos + 1)
            .map_or(out_time.size(), |(next, _)| next.divisor_stride);
        if term.divisor_stride * extent != end {
            return Err(ContractTimeError::PaddingNotPreserved {
                in_time: in_time.clone(),
                out_time: out_time.clone(),
            });
        }
    }
    Ok(())
}
