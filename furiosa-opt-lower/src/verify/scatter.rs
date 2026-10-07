use furiosa_mapping::{FindAxisError, Mapping, MappingExt, SequencerError, SequencerMode, sequence};

use super::indirect::{DmPlacement, holds_one_copy};

/// What one index entry writes into a destination, and the stride an offset counts in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScatterPayload {
    /// Destination element stride between adjacent indexed axis positions.
    pub indexed_axis_stride: usize,
    /// The destination with the indexed axis taken out.
    pub payload: Mapping,
}

impl ScatterPayload {
    /// Device byte stride between indexed-axis positions, the unit a byte-offset index counts in.
    pub fn indexed_axis_stride_bytes(&self, element_bits: usize) -> Result<usize, super::ElementSizeError> {
        super::size_in_bytes(element_bits, self.indexed_axis_stride)
    }
}

/// Invalid scatter mapping for one indirect destination region.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ScatterValidationError {
    #[error("scatter updates {updates} cannot supply payload {payload} once per entry of domain {domain}: {error}")]
    UpdatesUnplaceable {
        updates: Box<Mapping>,
        payload: Box<Mapping>,
        domain: Box<Mapping>,
        error: Box<SequencerError>,
    },
    #[error("scatter indexed axis must name an axis")]
    EmptyIndexedAxis,
    #[error("scatter indexed axis {indexed_axis} must be consecutive in destination mapping {destination}: {error}")]
    IndexedAxisNotConsecutive {
        destination: Mapping,
        indexed_axis: Mapping,
        error: FindAxisError,
    },
    /// The indexed axis region is located in the destination but is not a digit it can give up.
    #[error(
        "scatter indexed axis {indexed_axis} is located in destination mapping {destination} but cannot be taken out of it: {error}"
    )]
    IndexedAxisNotRemovable {
        destination: Mapping,
        indexed_axis: Mapping,
        error: FindAxisError,
    },
    #[error(
        "scatter source payload {source_payload} cannot be placed in destination payload {destination_payload}: {error}"
    )]
    PayloadPlacement {
        source_payload: Mapping,
        destination_payload: Mapping,
        error: SequencerError,
    },
}

/// Checks that the updates supply `payload` once per entry of `domain`.
///
/// The sequencer places the two, so neither has to be one consecutive region of the updates: a
/// domain may be split around a payload axis, or spread across the cluster, slice and element
/// mappings.
pub fn validate_scatter_updates(
    updates: DmPlacement<'_>,
    payload: &Mapping,
    domain: &Mapping,
) -> Result<(), ScatterValidationError> {
    let updates = updates.to_mapping();
    let stream = payload.clone().pair(domain.clone()).normalize();
    sequence(&[&updates], &[&stream], SequencerMode::Read).map_err(|error| {
        ScatterValidationError::UpdatesUnplaceable {
            updates: Box::new(updates.clone()),
            payload: Box::new(payload.clone()),
            domain: Box::new(domain.clone()),
            error: Box::new(error),
        }
    })?;
    Ok(())
}

/// Derives one entry's destination payload, retaining padding and indexed-axis stride.
pub fn scatter_payload(
    destination: &Mapping,
    indexed_axis: &Mapping,
) -> Result<ScatterPayload, ScatterValidationError> {
    // A placement holding one copy names no position of its own, so there is nothing to index.
    if holds_one_copy(indexed_axis) {
        return Err(ScatterValidationError::EmptyIndexedAxis);
    }
    // An axis the destination does not hold, or holds in pieces, fails the same call.
    let indexed_axis_stride =
        destination
            .find_axis(indexed_axis)
            .map_err(|error| ScatterValidationError::IndexedAxisNotConsecutive {
                destination: destination.clone(),
                indexed_axis: indexed_axis.clone(),
                error,
            })?;
    // Finding an axis does not guarantee it can be removed as one region.
    let payload = destination
        .remove_axis(indexed_axis)
        .map_err(|error| ScatterValidationError::IndexedAxisNotRemovable {
            destination: destination.clone(),
            indexed_axis: indexed_axis.clone(),
            error,
        })?
        .normalize();
    Ok(ScatterPayload {
        indexed_axis_stride,
        payload,
    })
}

/// Checks that the source payload lands in what the indexed axis leaves of the destination.
pub fn validate_scatter_payload(
    source_payload: &Mapping,
    destination: &ScatterPayload,
) -> Result<Mapping, ScatterValidationError> {
    let source_payload = source_payload.clone().normalize();
    sequence(&[&source_payload], &[&destination.payload], SequencerMode::Write).map_err(|error| {
        ScatterValidationError::PayloadPlacement {
            source_payload: source_payload.clone(),
            destination_payload: destination.payload.clone(),
            error,
        }
    })?;
    Ok(source_payload)
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![K = 4, A = 2, C = 8, D = 3, E = 2];

    #[test]
    fn accepts_one_indexed_axis() {
        let payload = <m![D]>::to_value();
        let destination = scatter_payload(&<m![C, D]>::to_value(), &<m![C]>::to_value()).unwrap();
        validate_scatter_payload(&payload, &destination).unwrap();

        assert_eq!(destination.payload, <m![D]>::to_value());
        assert_eq!(destination.indexed_axis_stride, D::SIZE);
    }

    #[test]
    fn accepts_one_consecutive_indexed_region() {
        let payload = <m![D]>::to_value();
        let destination = scatter_payload(&<m![C, E, D]>::to_value(), &<m![C, E]>::to_value()).unwrap();
        validate_scatter_payload(&payload, &destination).unwrap();

        assert_eq!(destination.payload, <m![D]>::to_value());
        assert_eq!(destination.indexed_axis_stride, D::SIZE);
    }

    #[test]
    fn removes_a_target_between_payload_axes() {
        let payload = <m![A, D]>::to_value();
        let destination = scatter_payload(&<m![A, C, D]>::to_value(), &<m![C]>::to_value()).unwrap();

        assert_eq!(
            validate_scatter_payload(&payload, &destination).unwrap(),
            payload.normalize()
        );
    }

    #[test]
    fn rejects_a_separated_indexed_axis() {
        let error = scatter_payload(&<m![C, D, E]>::to_value(), &<m![C, E]>::to_value()).unwrap_err();

        assert!(
            matches!(
                error,
                ScatterValidationError::IndexedAxisNotConsecutive {
                    error: FindAxisError::ScatteredInMapping,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn rejects_a_target_absent_from_the_destination() {
        let error = scatter_payload(&<m![C, D]>::to_value(), &<m![E]>::to_value()).unwrap_err();

        assert!(
            matches!(
                error,
                ScatterValidationError::IndexedAxisNotConsecutive {
                    error: FindAxisError::NotInMapping,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn rejects_a_payload_the_destination_cannot_hold() {
        let destination = scatter_payload(&<m![C, D]>::to_value(), &<m![C]>::to_value()).unwrap();
        let error = validate_scatter_payload(&<m![A, D]>::to_value(), &destination).unwrap_err();

        assert!(
            matches!(error, ScatterValidationError::PayloadPlacement { .. }),
            "{error}"
        );
    }

    fn updates_hold(updates: &Mapping, payload: &Mapping, domain: &Mapping) -> Result<(), ScatterValidationError> {
        let whole = <m![1]>::to_value();
        let placement = DmPlacement {
            cluster: &whole,
            slice: &whole,
            element: updates,
        };
        validate_scatter_updates(placement, payload, domain)
    }

    /// The sequencer places the domain and the payload, so neither has to be one consecutive region.
    #[test]
    fn accepts_a_domain_split_around_a_payload_axis() {
        updates_hold(
            &<m![K / 2, A, K % 2, D]>::to_value(),
            &<m![A, D]>::to_value(),
            &<m![K]>::to_value(),
        )
        .unwrap();
    }

    #[test]
    fn accepts_updates_that_pad_around_what_they_supply() {
        updates_hold(
            &<m![1 # 64, K, D]>::to_value(),
            &<m![D]>::to_value(),
            &<m![K]>::to_value(),
        )
        .unwrap();
    }

    #[test]
    fn rejects_updates_holding_a_cell_no_entry_reads() {
        let error = updates_hold(&<m![K, D, E]>::to_value(), &<m![D]>::to_value(), &<m![K]>::to_value()).unwrap_err();

        assert!(
            matches!(error, ScatterValidationError::UpdatesUnplaceable { .. }),
            "{error}"
        );
    }

    #[test]
    fn rejects_an_empty_indexed_axis() {
        assert_eq!(
            scatter_payload(&<m![C, D]>::to_value(), &<m![1]>::to_value()),
            Err(ScatterValidationError::EmptyIndexedAxis)
        );
    }
}
