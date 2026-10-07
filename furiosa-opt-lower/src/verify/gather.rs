//! Checks a gather's payload, index domain, and output placement at their declaration stages.

use furiosa_mapping::{FindAxisError, Mapping, MappingExt, SequencerError, SequencerMode, sequence};

use super::indirect::DmPlacement;

/// What one index entry reads out of a table, and the stride an offset counts in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatherPayload {
    /// The table with the indexed axis removed, padding included.
    pub payload: Mapping,
    /// Source element stride between adjacent indexed-axis positions.
    pub indexed_axis_stride: usize,
}

/// Invalid table or indexed-axis mapping for a gather payload.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum GatherMappingError {
    #[error("gather indexed axis {indexed_axis} must be consecutive in table mapping {table}: {error}")]
    IndexedAxisNotConsecutive {
        table: Mapping,
        indexed_axis: Mapping,
        error: FindAxisError,
    },
    /// The indexed axis is located but cannot be removed as one region.
    #[error(
        "gather indexed axis {indexed_axis} is located in table mapping {table} but cannot be taken out of it: {error}"
    )]
    IndexedAxisNotRemovable {
        table: Mapping,
        indexed_axis: Mapping,
        error: FindAxisError,
    },
    #[error(
        "gather payload must be table mapping {table} with indexed axis {indexed_axis} removed; \
         expected {expected}, declared {declared}"
    )]
    PayloadMismatch {
        table: Box<Mapping>,
        indexed_axis: Box<Mapping>,
        expected: Box<Mapping>,
        declared: Box<Mapping>,
    },
}

/// Invalid output placement for a gather payload and index domain.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum GatherValidationError {
    #[error("gather output {output} cannot hold payload {payload} once per entry of domain {domain}: {error}")]
    OutputPlacement {
        payload: Box<Mapping>,
        domain: Box<Mapping>,
        output: Box<Mapping>,
        error: Box<SequencerError>,
    },
}

/// Derives one entry's payload, retaining the table's padding.
pub fn gather_payload(table: &Mapping, indexed_axis: &Mapping) -> Result<GatherPayload, GatherMappingError> {
    // `find_axis` decides the only rule: the region is consecutive. How many axes name it does not
    // matter, and an axis the table does not hold at all fails the same call.
    let indexed_axis_stride =
        table
            .find_axis(indexed_axis)
            .map_err(|error| GatherMappingError::IndexedAxisNotConsecutive {
                table: table.clone(),
                indexed_axis: indexed_axis.clone(),
                error,
            })?;
    // A located axis may still be inseparable from its surrounding mapping.
    let payload = table
        .remove_axis(indexed_axis)
        .map_err(|error| GatherMappingError::IndexedAxisNotRemovable {
            table: table.clone(),
            indexed_axis: indexed_axis.clone(),
            error,
        })?;
    Ok(GatherPayload {
        payload: payload.normalize(),
        indexed_axis_stride,
    })
}

/// Checks the payload a kernel declared against the one its table and indexed axis determine.
pub fn validate_gather_payload(
    table: &Mapping,
    indexed_axis: &Mapping,
    declared: &Mapping,
) -> Result<GatherPayload, GatherMappingError> {
    let derived = gather_payload(table, indexed_axis)?;
    let declared_normal = declared.clone().normalize();
    if derived.payload != declared_normal {
        return Err(GatherMappingError::PayloadMismatch {
            table: Box::new(table.clone()),
            indexed_axis: Box::new(indexed_axis.clone()),
            expected: Box::new(derived.payload),
            declared: Box::new(declared_normal),
        });
    }
    Ok(derived)
}

/// Checks that the output holds the payload once per domain entry.
pub fn validate_gather_output(
    payload: &Mapping,
    domain: &Mapping,
    output: DmPlacement<'_>,
) -> Result<(), GatherValidationError> {
    let output = output.to_mapping();
    let stream = payload.clone().pair(domain.clone()).normalize();
    sequence(&[&output], &[&stream], SequencerMode::Write).map_err(|error| GatherValidationError::OutputPlacement {
        payload: Box::new(payload.clone()),
        domain: Box::new(domain.clone()),
        output: Box::new(output),
        error: Box::new(error),
    })?;
    Ok(())
}

impl GatherPayload {
    /// Device byte stride between indexed-axis positions.
    pub fn indexed_axis_stride_bytes(&self, element_bits: usize) -> Result<usize, super::ElementSizeError> {
        super::size_in_bytes(element_bits, self.indexed_axis_stride)
    }
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![A = 2, K = 4, D = 2, V0 = 2, V1 = 4, CL = 2, WIDE = 10];

    fn payload(table: Mapping, indexed_axis: Mapping) -> Result<GatherPayload, GatherMappingError> {
        gather_payload(&table, &indexed_axis)
    }

    #[test]
    fn the_payload_is_the_table_without_the_indexed_axis() {
        let result = payload(<m![A, K, D]>::to_value(), <m![K]>::to_value()).unwrap();

        assert_eq!(result.payload, <m![A, D]>::to_value().normalize());
        assert_eq!(result.indexed_axis_stride, D::SIZE);
    }

    #[test]
    fn the_payload_keeps_the_tables_padding() {
        let padded_axis = payload(<m![A # 8, K, D]>::to_value(), <m![K]>::to_value()).unwrap();
        assert_eq!(padded_axis.payload, <m![A # 8, D]>::to_value().normalize());
        assert_eq!(padded_axis.indexed_axis_stride, D::SIZE);

        let pad_only = payload(<m![1 # 64, K, D]>::to_value(), <m![K]>::to_value()).unwrap();
        assert_eq!(pad_only.payload, <m![1 # 64, D]>::to_value().normalize());
    }

    #[test]
    fn a_declared_payload_may_use_any_spelling_of_the_same_mapping() {
        let table = <m![1 # 64, K, D]>::to_value();
        let obvious = <m![1 # 64, D]>::to_value();

        let checked = validate_gather_payload(&table, &<m![K]>::to_value(), &obvious).unwrap();

        assert_eq!(checked.payload, obvious.normalize());
        validate_gather_payload(&table, &<m![K]>::to_value(), &checked.payload).unwrap();
    }

    #[test]
    fn rejects_a_declared_payload_that_keeps_the_indexed_axis() {
        let error = validate_gather_payload(
            &<m![A, K, D]>::to_value(),
            &<m![K]>::to_value(),
            &<m![A, K, D]>::to_value(),
        )
        .unwrap_err();

        assert!(matches!(error, GatherMappingError::PayloadMismatch { .. }), "{error}");
    }

    /// Two consecutive table axes are one indexed region, and the payload is what is left.
    #[test]
    fn accepts_an_indexed_axis_naming_two_consecutive_axes() {
        let result = payload(<m![A, K, D]>::to_value(), <m![A, K]>::to_value()).unwrap();

        assert_eq!(result.payload, <m![D]>::to_value().normalize());
        assert_eq!(result.indexed_axis_stride, D::SIZE);
    }

    #[test]
    fn indexes_by_one_digit_of_a_table_axis() {
        let result = payload(<m![K, D]>::to_value(), <m![K / 2]>::to_value()).unwrap();

        assert_eq!(result.payload, <m![K % 2, D]>::to_value().normalize());
        assert_eq!(result.indexed_axis_stride, <m![K % 2, D]>::SIZE);
    }

    #[test]
    fn rejects_an_indexed_axis_the_table_holds_in_pieces() {
        let error = payload(<m![K / 2, D, K % 2]>::to_value(), <m![K]>::to_value()).unwrap_err();

        assert!(
            matches!(
                error,
                GatherMappingError::IndexedAxisNotConsecutive {
                    error: FindAxisError::ScatteredInMapping,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn rejects_an_indexed_axis_the_table_steps_through_but_cannot_give_up() {
        // `% 4` keeps only part of the padded axis, so locating it cannot extract it.
        let error = payload(<m![WIDE]>::to_value(), <m![WIDE # 12 % 4]>::to_value()).unwrap_err();

        assert!(
            matches!(error, GatherMappingError::IndexedAxisNotRemovable { .. }),
            "{error}"
        );
    }

    #[test]
    fn gather_output_accepts_a_permutation() {
        validate_gather_output(
            &<m![D]>::to_value(),
            &<m![V0, V1]>::to_value(),
            DmPlacement {
                cluster: &<m![V1]>::to_value(),
                slice: &<m![V0]>::to_value(),
                element: &<m![D]>::to_value(),
            },
        )
        .unwrap();
    }

    #[test]
    fn gather_output_rejects_a_mapping_that_drops_a_domain_axis() {
        assert!(matches!(
            validate_gather_output(
                &<m![D]>::to_value(),
                &<m![V0, V1]>::to_value(),
                DmPlacement {
                    cluster: &<m![1]>::to_value(),
                    slice: &<m![V0]>::to_value(),
                    element: &<m![D]>::to_value(),
                },
            ),
            Err(GatherValidationError::OutputPlacement { .. }),
        ));
    }
}
