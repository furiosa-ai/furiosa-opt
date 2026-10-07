use furiosa_mapping::{FindAxisError, Mapping, MappingExt};

use crate::DM_WRITE_ALIGN_BYTES;

/// Slice operation and the constraints it must satisfy.
pub enum SliceRequest {
    Layout {
        axes: Vec<Mapping>,
        element: Mapping,
        output: Mapping,
    },
    AlignedStrides {
        axes: Vec<Mapping>,
        element: Mapping,
        output: Mapping,
        element_bits: usize,
    },
}

/// Derived output mapping and source strides for a valid slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlicePlan {
    pub output: Mapping,
    pub strides: Vec<usize>,
}

/// Why an axis slice cannot produce its requested output mapping.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SliceError {
    #[error("slice: cannot locate AxisToSlice {axis} in Element {element}: {source}")]
    AxisNotFound {
        axis: Mapping,
        element: Mapping,
        #[source]
        source: FindAxisError,
    },
    #[error("slice: AxisToSlice {axis} sits at stride {stride}, which does not divide Element {element}")]
    StrideDoesNotDivide {
        axis: Mapping,
        element: Mapping,
        stride: usize,
    },
    #[error("slice: AxisToSlice {axis} is not a complete axis of Element {element}; the located run contains {found}")]
    IncompleteAxis {
        axis: Mapping,
        element: Mapping,
        found: Mapping,
    },
    #[error("slice: axes {axes:?} overlap in Element {element}")]
    OverlappingAxes { axes: Vec<Mapping>, element: Mapping },
    #[error("slice: AxisToSlice {axis} must be the outermost axis of Element {element}; the outer run is {outer}")]
    NotOutermost {
        axis: Mapping,
        element: Mapping,
        outer: Mapping,
    },
    #[error("slice: the tail under AxisToSlice contains a must-not-write padding region: {tail}")]
    BottomHoleTail { tail: Mapping },
    #[error("slice: Element2 must be Element with AxisToSlice removed; expected {expected}, got {output}")]
    UnexpectedOutput { expected: Mapping, output: Mapping },
    #[error(
        "slice: AxisToSlice {axis} has stride {stride} elements ({stride_bits} bits) in Element \
         {element}; a DM output requires the sliced-axis stride to be aligned to {alignment_bytes} bytes"
    )]
    UnalignedDmStride {
        axis: Mapping,
        element: Mapping,
        stride: usize,
        stride_bits: u128,
        alignment_bytes: usize,
    },
}

/// Derives a slice plan after enforcing the requested layout constraints.
pub fn config_slice(request: SliceRequest) -> Result<SlicePlan, SliceError> {
    match request {
        SliceRequest::Layout { axes, element, output } => config_axes(&axes, &element, output),
        SliceRequest::AlignedStrides {
            axes,
            element,
            output,
            element_bits,
        } => {
            let plan = config_axes(&axes, &element, output)?;
            for (axis, &stride) in axes.iter().zip(&plan.strides) {
                ensure_dm_stride(axis, &element, stride, element_bits)?;
            }
            Ok(plan)
        }
    }
}

/// Configures an outermost DM slice and returns its single source stride.
pub fn config_outermost_dm_slice(
    axis: Mapping,
    element: Mapping,
    output: Mapping,
    element_bits: usize,
) -> Result<usize, SliceError> {
    let stride = element.find_axis(&axis).map_err(|source| SliceError::AxisNotFound {
        axis: axis.clone(),
        element: element.clone(),
        source,
    })?;
    let actual_output = slice_layout(&axis, &element)?.normalize();
    ensure_dm_stride(&axis, &element, stride, element_bits)?;
    let output = output.normalize();
    if actual_output != output {
        return Err(SliceError::UnexpectedOutput {
            expected: actual_output,
            output,
        });
    }

    let (outer, tail) = element.split_at(stride).map_err(|_| SliceError::StrideDoesNotDivide {
        axis: axis.clone(),
        element: element.clone(),
        stride,
    })?;
    if outer.normalize() != axis.normalize() {
        return Err(SliceError::NotOutermost { axis, element, outer });
    }
    if tail.has_bottom_pad() {
        return Err(SliceError::BottomHoleTail { tail });
    }

    Ok(stride)
}

fn config_axes(axes: &[Mapping], element: &Mapping, output: Mapping) -> Result<SlicePlan, SliceError> {
    let strides = axes
        .iter()
        .map(|axis| {
            element.find_axis(axis).map_err(|source| SliceError::AxisNotFound {
                axis: axis.clone(),
                element: element.clone(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = slice_axes_layout(axes, element)?.normalize();
    let output = output.normalize();
    if output != expected {
        return Err(SliceError::UnexpectedOutput { expected, output });
    }
    Ok(SlicePlan { output, strides })
}

fn ensure_dm_stride(axis: &Mapping, element: &Mapping, stride: usize, element_bits: usize) -> Result<(), SliceError> {
    let stride_bits = stride as u128 * element_bits as u128;
    let alignment_bits = DM_WRITE_ALIGN_BYTES as u128 * 8;
    if !stride_bits.is_multiple_of(alignment_bits) {
        return Err(SliceError::UnalignedDmStride {
            axis: axis.clone(),
            element: element.clone(),
            stride,
            stride_bits,
            alignment_bytes: DM_WRITE_ALIGN_BYTES,
        });
    }
    Ok(())
}

fn slice_axes_layout(axes: &[Mapping], element: &Mapping) -> Result<Mapping, SliceError> {
    let mut expected = element.clone();
    for axis in axes {
        match slice_layout(axis, &expected) {
            Ok(next) => expected = next,
            Err(SliceError::AxisNotFound { .. }) => {
                // Every axis was found in the input, so this axis overlapped one removed earlier.
                return Err(SliceError::OverlappingAxes {
                    axes: axes.to_vec(),
                    element: element.clone(),
                });
            }
            Err(SliceError::IncompleteAxis { axis, found, .. }) => {
                return Err(SliceError::IncompleteAxis {
                    axis,
                    element: element.clone(),
                    found,
                });
            }
            Err(error) => return Err(error),
        }
    }
    Ok(expected)
}

fn slice_layout(axis: &Mapping, element: &Mapping) -> Result<Mapping, SliceError> {
    let stride = element.find_axis(axis).map_err(|source| SliceError::AxisNotFound {
        axis: axis.clone(),
        element: element.clone(),
        source,
    })?;
    let (through_axis, inner) = element.split_at(stride).map_err(|_| SliceError::StrideDoesNotDivide {
        axis: axis.clone(),
        element: element.clone(),
        stride,
    })?;
    let (outer, found) = through_axis
        .split_at(axis.size())
        .map_err(|_| SliceError::IncompleteAxis {
            axis: axis.clone(),
            element: element.clone(),
            found: through_axis.clone(),
        })?;
    if found.normalize() != axis.normalize() {
        return Err(SliceError::IncompleteAxis {
            axis: axis.clone(),
            element: element.clone(),
            found,
        });
    }
    Ok(outer.pair(inner).normalize())
}
