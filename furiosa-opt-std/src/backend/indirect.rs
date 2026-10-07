//! Index units and validation for indirect gather/scatter operations.

/// The unit of an indirect DMA's `i32` index values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexUnit {
    /// Raw positions along the indexed axis, as held in SPM.
    Positions,
    /// Byte offsets, carrying the byte stride of one indexed-axis position.
    Bytes(std::num::NonZeroUsize),
}

/// Invalid indirect index value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IndexValueError {
    /// A negative index value.
    #[error("scatter/gather index at cell {cell} must be non-negative, got {raw}")]
    Negative {
        /// Cell of the index list holding the value.
        cell: usize,
        /// The value as the index holds it.
        raw: i32,
    },
    /// A byte offset that is not an indexed-axis position boundary.
    #[error("scatter/gather index at cell {cell} must be a multiple of byte stride {stride}, got {raw}")]
    OffBoundary {
        /// Cell of the index list holding the value.
        cell: usize,
        /// The value as the index holds it.
        raw: i32,
        /// Bytes one position spans.
        stride: usize,
    },
    /// A decoded position outside the indexed axis.
    #[error("scatter/gather index at cell {cell} selects position {position} of an axis with {axis_size}")]
    PastEnd {
        /// Cell of the index list holding the value.
        cell: usize,
        /// The position the value decodes to.
        position: usize,
        /// Positions the indexed axis has.
        axis_size: usize,
    },
}

impl IndexUnit {
    /// Decodes an index value into an indexed-axis position.
    pub fn decode(self, raw: i32, cell: usize) -> Result<usize, IndexValueError> {
        let offset = usize::try_from(raw).map_err(|_| IndexValueError::Negative { cell, raw })?;
        let Self::Bytes(stride) = self else { return Ok(offset) };
        if offset % stride.get() != 0 {
            return Err(IndexValueError::OffBoundary {
                cell,
                raw,
                stride: stride.get(),
            });
        }
        Ok(offset / stride.get())
    }
}
