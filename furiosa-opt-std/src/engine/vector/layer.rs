//! Layer configurations for Vector Engine pipeline.

use smart_default::SmartDefault;

/// VectorEngine FxpToFp layer configuration.
#[derive(Debug, Clone, Copy, SmartDefault)]
pub struct FxpToFp(u32);

impl FxpToFp {
    /// Creates a new FxpToFp configuration with the given integer width.
    pub(crate) fn new(int_width: u32) -> Self {
        Self(int_width)
    }

    /// Returns the integer width.
    pub(crate) fn int_width(&self) -> u32 {
        self.0
    }
}

/// VectorEngine FpToFxp layer configuration.
#[derive(Debug, Clone, Copy, SmartDefault)]
pub struct FpToFxp(u32);

impl FpToFxp {
    /// Creates a new FpToFxp configuration with the given integer width.
    pub(crate) fn new(int_width: u32) -> Self {
        Self(int_width)
    }

    /// Returns the integer width.
    pub(crate) fn int_width(&self) -> u32 {
        self.0
    }
}

/// Bit-preserving reread of the 32-bit stream as the other VE scalar type, the op-side handle for
/// [`vector_reinterpret`](crate::prelude::VectorTensor::vector_reinterpret).
///
/// Unlike its two siblings above it configures no layer, because a reinterpret has no hardware stage
/// to configure. It exists so both the single-stream and the pair path read the element semantics from
/// one `HasConversionOp` impl; the direction comes from that impl's `D` / `D2`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reinterpret;
