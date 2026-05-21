//! Layer configurations for Vector Engine pipeline.

use smart_default::SmartDefault;

/// VectorEngine FxpToFp layer configuration.
#[derive(Debug, Clone, Copy, SmartDefault)]
pub struct FxpToFp(u32);

impl FxpToFp {
    /// Creates a new FxpToFp configuration with the given integer width.
    pub fn new(int_width: u32) -> Self {
        Self(int_width)
    }

    /// Returns the integer width.
    pub fn int_width(&self) -> u32 {
        self.0
    }
}

/// VectorEngine FpToFxp layer configuration.
#[derive(Debug, Clone, Copy, SmartDefault)]
pub struct FpToFxp(u32);

impl FpToFxp {
    /// Creates a new FpToFxp configuration with the given integer width.
    pub fn new(int_width: u32) -> Self {
        Self(int_width)
    }

    /// Returns the integer width.
    pub fn int_width(&self) -> u32 {
        self.0
    }
}
