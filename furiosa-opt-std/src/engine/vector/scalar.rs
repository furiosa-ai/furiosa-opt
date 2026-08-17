//! VeScalar trait for Vector Engine scalar types.

use crate::scalar::MaterializableScalar;

/// Marker trait for scalar types supported by Vector Engine.
/// Only i32 and f32 are supported.
///
/// Sealed: i32 and f32 are the only VE scalars, so every impl round-trips its bits.
/// `MaterializableScalar` is a supertrait so that the VE-generic whole-tensor ops stay callable
/// under a bare `D: VeScalar` bound.
pub trait VeScalar: MaterializableScalar + sealed::Sealed {
    /// The 32 bits the Vector Engine holds for `self`.
    fn to_raw_bits(self) -> u32;

    /// The value those 32 bits stand for.
    fn from_raw_bits(bits: u32) -> Self;

    /// Ordered comparison in the scalar's own order: signed for `i32`, IEEE for `f32`. Named to avoid
    /// shadowing `PartialOrd::lt` at concrete call sites.
    fn lt_scalar(self, other: Self) -> bool;

    /// Rereads `self`'s 32 bits as `D2`, leaving them untouched: the functional model of
    /// [`vector_reinterpret`](crate::prelude::VectorTensor::vector_reinterpret). A reinterpret, not a
    /// conversion: `1.0f32.reinterpret::<i32>()` is `0x3f80_0000`, not `1`. `D2 == Self` is the identity.
    fn reinterpret<D2: VeScalar>(self) -> D2 {
        D2::from_raw_bits(self.to_raw_bits())
    }
}

mod sealed {
    /// Private supertrait of [`VeScalar`](super::VeScalar): a third VE scalar cannot be declared
    /// outside this module.
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

impl VeScalar for i32 {
    fn to_raw_bits(self) -> u32 {
        self as u32
    }

    fn from_raw_bits(bits: u32) -> Self {
        bits as i32
    }

    fn lt_scalar(self, other: Self) -> bool {
        self < other
    }
}

impl VeScalar for f32 {
    fn to_raw_bits(self) -> u32 {
        self.to_bits()
    }

    fn from_raw_bits(bits: u32) -> Self {
        Self::from_bits(bits)
    }

    fn lt_scalar(self, other: Self) -> bool {
        self < other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bits survive a reinterpret in both directions, and a reinterpret to the same scalar is the
    /// identity (the `D == D2` case `HasConversionOp for Reinterpret` admits).
    #[test]
    fn reinterpret_preserves_the_bits() {
        // ±0.0, the sign-bit boundary, the smallest normal, the smallest denormal, a quiet NaN.
        for bits in [0u32, 0x8000_0000, 0x7fff_ffff, 0x0080_0000, 1, 0x7fc0_0000] {
            let x = f32::from_raw_bits(bits);
            assert_eq!(x.reinterpret::<i32>(), bits as i32);
            assert_eq!(x.reinterpret::<i32>().reinterpret::<f32>().to_raw_bits(), bits);
            assert_eq!(x.reinterpret::<f32>().to_raw_bits(), bits);
        }
        assert_eq!(i32::MIN.reinterpret::<f32>().to_raw_bits(), 0x8000_0000);
        assert_eq!((-1i32).reinterpret::<f32>().to_raw_bits(), u32::MAX);
    }
}
