use crate::scalar::{bf16, f8e4m3, i4};
use furiosa_mapping::M;

use super::scalar::Scalar;
use furiosa_mapping_macro::primitive;

/// Trait for types that can be cast during fetch operations.
pub trait FetchCast<D: Scalar>: Into<D> + Cast<D> {}

// TODO: complete list of fetch conversions
// Int4ToInt5,
// Int4ToInt32,
// Int8ToInt9,
// Int8ToInt32,
// Int16ToInt32,
// Float8e4m3ToFloat32,
// Float8e5m2ToFloat32,
// Bfloat16ToFloat32,
// Float16ToFloat32,
// Float32ToBfloat16,
// // Renegade-S only
// Int4ToInt9,
// Int16ToInt9,
// Float8e4m3ToBfloat16,
// Float8e5m2ToBfloat16,

// Identity casts
impl<D> FetchCast<D> for D where D: Scalar {}

impl FetchCast<i32> for i8 {}
impl FetchCast<f32> for bf16 {}
impl FetchCast<f32> for f8e4m3 {}
impl FetchCast<i32> for i4 {}

/// Trait for casting between scalar types.
pub trait Cast<D: Scalar> {
    /// Casts self to target type D.
    fn cast(self) -> D;
}

impl<D: Scalar> Cast<D> for D {
    fn cast(self) -> D {
        self
    }
}

impl Cast<i32> for i8 {
    fn cast(self) -> i32 {
        self as i32
    }
}

impl Cast<i8> for i32 {
    fn cast(self) -> i8 {
        self as i8
    }
}

impl Cast<f32> for bf16 {
    fn cast(self) -> f32 {
        self.to_f32()
    }
}

impl Cast<bf16> for f32 {
    fn cast(self) -> bf16 {
        bf16::from_f32(self)
    }
}

impl Cast<f32> for f8e4m3 {
    fn cast(self) -> f32 {
        self.to_f32()
    }
}

impl Cast<f8e4m3> for f32 {
    fn cast(self) -> f8e4m3 {
        f8e4m3::from_f32(self)
    }
}

impl Cast<i32> for i4 {
    fn cast(self) -> i32 {
        self.to_i32()
    }
}

impl Cast<i4> for i32 {
    fn cast(self) -> i4 {
        i4::from_i32(self)
    }
}

/// Output type for contraction (DPE accumulates in wider type).
pub trait ContractionCast: Scalar {
    /// The wider scalar type that accumulates contraction results.
    type Output: Scalar;
}

/// Trait for stream tensors that can be cast to a different scalar type.
///
/// FIXME: This trait exists purely to disambiguate tensor `cast()` from scalar
/// `Cast::cast()` for rust-analyzer.
pub trait StreamCast<D: Scalar> {
    /// The output tensor type after casting to scalar type `D2`.
    type CastOutput<D2: Scalar, OutPacket: M>
    where
        D: Cast<D2>;

    /// Casts this tensor's scalar type from `D` to `D2`.
    ///
    /// The cast engine operates on a single 32-byte flit.
    /// The input packet must already be exactly 32 bytes (ensured by the Collect Engine).
    /// After casting, the output packet is padded back to 32 bytes.
    /// Time passes through unchanged.
    #[primitive(StreamCast::cast)]
    fn cast<D2: Scalar, OutPacket: M>(self) -> Self::CastOutput<D2, OutPacket>
    where
        D: Cast<D2>;
}

impl ContractionCast for i8 {
    type Output = i32;
}

impl ContractionCast for bf16 {
    type Output = f32;
}

impl ContractionCast for f8e4m3 {
    type Output = f32;
}

impl ContractionCast for i4 {
    type Output = i32;
}
