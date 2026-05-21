//! VeScalar trait for Vector Engine scalar types.

use crate::scalar::Scalar;

/// The kind of scalar type supported by Vector Engine.
#[derive(Debug, PartialEq, Eq)]
pub enum VeScalarKind {
    /// 32-bit floating point.
    F32,
    /// 32-bit integer.
    I32,
}

/// Marker trait for scalar types supported by Vector Engine.
/// Only i32 and f32 are supported.
///
/// This trait is sealed and cannot be implemented outside this module.
pub trait VeScalar: Scalar {
    /// The kind of this scalar type.
    const KIND: VeScalarKind;
}

impl VeScalar for i32 {
    const KIND: VeScalarKind = VeScalarKind::I32;
}

impl VeScalar for f32 {
    const KIND: VeScalarKind = VeScalarKind::F32;
}
