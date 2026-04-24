//! Type-level encoding of tensor presence.
//!
//! The [`TensorState`] trait encodes at the type level whether a value currently
//! holds a tensor and, if so, what scalar type and memory mapping that tensor has.
//!
//! Two implementations are provided:
//! - [`NoTensor`] — no tensor is present (empty data).
//! - [`HasTensor<D, Mapping>`] — a [`Tensor<D, Mapping>`] is present.

use std::fmt::Debug;

use crate::tensor::Tensor;
use crate::vector_engine::scalar::VeScalar;
use furiosa_mapping::M;

/// Marker trait that tracks tensor presence at compile time.
///
/// The type parameter `D` ties the stored tensor's scalar type to the pipeline's current
/// scalar type, ensuring at compile time that tensor reads match the pipeline's `D`.
///
/// Implementors either hold no data ([`NoTensor`]) or store a [`Tensor<D, Mapping>`] ([`HasTensor`]).
pub trait TensorState<D: VeScalar>: Debug {
    /// Clones the tensor data, transposing to target mapping if needed.
    fn clone_tensor_as<TargetMapping: M>(&self) -> Option<Tensor<D, TargetMapping>>;
}

/// No tensor is present.
#[derive(Debug)]
pub struct NoTensor;
impl<D: VeScalar> TensorState<D> for NoTensor {
    fn clone_tensor_as<TargetMapping: M>(&self) -> Option<Tensor<D, TargetMapping>> {
        None
    }
}

/// A [`Tensor`] with scalar type `D` and memory layout `Mapping` is present.
#[derive(Debug)]
pub struct HasTensor<D: VeScalar, Mapping: M> {
    data: Tensor<D, Mapping>,
}

impl<D: VeScalar, Mapping: M> HasTensor<D, Mapping> {
    /// Wraps a tensor into a `HasTensor`.
    pub fn new(tensor: Tensor<D, Mapping>) -> Self {
        Self { data: tensor }
    }
}

impl<D: VeScalar, Mapping: M> From<Tensor<D, Mapping>> for HasTensor<D, Mapping> {
    fn from(tensor: Tensor<D, Mapping>) -> Self {
        Self::new(tensor)
    }
}

impl<D: VeScalar, Mapping: M> TensorState<D> for HasTensor<D, Mapping> {
    fn clone_tensor_as<TargetMapping: M>(&self) -> Option<Tensor<D, TargetMapping>> {
        let cloned = self.data.clone();
        Some(cloned.transpose::<TargetMapping>(true))
    }
}
