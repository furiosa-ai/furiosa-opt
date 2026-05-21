use std::marker::PhantomData;

use furiosa_mapping::*;

use super::Tensor;
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::raw::RawTensor;

/// Mutable view into a tensor.
pub struct TensorViewMut<'l, D: Scalar, Mapping: M, B: Backend = CurrentBackend> {
    inner: &'l mut B::RawTensor<D>,
    offset: Index,
    _marker: PhantomData<(Mapping, B)>,
}

impl<'l, D: Scalar, Mapping: M, B: Backend> std::fmt::Debug for TensorViewMut<'l, D, Mapping, B>
where
    B::RawTensor<D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorViewMut")
            .field("inner", &self.inner)
            .field("offset", &self.offset)
            .finish()
    }
}

/// Immutable view into a tensor.
pub struct TensorView<'l, D: Scalar, Mapping: M, B: Backend = CurrentBackend> {
    inner: &'l B::RawTensor<D>,
    offset: Index,
    _marker: PhantomData<(Mapping, B)>,
}

impl<'l, D: Scalar, Mapping: M, B: Backend> std::fmt::Debug for TensorView<'l, D, Mapping, B>
where
    B::RawTensor<D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("inner", &self.inner)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'l, D: Scalar, Mapping: M, B: Backend> Clone for TensorView<'l, D, Mapping, B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            offset: self.offset.clone(),
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, Mapping: M, B: Backend> From<TensorViewMut<'l, D, Mapping, B>> for TensorView<'l, D, Mapping, B> {
    fn from(view: TensorViewMut<'l, D, Mapping, B>) -> Self {
        Self {
            inner: view.inner,
            offset: view.offset,
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorViewMut<'l, D, E, B> {
    /// Creates a new tensor view mut.
    pub(crate) fn new(inner: &'l mut B::RawTensor<D>) -> Self {
        Self {
            inner,
            offset: Index::new(),
            _marker: PhantomData,
        }
    }

    /// Splits the tensor view by tiling.
    pub fn tile<I: M, E2: M, const LEN: usize>(&mut self, start: usize) -> TensorViewMut<'l, D, E2, B> {
        assert_div::<I, E, E2, LEN>();
        let mut offset = self.offset.clone();
        offset.add_mapping::<I>(start);
        TensorViewMut {
            inner: unsafe { &mut *(self.inner as *mut _) }, // TODO: violating Rust aliasing rules...
            offset,
            _marker: PhantomData,
        }
    }

    /// Transposes from a tensor. Delegates to [`RawTensor::write_transpose`].
    pub fn write_transpose<'lsrc, Src: M>(&mut self, src: TensorView<'lsrc, D, Src, B>, allow_broadcast: bool) {
        self.inner
            .write_transpose::<Src, E>(src.inner, &src.offset, &self.offset, allow_broadcast);
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Splits the tensor view by tiling.
    pub fn tile<I: M, E2: M, const LEN: usize>(&self, start: usize) -> TensorView<'l, D, E2, B> {
        assert_div::<I, E, E2, LEN>();
        let mut offset = self.offset.clone();
        offset.add_mapping::<I>(start);
        TensorView {
            inner: self.inner,
            offset,
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Reads the tensor view into a new tensor. Delegates to [`RawTensor::write_transpose`],
    /// whose body runs `transpose_broadcast` on every storage (so Typecheck still validates).
    pub fn read(self) -> Tensor<D, E, B> {
        let mut result = Tensor::uninit();
        result.view_mut().write_transpose(self, false);
        result
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Creates a new tensor view.
    pub(crate) fn new(inner: &'l B::RawTensor<D>) -> Self {
        Self {
            inner,
            offset: Index::new(),
            _marker: PhantomData,
        }
    }
}
