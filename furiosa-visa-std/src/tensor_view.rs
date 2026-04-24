use std::marker::PhantomData;

use furiosa_mapping::*;

use super::raw_tensor::*;
use super::scalar::*;
use super::tensor::*;

/// Mutable view into a tensor.
#[derive(Debug)]
pub struct TensorViewMut<'l, D: Scalar, Mapping: M> {
    inner: &'l mut RawTensor<D>,
    offset: Index,
    _marker: PhantomData<Mapping>,
}

/// Immutable view into a tensor.
#[derive(Debug, Clone)]
pub struct TensorView<'l, D: Scalar, Mapping: M> {
    inner: &'l RawTensor<D>,
    offset: Index,
    _marker: PhantomData<Mapping>,
}

impl<'l, D: Scalar, Mapping: M> From<TensorViewMut<'l, D, Mapping>> for TensorView<'l, D, Mapping> {
    fn from(view: TensorViewMut<'l, D, Mapping>) -> Self {
        Self {
            inner: view.inner,
            offset: view.offset,
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M> TensorViewMut<'l, D, E> {
    /// Creates a new tensor view mut.
    pub fn new(inner: &'l mut RawTensor<D>) -> Self {
        Self {
            inner,
            offset: Index::new(),
            _marker: PhantomData,
        }
    }

    /// Splits the tensor view by tiling.
    pub fn tile<I: M, E2: M, const LEN: usize>(&mut self, start: usize) -> TensorViewMut<'l, D, E2> {
        assert_div::<I, E, E2, LEN>();
        let mut offset = self.offset.clone();
        offset.add_mapping::<I>(start);
        TensorViewMut {
            inner: unsafe { &mut *(self.inner as *mut _) }, // TODO: violating Rust aliasing rules...
            offset,
            _marker: PhantomData,
        }
    }

    /// Transposes from a tensor.
    pub fn write_transpose<'lsrc, Src: M>(&mut self, src: TensorView<'lsrc, D, Src>, allow_broadcast: bool) {
        // Calculate the transpose mapping.
        let src_mapping = Src::to_value();
        let dst_mapping = E::to_value();
        let d = dst_mapping.divide_relaxed(&src_mapping).exact().unwrap_or_else(|e| {
            panic!(
                "[TensorViewMut::write_transpose] failed to divide mapping expressions for transpose: {:?}\n\
                     src_mapping: {:?}\n\
                     dst_mapping: {:?}\n\
                     error: {:?}",
                e, src_mapping, dst_mapping, e
            )
        });
        let broadcast = d.dividend_residue;

        // Check if broadcast requirement is satisfied.
        if !allow_broadcast {
            assert!(broadcast.is_padding());
        }

        // Perform the transpose.
        self.inner
            .write_broadcast(src.inner, src_mapping.factorize(), broadcast, &src.offset, &self.offset);
    }
}

impl<'l, D: Scalar, E: M> TensorView<'l, D, E> {
    /// Splits the tensor view by tiling.
    pub fn tile<I: M, E2: M, const LEN: usize>(&self, start: usize) -> TensorView<'l, D, E2> {
        assert_div::<I, E, E2, LEN>();
        let mut offset = self.offset.clone();
        offset.add_mapping::<I>(start);
        TensorView {
            inner: self.inner,
            offset,
            _marker: PhantomData,
        }
    }

    /// Reads the tensor view into a new tensor.
    pub fn read(self) -> Tensor<D, E> {
        let mut result = Tensor::<D, E>::uninit();
        result.view_mut().write_transpose(self, false);
        result
    }
}

impl<'l, D: Scalar, E: M> TensorView<'l, D, E> {
    /// Creates a new tensor view.
    pub fn new(inner: &'l RawTensor<D>) -> Self {
        Self {
            inner,
            offset: Index::new(),
            _marker: PhantomData,
        }
    }
}
