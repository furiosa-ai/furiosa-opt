use std::marker::PhantomData;

use abi_stable::std_types::RSlice;
use furiosa_mapping::*;
use ndarray::ArrayD;
use ndarray::IxDyn;
use rand::Rng;
use rand::distr::StandardUniform;

use super::raw_tensor::*;
use super::scalar::*;
use super::tensor_view::*;

/// Tensor with scalar type `D` with axes determined by `Mapping`.
#[derive(Debug, Clone)]
pub struct Tensor<D: Scalar, Mapping: M> {
    inner: RawTensor<D>,
    _marker: PhantomData<Mapping>,
}

impl<D: Scalar, Mapping: M> Tensor<D, Mapping> {
    /// Creates a new tensor from a buffer.
    #[furiosa_mapping_macro::primitive(Tensor::from_buf)]
    pub fn from_buf(data: Vec<Opt<D>>) -> Self {
        <crate::runtime::CurrentBackend as crate::runtime::Backend>::from_buf::<D, Mapping>(data)
    }

    /// Returns the tensor data as a flat vector in the physical layout order defined by `Mapping`.
    #[furiosa_mapping_macro::primitive(Tensor::to_buf)]
    pub fn to_buf(&self) -> Vec<Opt<D>> {
        <crate::runtime::CurrentBackend as crate::runtime::Backend>::to_buf::<D, Mapping>(self)
    }

    pub(crate) fn from_raw(inner: RawTensor<D>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub(crate) fn raw(&self) -> &RawTensor<D> {
        &self.inner
    }

    /// Creates a new tensor from a function.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(&Vec<Term>, &IxDyn) -> Opt<D>,
    {
        Self {
            inner: RawTensor::<D>::from_fn::<Mapping, F>(f),
            _marker: PhantomData,
        }
    }

    /// Creates a zero tensor.
    pub fn zero() -> Self {
        Self::from_fn(|_, _| Opt::Init(D::zero()))
    }

    /// Creates a random tensor.
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Self::from_fn(|_, _| Opt::Init(rng.random::<D>()))
    }

    /// Creates an uninitialized tensor.
    pub fn uninit() -> Self {
        Self::from_fn(|_, _| Opt::Uninit)
    }

    /// Creates a mutable view of the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Mapping> {
        TensorViewMut::new(&mut self.inner)
    }

    /// Creates an immutable view of the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Mapping> {
        TensorView::new(&self.inner)
    }

    /// Returns a reference to the underlying data array.
    pub(crate) fn data(&self) -> &ndarray::ArrayD<Opt<D>> {
        &self.inner.data
    }

    /// Returns a mutable reference to the underlying data array.
    pub fn data_mut(&mut self) -> &mut ArrayD<Opt<D>> {
        &mut self.inner.data
    }

    /// Transmutes the tensor to a different mapping.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible with the new mapping.
    pub unsafe fn transmute<Mapping2: M>(self) -> Tensor<D, Mapping2> {
        Tensor {
            inner: self.inner,
            _marker: PhantomData,
        }
    }

    /// Reshapes the tensor to a different mapping.
    /// change Src into Dst, final mapping is Mapping2.
    ///
    /// # Safety
    ///
    /// TODO(jeongmin.park); document safety.
    pub unsafe fn reshape<Src: M, Dst: M, Mapping2: M>(self) -> Tensor<D, Mapping2> {
        if Src::to_value() == Dst::to_value() {
            assert_eq!(Mapping::to_value(), Mapping2::to_value());

            return Tensor {
                inner: self.inner,
                _marker: PhantomData,
            };
        }

        assert_eq!(Src::SIZE, Dst::SIZE);
        assert_eq!(
            Src::to_value().factorize().into_inner().len(),
            Dst::to_value().factorize().into_inner().len(),
            "TODO: when Src/Dst have different length, src/dst quotient have different paddings"
        );

        let src_quotient = Mapping::to_value()
            .divide_relaxed(&Src::to_value())
            .exact()
            .expect("[reshape] failed to divide by the mapping by reshape source expression")
            .dividend_residue;

        let dst_quotient = Mapping2::to_value()
            .divide_relaxed(&Dst::to_value())
            .exact()
            .expect("[reshape] failed to divide by the output mapping by reshape destination expression")
            .dividend_residue;

        assert_eq!(
            src_quotient, dst_quotient,
            "[reshape] inconsistent reshape: quotient parts do not match"
        );

        let mut output = Tensor::<D, Mapping2>::uninit();

        for idx in 0..Src::SIZE {
            let src_offset = {
                let mut offset = Index::new();
                offset.add_mapping::<Src>(idx);
                offset
            };
            let dst_offset = {
                let mut offset = Index::new();
                offset.add_mapping::<Dst>(idx);
                offset
            };

            output.inner.write_broadcast(
                &self.inner,
                src_quotient.clone(),
                FMapping::new(),
                &src_offset,
                &dst_offset,
            );
        }

        output
    }

    /// Maps the tensor with a function.
    pub fn map<D2: Scalar, F>(&self, f: F) -> Tensor<D2, Mapping>
    where
        F: Fn(&Opt<D>) -> Opt<D2>,
    {
        Tensor {
            inner: self.inner.map(f),
            _marker: PhantomData,
        }
    }

    /// Zips two tensors with a function.
    pub fn zip_with<D2: Scalar, D3: Scalar, F>(&self, other: &Tensor<D2, Mapping>, f: F) -> Tensor<D3, Mapping>
    where
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>,
    {
        Tensor {
            inner: self.inner.zip_with(&other.inner, f),
            _marker: PhantomData,
        }
    }

    /// Performs reduction (sum) over axes not present in `Dst`.
    pub fn reduce_add<Dst: M>(&self) -> Tensor<D, Dst> {
        Tensor {
            inner: self.inner.reduce_add(&gen_axes::<Dst>()),
            _marker: PhantomData,
        }
    }

    /// Reduces axes present in self but absent in Dst, using a custom reduce function.
    pub fn reduce<Dst: M>(&self, reduce_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>, identity: Opt<D>) -> Tensor<D, Dst> {
        Tensor {
            inner: self.inner.reduce(&gen_axes::<Dst>(), reduce_fn, identity),
            _marker: PhantomData,
        }
    }

    /// Performs transpose.
    pub fn transpose<Dst: M>(&self, allow_broadcast: bool) -> Tensor<D, Dst> {
        let mut dst = Tensor::<D, Dst>::uninit();
        dst.view_mut().write_transpose(self.view(), allow_broadcast);
        dst
    }

    /// Performs reduction followed by broadcasting to match destination axes.
    /// This is useful when the destination has broadcast axes that should be preserved.
    ///
    /// # Examples
    /// ```ignore
    ///
    /// let src: Tensor<f32, m![ABCD]> = Tensor::from_vec(...);  // Shape: ABCD
    /// let dst: Tensor<f32, m![ABE]> = src.reduce_then_broadcast();  // Reduces CD, broadcasts to ABE
    /// ```
    pub fn reduce_then_broadcast<Dst: M>(&self) -> Tensor<D, Dst> {
        let mut dst = Tensor::<D, Dst>::uninit();

        // Perform reduction (sum)
        let reduced = self.inner.reduce_add(&dst.inner.axes);

        // Convert axes to FMapping.
        let src_fmapping = FMapping::from_axes(RSlice::from(reduced.axes.as_slice()));
        let dst_fmapping = FMapping::from_axes(RSlice::from(dst.inner.axes.as_slice()));

        // Calculate broadcast factor: divide dst by src
        let broadcast = dst_fmapping
            .divide_relaxed(src_fmapping.clone())
            .exact()
            .expect("Failed to calculate broadcast factor")
            .dividend_residue;

        // Fill the result.
        dst.inner
            .write_broadcast(&reduced, src_fmapping, broadcast, &Index::new(), &Index::new());

        dst
    }

    /// Performs reduction with a custom function, followed by broadcasting to match destination axes.
    pub fn reduce_then_broadcast_with<Dst: M>(
        &self,
        reduce_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
        identity: Opt<D>,
    ) -> Tensor<D, Dst> {
        let mut dst = Tensor::<D, Dst>::uninit();

        // Perform reduction
        let reduced = self.inner.reduce(&dst.inner.axes, reduce_fn, identity);

        // Convert axes to FMapping.
        let src_fmapping = FMapping::from_axes(RSlice::from(reduced.axes.as_slice()));
        let dst_fmapping = FMapping::from_axes(RSlice::from(dst.inner.axes.as_slice()));

        // Calculate broadcast factor: divide dst by src
        let broadcast = dst_fmapping
            .divide_relaxed(src_fmapping.clone())
            .exact()
            .expect("Failed to calculate broadcast factor")
            .dividend_residue;

        // Fill the result.
        dst.inner
            .write_broadcast(&reduced, src_fmapping, broadcast, &Index::new(), &Index::new());

        dst
    }

    /// Scatter elements from self into dst at positions given by index tensor.
    ///
    /// When `scaled=true`, indices are byte offsets; when `false`, logical positions.
    pub fn write_scatter<Key: M, Dst: M, Idx: M>(
        &self,
        dst: &mut Tensor<D, Dst>,
        index: &Tensor<i32, Idx>,
        scaled: bool,
    ) {
        let src_fmapping = Mapping::to_value().factorize();
        let dst_fmapping = Dst::to_value().factorize();
        let key = Key::to_value().factorize();

        let index_stride = if scaled {
            let payload = src_fmapping
                .clone()
                .divide_relaxed(key.clone())
                .exact()
                .expect("Src must contain scatter key")
                .dividend_residue;
            payload.remove_padding().size() * std::mem::size_of::<D>()
        } else {
            1
        };

        let indices: Vec<usize> = index
            .to_buf()
            .into_iter()
            .map(|opt| {
                let Opt::Init(v) = opt else {
                    panic!("Scatter index must be initialized")
                };
                v as usize / index_stride
            })
            .collect();

        dst.inner
            .write_scatter(&self.inner, src_fmapping, dst_fmapping, key, &indices);
    }

    /// Performs contraction between two tensors.
    pub fn contraction<Union: M, Lhs: M, Rhs: M>(lhs: &Tensor<D, Lhs>, rhs: &Tensor<D, Rhs>) -> Self {
        {
            lhs.transpose::<Union>(true)
                .zip_with(&rhs.transpose(true), |a, b| a * b)
                .reduce_add()
        }
    }
}
