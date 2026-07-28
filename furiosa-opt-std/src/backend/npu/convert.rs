use furiosa_mapping::{M, Pair};

use super::ffi::{furiosa_npu_buffer_from, furiosa_npu_buffer_offset, rt};
use super::{Buffer, Kernel};
use crate::scalar::Scalar;
use crate::tensor::memory::{HbmTensor, HbmTensorView, HbmTensorViewMut};

fn to_buffer<D: Scalar, Chip: M, Element: M>(addr: u64) -> Buffer {
    let len = Pair::<Chip, Element>::SIZE * std::mem::size_of::<D>();
    Buffer::from_raw(unsafe { furiosa_npu_buffer_from(rt(), addr, len) })
}

impl<D: Scalar, Chip: M, Element: M> From<&HbmTensor<D, Chip, Element>> for Buffer {
    fn from(tensor: &HbmTensor<D, Chip, Element>) -> Buffer {
        to_buffer::<D, Chip, Element>(tensor.address())
    }
}

impl<D: Scalar, Chip: M, Element: M> From<&HbmTensorView<'_, D, Chip, Element>> for Buffer {
    fn from(view: &HbmTensorView<'_, D, Chip, Element>) -> Buffer {
        to_buffer::<D, Chip, Element>(view.address())
    }
}

impl<D: Scalar, Chip: M, Element: M> From<&HbmTensorViewMut<'_, D, Chip, Element>> for Buffer {
    fn from(view: &HbmTensorViewMut<'_, D, Chip, Element>) -> Buffer {
        to_buffer::<D, Chip, Element>(view.address())
    }
}

impl<D: Scalar, Chip: M, Element: M> From<Buffer> for HbmTensor<D, Chip, Element> {
    fn from(buf: Buffer) -> Self {
        let addr = unsafe { furiosa_npu_buffer_offset(buf.as_ptr()) };
        unsafe { Self::from_addr(addr) }.owns(buf)
    }
}

/// Reconstructs a kernel's return value from its device output buffers, the output-side mirror of
/// [`ExtendBuffers`]: `alloc_outputs` allocates one buffer per output, then (after the kernel fills
/// them) `from_buffers` rebuilds the value in the same order. Implemented for a single [`HbmTensor`]
/// and for tuples (output order = tuple order).
pub trait KernelOutput: Sized {
    /// Number of leaf output buffers this value owns.
    fn output_count() -> usize;
    /// Allocates this value's output buffers on `kernel`, appending them to `out` in output order.
    fn alloc_outputs_into(kernel: &Kernel, out: &mut Vec<Buffer>);
    /// Rebuilds the value by taking its buffers off the front of `buffers`, in output order.
    fn take_from(buffers: &mut impl Iterator<Item = Buffer>) -> Self;

    /// Allocates all output buffers into one `Vec` (the slice `Kernel::run` writes into).
    fn alloc_outputs(kernel: &Kernel) -> Vec<Buffer> {
        let mut out = Vec::with_capacity(Self::output_count());
        Self::alloc_outputs_into(kernel, &mut out);
        out
    }
    /// Rebuilds the value from exactly its filled output buffers, erroring on a miscount.
    fn from_buffers(buffers: Vec<Buffer>) -> Self {
        assert_eq!(
            buffers.len(),
            Self::output_count(),
            "expected {} output buffers, got {}",
            Self::output_count(),
            buffers.len(),
        );
        Self::take_from(&mut buffers.into_iter())
    }
}

impl<D: Scalar, Chip: M, Element: M> KernelOutput for HbmTensor<D, Chip, Element> {
    fn output_count() -> usize {
        1
    }
    fn alloc_outputs_into(kernel: &Kernel, out: &mut Vec<Buffer>) {
        out.push(kernel.alloc(Self::size()));
    }
    fn take_from(buffers: &mut impl Iterator<Item = Buffer>) -> Self {
        buffers.next().expect("HbmTensor output is missing its buffer").into()
    }
}

macro_rules! impl_kernel_output_tuple {
    () => {};
    ($T0:ident $(, $T:ident)*) => {
        impl<$T0: KernelOutput $(, $T: KernelOutput)*> KernelOutput for ($T0, $($T,)*) {
            fn output_count() -> usize {
                $T0::output_count() $(+ $T::output_count())*
            }
            fn alloc_outputs_into(kernel: &Kernel, out: &mut Vec<Buffer>) {
                $T0::alloc_outputs_into(kernel, out);
                $( $T::alloc_outputs_into(kernel, out); )*
            }
            fn take_from(buffers: &mut impl Iterator<Item = Buffer>) -> Self {
                // Tuple fields evaluate left to right, so each output takes its buffers in order.
                ($T0::take_from(buffers), $($T::take_from(buffers),)*)
            }
        }
        impl_kernel_output_tuple!($($T),*);
    };
}

impl_kernel_output_tuple!(
    T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23
);

/// `Extend`-shaped trait for `Vec<Buffer>`, narrowed to this crate's needs.
///
/// Has the identical signature to `core::iter::Extend<A>`, same `&mut self`
/// receiver, same `IntoIterator<Item = A>` parameter, same `extend` method
/// name.
pub trait ExtendBuffers<A> {
    /// Extends `self` with one DMA buffer per leaf tensor produced by `iter`.
    fn extend<I: IntoIterator<Item = A>>(&mut self, iter: I);
}

impl ExtendBuffers<()> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = ()>>(&mut self, _iter: I) {}
}

impl<D: Scalar, Chip: M, E: M> ExtendBuffers<HbmTensor<D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = HbmTensor<D, Chip, E>>>(&mut self, iter: I) {
        for t in iter {
            self.push((&t).into());
        }
    }
}
impl<'a, D: Scalar, Chip: M, E: M> ExtendBuffers<&'a HbmTensor<D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = &'a HbmTensor<D, Chip, E>>>(&mut self, iter: I) {
        for t in iter {
            self.push(t.into());
        }
    }
}
impl<'a, D: Scalar, Chip: M, E: M> ExtendBuffers<&'a mut HbmTensor<D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = &'a mut HbmTensor<D, Chip, E>>>(&mut self, iter: I) {
        for t in iter {
            self.push((&*t).into());
        }
    }
}

impl<'a, D: Scalar, Chip: M, E: M> ExtendBuffers<HbmTensorView<'a, D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = HbmTensorView<'a, D, Chip, E>>>(&mut self, iter: I) {
        for v in iter {
            self.push((&v).into());
        }
    }
}
impl<'a, 'b, D: Scalar, Chip: M, E: M> ExtendBuffers<&'a HbmTensorView<'b, D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = &'a HbmTensorView<'b, D, Chip, E>>>(&mut self, iter: I) {
        for v in iter {
            self.push(v.into());
        }
    }
}
impl<'a, 'b, D: Scalar, Chip: M, E: M> ExtendBuffers<&'a mut HbmTensorView<'b, D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = &'a mut HbmTensorView<'b, D, Chip, E>>>(&mut self, iter: I) {
        for v in iter {
            self.push((&*v).into());
        }
    }
}

impl<'a, D: Scalar, Chip: M, E: M> ExtendBuffers<HbmTensorViewMut<'a, D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = HbmTensorViewMut<'a, D, Chip, E>>>(&mut self, iter: I) {
        for v in iter {
            self.push((&v).into());
        }
    }
}
impl<'a, 'b, D: Scalar, Chip: M, E: M> ExtendBuffers<&'a HbmTensorViewMut<'b, D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = &'a HbmTensorViewMut<'b, D, Chip, E>>>(&mut self, iter: I) {
        for v in iter {
            self.push(v.into());
        }
    }
}
impl<'a, 'b, D: Scalar, Chip: M, E: M> ExtendBuffers<&'a mut HbmTensorViewMut<'b, D, Chip, E>> for Vec<Buffer> {
    fn extend<I: IntoIterator<Item = &'a mut HbmTensorViewMut<'b, D, Chip, E>>>(&mut self, iter: I) {
        for v in iter {
            self.push((&*v).into());
        }
    }
}

macro_rules! impl_extend_buffers_tuple {
    () => {};
    (($T0:ident, $t0:ident) $(, ($T:ident, $t:ident))*) => {
        impl<$T0 $(, $T)*> ExtendBuffers<($T0, $($T,)*)> for Vec<Buffer>
        where
            Self: ExtendBuffers<$T0> $(+ ExtendBuffers<$T>)*
        {
            fn extend<__I: IntoIterator<Item = ($T0, $($T,)*)>>(&mut self, iter: __I) {
                for ($t0, $($t,)*) in iter {
                    ExtendBuffers::extend(self, ::std::iter::once($t0));
                    $( ExtendBuffers::extend(self, ::std::iter::once($t)); )*
                }
            }
        }
        impl_extend_buffers_tuple!($(($T, $t)),*);
    };
}

impl_extend_buffers_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6),
    (T7, t7),
    (T8, t8),
    (T9, t9),
    (T10, t10),
    (T11, t11),
    (T12, t12),
    (T13, t13),
    (T14, t14),
    (T15, t15),
    (T16, t16),
    (T17, t17),
    (T18, t18),
    (T19, t19),
    (T20, t20),
    (T21, t21),
    (T22, t22),
    (T23, t23)
);
