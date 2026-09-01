use furiosa_mapping::M;

use super::Buffer;
use super::ffi::{furiosa_npu_buffer_from, rt};
use crate::scalar::Scalar;
use crate::tensor::memory::{HbmTensor, HbmTensorView, HbmTensorViewMut};

/// Registers `[addr, addr + len)` as a device buffer. The caller states `len` because a tensor and a
/// view of it measure it differently, and the runtime rejects a span past the allocation.
fn to_buffer(addr: Option<u64>, len: usize) -> Buffer {
    // An unplaced handle belongs inside a kernel, where the compiled program owns the placement.
    // Reaching this conversion means one was passed to `launch`, where only a real allocation (a host
    // `to_hbm` upload, or a previous launch's result) can be registered.
    let addr = addr.expect("a kernel argument must name a device allocation, not an unplaced handle");
    Buffer::from_raw(unsafe { furiosa_npu_buffer_from(rt(), addr, len) })
}

impl<D: Scalar, Chip: M, Element: M> From<&HbmTensor<D, Chip, Element>> for Buffer {
    fn from(tensor: &HbmTensor<D, Chip, Element>) -> Buffer {
        to_buffer(tensor.address(), HbmTensor::<D, Chip, Element>::size())
    }
}

impl<D: Scalar, Chip: M, Element: M> From<&HbmTensorView<'_, D, Chip, Element>> for Buffer {
    fn from(view: &HbmTensorView<'_, D, Chip, Element>) -> Buffer {
        to_buffer(view.address(), view.addressable_len())
    }
}

impl<D: Scalar, Chip: M, Element: M> From<&HbmTensorViewMut<'_, D, Chip, Element>> for Buffer {
    fn from(view: &HbmTensorViewMut<'_, D, Chip, Element>) -> Buffer {
        to_buffer(view.address(), view.addressable_len())
    }
}

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
