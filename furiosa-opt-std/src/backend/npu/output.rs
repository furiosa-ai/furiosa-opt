use furiosa_mapping::M;

use super::ffi::furiosa_npu_buffer_offset;
use super::{Buffer, Kernel};
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::memory::HbmTensor;

impl<D: Scalar, Chip: M, Element: M> From<Buffer> for HbmTensor<D, Chip, Element> {
    fn from(buf: Buffer) -> Self {
        let addr = unsafe { furiosa_npu_buffer_offset(buf.as_ptr()) };
        Self::from_parts(Tensor::zeroed(), Some(addr)).owns(buf)
    }
}

/// Reconstructs a kernel return value from output buffers in declaration order.
pub trait KernelOutput: Sized {
    /// Number of leaf output buffers this value owns.
    fn output_count() -> usize;
    /// Allocates this value's output buffers on `kernel`.
    fn alloc_outputs_into(kernel: &Kernel, out: &mut Vec<Buffer>);
    /// Rebuilds the value by taking buffers in output order.
    fn take_from(buffers: &mut impl Iterator<Item = Buffer>) -> Self;

    /// Allocates all output buffers.
    fn alloc_outputs(kernel: &Kernel) -> Vec<Buffer> {
        let mut out = Vec::with_capacity(Self::output_count());
        Self::alloc_outputs_into(kernel, &mut out);
        out
    }

    /// Rebuilds the value from exactly its output buffers.
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

/// Binds a logical return value to existing device buffers without allocating new HBM.
pub trait KernelOutputDestination {
    /// Number of leaf output buffers in this value.
    fn output_count() -> usize;
    /// Appends one existing buffer per output leaf in positional order.
    fn extend_buffers(&mut self, buffers: &mut Vec<Buffer>);
}

impl KernelOutputDestination for () {
    fn output_count() -> usize {
        0
    }

    fn extend_buffers(&mut self, _buffers: &mut Vec<Buffer>) {}
}

impl<D: Scalar, Chip: M, Element: M> KernelOutputDestination for HbmTensor<D, Chip, Element> {
    fn output_count() -> usize {
        1
    }

    fn extend_buffers(&mut self, buffers: &mut Vec<Buffer>) {
        buffers.push((&*self).into());
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
    (($T0:ident, $t0:ident) $(, ($T:ident, $t:ident))*) => {
        impl<$T0 $(, $T)*> KernelOutput for ($T0, $($T,)*)
        where
            $T0: KernelOutput $(, $T: KernelOutput)*,
        {
            fn output_count() -> usize {
                $T0::output_count() $(+ $T::output_count())*
            }

            fn alloc_outputs_into(kernel: &Kernel, out: &mut Vec<Buffer>) {
                $T0::alloc_outputs_into(kernel, out);
                $( $T::alloc_outputs_into(kernel, out); )*
            }

            fn take_from(buffers: &mut impl Iterator<Item = Buffer>) -> Self {
                ($T0::take_from(buffers), $($T::take_from(buffers),)*)
            }
        }

        impl<$T0 $(, $T)*> KernelOutputDestination for ($T0, $($T,)*)
        where
            $T0: KernelOutputDestination $(, $T: KernelOutputDestination)*,
        {
            fn output_count() -> usize {
                $T0::output_count() $(+ $T::output_count())*
            }

            fn extend_buffers(&mut self, buffers: &mut Vec<Buffer>) {
                let ($t0, $($t,)*) = self;
                $t0.extend_buffers(buffers);
                $( $t.extend_buffers(buffers); )*
            }
        }

        impl_kernel_output_tuple!($(($T, $t)),*);
    };
}

impl_kernel_output_tuple!(
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
