use furiosa_mapping::M;
use furiosa_opt_rt::Buffer;

use super::Function;
use crate::Error;
use crate::runtime::{Buffers, DeviceSend};
use crate::scalar::Scalar;
use crate::tensor::memory::HbmTensor;

impl<D: Scalar, Chip: M, Element: M> From<Buffer> for HbmTensor<D, Chip, Element> {
    fn from(buffer: Buffer) -> Self {
        HbmTensor::unbacked().placed(buffer)
    }
}

/// Reconstructs a device-function return value from output buffers in declaration order.
pub trait DeviceOutput: DeviceSend + Sized {
    /// Allocates this value's output buffers on `function`, pushing them onto `buffers`.
    fn alloc_into(function: &Function, buffers: &mut Buffers) -> Result<(), Error>;
    /// Rebuilds the value by taking buffers in output order.
    fn take(buffers: &mut impl Iterator<Item = Buffer>) -> Self;
}

impl DeviceOutput for () {
    fn alloc_into(_: &Function, _: &mut Buffers) -> Result<(), Error> {
        Ok(())
    }

    fn take(_: &mut impl Iterator<Item = Buffer>) {}
}

impl<D: Scalar, Chip: M, Element: M> DeviceOutput for HbmTensor<D, Chip, Element> {
    fn alloc_into(function: &Function, buffers: &mut Buffers) -> Result<(), Error> {
        buffers.push(function.alloc(Function::device_bytes::<Chip>(Self::size()))?);
        Ok(())
    }

    fn take(buffers: &mut impl Iterator<Item = Buffer>) -> Self {
        buffers.next().expect("HbmTensor output is missing its buffer").into()
    }
}

macro_rules! impl_device_output_tuple {
    () => {};
    ($T0:ident $(, $T:ident)*) => {
        impl<$T0: DeviceOutput $(, $T: DeviceOutput)*> DeviceOutput for ($T0, $($T,)*) {
            fn alloc_into(function: &Function, buffers: &mut Buffers) -> Result<(), Error> {
                $T0::alloc_into(function, buffers)?;
                $( $T::alloc_into(function, buffers)?; )*
                Ok(())
            }

            fn take(buffers: &mut impl Iterator<Item = Buffer>) -> Self {
                ($T0::take(buffers), $($T::take(buffers),)*)
            }
        }

        impl_device_output_tuple!($($T),*);
    };
}

impl_device_output_tuple!(
    T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23
);
