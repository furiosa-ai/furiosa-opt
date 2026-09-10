use furiosa_mapping::M;
use furiosa_opt_rt::Buffer;

use crate::Error;
use crate::backend::Backend;
use crate::runtime::{Buffers, DeviceSend};
use crate::scalar::Scalar;
use crate::tensor::memory::{HbmTensor, HbmTensorView, HbmTensorViewMut};

/// Cpu and compiler-placed tensors have no device allocation, so a host launch cannot name them.
fn placed(buffer: Option<Buffer>, buffers: &mut Buffers) -> Result<(), Error> {
    buffers.push(buffer.ok_or(Error::Unplaced)?);
    Ok(())
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend for HbmTensor<D, Chip, Element, B> {
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
        placed(self.buffer().cloned(), buffers)
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend for HbmTensorView<'_, D, Chip, Element, B> {
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
        placed(self.buffer(), buffers)
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend for HbmTensorViewMut<'_, D, Chip, Element, B> {
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
        placed(self.buffer(), buffers)
    }
}

impl DeviceSend for () {
    fn bind(&self, _: &mut Buffers) -> Result<(), Error> {
        Ok(())
    }
}

impl<T> DeviceSend for std::marker::PhantomData<T> {
    fn bind(&self, _: &mut Buffers) -> Result<(), Error> {
        Ok(())
    }
}

impl<T: DeviceSend + ?Sized> DeviceSend for &T {
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
        T::bind(self, buffers)
    }
}

impl<T: DeviceSend + ?Sized> DeviceSend for &mut T {
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
        T::bind(self, buffers)
    }
}

impl<T: DeviceSend, const N: usize> DeviceSend for [T; N] {
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
        self.iter().try_for_each(|value| value.bind(buffers))
    }
}

macro_rules! impl_device_send_tuple {
    ($($ty:ident : $value:ident),+ $(,)?) => {
        impl<$($ty: DeviceSend),+> DeviceSend for ($($ty,)+) {
            fn bind(&self, buffers: &mut Buffers) -> Result<(), Error> {
                let ($($value,)+) = self;
                $($value.bind(buffers)?;)+
                Ok(())
            }
        }
    };
}

impl_device_send_tuple!(T0: t0);
impl_device_send_tuple!(T0: t0, T1: t1);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21, T22: t22);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21, T22: t22, T23: t23);
