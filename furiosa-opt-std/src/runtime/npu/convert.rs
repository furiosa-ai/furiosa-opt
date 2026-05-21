use furiosa_mapping::{M, Pair};

use super::Buffer;
use super::ffi::{lib, rt};
use crate::scalar::Scalar;
use crate::tensor::memory::{HbmTensor, HbmTensorView, HbmTensorViewMut};

fn to_buffer<D: Scalar, Chip: M, Element: M>(addr: u64) -> Buffer {
    let len = Pair::<Chip, Element>::SIZE * std::mem::size_of::<D>();
    Buffer::from_raw(unsafe { lib().furiosa_buffer_from_npu(rt(), addr, len) })
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
        unsafe { Self::from_addr(lib().furiosa_buffer_addr(buf.as_ptr())) }
    }
}
