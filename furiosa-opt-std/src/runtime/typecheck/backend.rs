use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::memory::{Address, HbmTensor, HostTensor};
use crate::tensor::raw::PhantomRawTensor;

use crate::runtime::backend::Backend;

/// Typecheck backend: shape/mapping-only, no host-side computation.
///
/// All per-operation overrides live on [`PhantomRawTensor`]'s `RawTensor` impl. The Backend
/// supplies only the host-side DMA stubs here.
#[derive(Debug, Clone, Copy)]
pub struct Typecheck;

impl<D: Scalar, Mapping: M> Tensor<D, Mapping, Typecheck> {
    /// Phantom tensor for Typecheck: metadata only, no values.
    ///
    /// `Tensor::uninit()` is the generic constructor but reads as element-level
    /// "uninitialized"; for Typecheck the whole tensor *is* the empty value, so this name
    /// matches the actual semantics.
    pub fn empty() -> Self {
        Self::uninit()
    }
}

impl Backend for Typecheck {
    type RawTensor<D: Scalar> = PhantomRawTensor<D>;

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        _host: &HostTensor<D, Element, Self>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        HbmTensor::new(Tensor::empty(), address)
    }

    async fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        _hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        Tensor::empty().into()
    }
}
