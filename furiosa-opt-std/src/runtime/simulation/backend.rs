use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::tensor::memory::{Address, HbmTensor, HostTensor};
use crate::tensor::raw::MathRawTensor;

use crate::runtime::backend::Backend;

/// Simulation backend: mapping-expression-based interpretation.
///
/// All per-operation behavior lives on [`MathRawTensor`]'s `RawTensor` impl. Simulation only
/// supplies the host-side DMA passthrough protocols here.
#[derive(Debug, Clone, Copy)]
pub struct Simulation;

impl Backend for Simulation {
    type RawTensor<D: Scalar> = MathRawTensor<D>;

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        HbmTensor::new(host.inner_tensor().transpose(true), address)
    }

    async fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        hbm.inner_tensor().transpose(true).into()
    }
}
