use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::tensor::memory::{Address, HbmTensor, HostTensor};
use crate::tensor::raw::BufRawTensor;

use crate::runtime::backend::Backend;

/// Emulation backend: host-side buffer emulator using `BufRawTensor` storage.
///
/// Holds a native `Vec<D>` host staging buffer per tensor, the same storage shape used by `Npu`,
/// but without an actual device behind it. Unlike `Simulation` (which interprets every operation
/// inline via mapping algebra on `MathRawTensor`), Emulation is the future "Cpu+Buffer" backend
/// where buffer-level semantics will eventually be implemented. Today the value-producing
/// per-operation methods on `BufRawTensor` are all `todo!()` placeholders.
#[derive(Debug, Clone, Copy)]
pub struct Emulation;

impl Backend for Emulation {
    type RawTensor<D: Scalar> = BufRawTensor<D>;

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        _host: &HostTensor<D, Element, Self>,
        _address: Address,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        todo!("Emulation::to_hbm: buffer semantics not implemented yet")
    }

    async fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        _hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        todo!("Emulation::from_hbm: buffer semantics not implemented yet")
    }
}
