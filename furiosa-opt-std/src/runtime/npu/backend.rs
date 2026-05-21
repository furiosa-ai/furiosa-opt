use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::tensor::memory::{Address, HbmTensor, HostTensor};
use crate::tensor::raw::BufRawTensor;

use super::Kernel;
use crate::runtime::backend::Backend;

/// NPU backend.
///
/// Host-side code only prepares native `Vec<D>` staging buffers and moves them between host
/// memory and NPU HBM through `to_hbm` / `from_hbm`. Tensor math is not interpreted on the host
/// CPU for this backend — value-producing operations live on `BufRawTensor`'s `RawTensor` impl
/// as `todo!()` placeholders shared with the Emulation backend. Under Npu the device kernel
/// produces values and the host never reaches those bodies at runtime.
#[derive(Debug, Clone, Copy)]
pub struct Npu;

impl Backend for Npu {
    type RawTensor<D: Scalar> = BufRawTensor<D>;

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        Kernel::write(host, address).await
    }

    async fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        Kernel::read(hbm).await
    }
}
