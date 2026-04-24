use furiosa_mapping::*;

use crate::memory_tensor::{Address, HbmTensor, HostTensor};
use crate::raw_tensor::*;
use crate::scalar::{Opt, Scalar};
use crate::tensor::Tensor;

use super::Kernel;

/// Backend for tensor operations.
///
/// `Cpu`: interprets operations via mapping expressions (default).
/// `Npu`: dispatches to device via PCIe DMA.
pub trait Backend {
    /// Create a tensor from a flat buffer.
    fn from_buf<D: Scalar, Mapping: M>(data: Vec<Opt<D>>) -> Tensor<D, Mapping>;
    /// Serialize a tensor to a flat buffer.
    fn to_buf<D: Scalar, Mapping: M>(tensor: &Tensor<D, Mapping>) -> Vec<Opt<D>>;
    /// Transfer host tensor to HBM.
    fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element>,
        address: Address,
    ) -> impl std::future::Future<Output = HbmTensor<D, Chip, Element2>>;
    /// Transfer HBM tensor to host.
    fn to_host<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element>,
    ) -> impl std::future::Future<Output = HostTensor<D, Element2>>;
}

/// CPU backend: mapping-expression-based interpretation.
#[derive(Debug)]
pub struct Cpu;

impl Backend for Cpu {
    fn from_buf<D: Scalar, Mapping: M>(data: Vec<Opt<D>>) -> Tensor<D, Mapping> {
        let mut inner = RawTensor::from_elem::<Mapping>(Opt::Uninit);
        let mapping = Mapping::to_value().factorize();
        for (index, math_index) in Index::new().gen_indexes(mapping).into_iter().enumerate() {
            inner.write_index(math_index, data[index]);
        }
        Tensor::from_raw(inner)
    }

    fn to_buf<D: Scalar, Mapping: M>(tensor: &Tensor<D, Mapping>) -> Vec<Opt<D>> {
        let mapping = Mapping::to_value().factorize();
        Index::new()
            .gen_indexes(mapping)
            .into_iter()
            .map(|index| tensor.raw().read_index(index))
            .collect()
    }

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2> {
        HbmTensor::new(host.inner_tensor().transpose(true), address)
    }

    async fn to_host<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element>,
    ) -> HostTensor<D, Element2> {
        hbm.inner_tensor().transpose(true).into()
    }
}

/// NPU backend: PCIe DMA to/from device.
#[derive(Debug)]
pub struct Npu;

impl Backend for Npu {
    fn from_buf<D: Scalar, Mapping: M>(data: Vec<Opt<D>>) -> Tensor<D, Mapping> {
        Tensor::from_raw(RawTensor::<D>::from_vec::<Mapping>(data))
    }

    fn to_buf<D: Scalar, Mapping: M>(tensor: &Tensor<D, Mapping>) -> Vec<Opt<D>> {
        tensor.raw().data.iter().cloned().collect()
    }

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2> {
        Kernel::write(host, address).await
    }

    async fn to_host<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element>,
    ) -> HostTensor<D, Element2> {
        Kernel::read(hbm).await
    }
}
