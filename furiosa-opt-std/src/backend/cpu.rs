use furiosa_mapping::*;
use std::sync::Arc;

use crate::Error;
use crate::scalar::{MaterializableScalar, Scalar};
use crate::storage::BufStorage;
use crate::tensor::Tensor;
use crate::tensor::memory::{HbmTensor, HostTensor};

use crate::backend::Backend;
use crate::cast::{ContractionAccumulator, ContractionCast};
use crate::context::{Dma, DmaContext};
use crate::runtime::Topology;

/// Cpu backend: host-side buffer interpreter using `BufStorage` storage.
///
/// Runs every operation over the physical staging buffer through `BufStorage`'s inherent
/// methods, with no device behind it, over the same `Vec<D>` staging buffer shape `Npu` uses.
/// DMA is the shared host-side `transpose` passthrough.
#[derive(Debug, Clone, Copy)]
pub struct Cpu;

impl Backend for Cpu {
    /// The host has no device to hold; a shared unit keeps the runtime's cache uniform across
    /// backends.
    type Device = Arc<()>;

    fn open(_: Topology) -> Result<Self::Device, Error> {
        Ok(Arc::new(()))
    }

    type Storage<D: Scalar> = BufStorage<D, Vec<u8>>;

    fn from_vec<D: Scalar>(_mapping: &Mapping, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        BufStorage::from_vec(data)
    }

    fn from_buf<D: MaterializableScalar>(_mapping: &Mapping, buf: Vec<u8>) -> Self::Storage<D> {
        BufStorage::from_buf(buf)
    }

    /// No device behind this backend, so the tensor is just its bytes; the address stays the
    /// placeholder every host-side handle carries.
    fn alloc_hbm<D: Scalar, Chip: M, Element: M>() -> HbmTensor<D, Chip, Element, Self> {
        HbmTensor::from_parts(Tensor::zeroed())
    }

    fn zeroed<D: Scalar>(mapping: &Mapping) -> Self::Storage<D> {
        BufStorage::zeroed(mapping.size())
    }

    fn into_vec<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &Mapping) -> Vec<D> {
        storage.into_vec(mapping)
    }

    fn into_buf<D: Scalar>(storage: Self::Storage<D>, mapping: &Mapping) -> Vec<u8> {
        storage.into_buf(mapping)
    }

    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl Fn(D) -> D2 + Sync) -> Self::Storage<D2> {
        src.map(f)
    }

    fn zip_with<D: MaterializableScalar, D2: MaterializableScalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> Self::Storage<D3> {
        a.zip_with(b, f)
    }

    fn zip3_with<D: MaterializableScalar, D2: MaterializableScalar, D3: MaterializableScalar, D4: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        c: &Self::Storage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> Self::Storage<D4> {
        a.zip3_with(b, c, f)
    }

    fn transpose<D: Scalar, Src: M, Dst: M>(
        dst: &mut Self::Storage<D>,
        src: &Self::Storage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &Mapping,
        dst_map: &Mapping,
        allow_broadcast: bool,
    ) {
        dst.transpose::<Src, Dst>(src, src_offset, dst_offset, src_map, dst_map, allow_broadcast);
    }

    fn reduce<D: MaterializableScalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D> {
        src.reduce::<Src, Dst, _>(reduce_fn, identity, allow_broadcast)
    }

    fn contraction<D: ContractionCast + MaterializableScalar>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &Mapping,
        rhs_map: &Mapping,
        pre_reduce: &Mapping,
        out: &Mapping,
    ) -> Self::Storage<D> {
        // `BufStorage` is a bare buffer with no layout of its own, so it reads operand strides from
        // `lhs_map`/`rhs_map` (the same reason its `transpose` above takes `src_map`/`dst_map`).
        BufStorage::contraction(lhs, rhs, lhs_map, rhs_map, pre_reduce, out)
    }

    fn contraction_prewidened<D: ContractionAccumulator>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &Mapping,
        rhs_map: &Mapping,
        pre_reduce: &Mapping,
        out: &Mapping,
    ) -> Self::Storage<D> {
        // `BufStorage` is a bare buffer with no layout of its own, so it reads operand strides from
        // `lhs_map`/`rhs_map` (the same reason its `transpose` above takes `src_map`/`dst_map`).
        BufStorage::contraction_prewidened(lhs, rhs, lhs_map, rhs_map, pre_reduce, out)
    }

    fn scatter<D: Scalar, Src: M, Key: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    ) {
        src.scatter::<Src, Key, Dst, Idx>(dst, index, scaled);
    }

    fn gather<D: MaterializableScalar, Src: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    ) {
        src.gather::<Src, Dst, Idx>(dst, index, scaled);
    }

    fn reshape<D: Scalar, Src: M, Dst: M>(src: &Self::Storage<D>) -> Self::Storage<D> {
        src.reshape::<Src, Dst>()
    }

    fn transmute<D: MaterializableScalar, Src: M, Dst: M>(
        storage: Self::Storage<D>,
        src_map: &Mapping,
        dst_map: &Mapping,
    ) -> Self::Storage<D> {
        storage.transmute(src_map, dst_map)
    }

    async fn to_hbm<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        _dma: &DmaContext<{ Dma::Pcie }, Self>,
    ) -> Result<HbmTensor<D, Chip, Element2, Self>, Error> {
        Ok(HbmTensor::from_parts(host.inner().transpose(true)))
    }

    async fn from_hbm<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
        _dma: &DmaContext<{ Dma::Pcie }, Self>,
    ) -> Result<HostTensor<D, Element2, Self>, Error> {
        Ok(hbm.inner().transpose(true).into())
    }

    async fn to_hbm_into<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        hbm: &mut HbmTensor<D, Chip, Element2, Self>,
    ) -> Result<(), Error> {
        *hbm = Self::to_hbm(host, dma).await?;
        Ok(())
    }

    async fn from_hbm_into<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        host: &mut HostTensor<D, Element2, Self>,
    ) -> Result<(), Error> {
        *host = Self::from_hbm(hbm, dma).await?;
        Ok(())
    }

    fn pin<D: MaterializableScalar>(storage: Self::Storage<D>) -> Result<Self::Storage<D>, Error> {
        Ok(storage)
    }
}
