use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::cast::ContractionCast;
use crate::scalar::{MaterializableScalar, Scalar};
use crate::storage::BufStorage;
use crate::tensor::memory::{HbmTensor, HostTensor};

use crate::backend::Backend;

/// Emulation backend: host-side buffer interpreter using `BufStorage` storage.
///
/// Runs every operation over the physical staging buffer through `BufStorage`'s inherent
/// methods, with no device behind it, over the same `Vec<D>` staging buffer shape `Npu` uses.
/// DMA is the shared host-side `transpose` passthrough.
#[derive(Debug, Clone, Copy)]
pub struct Emulation;

impl Backend for Emulation {
    type Storage<D: Scalar> = BufStorage<D, Vec<u8>>;

    fn from_vec<D: Scalar>(_mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        BufStorage::from_vec(data)
    }

    fn from_buf<D: Scalar>(_mapping: &MappingValue, buf: Vec<u8>) -> Self::Storage<D> {
        BufStorage::from_buf(buf)
    }

    fn zeroed<D: Scalar>(mapping: &MappingValue) -> Self::Storage<D> {
        BufStorage::zeroed(mapping.size())
    }

    fn into_vec<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<D> {
        storage.into_vec(mapping)
    }

    fn into_buf<D: Scalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<u8> {
        storage.into_buf(mapping)
    }

    fn map<D: MaterializableScalar, D2: Scalar>(
        src: &Self::Storage<D>,
        f: impl Fn(D) -> D2 + Sync,
    ) -> Self::Storage<D2> {
        src.map(f)
    }

    fn map_bounded<D: Scalar, D2: Scalar>(
        src: &Self::Storage<D>,
        len: usize,
        f: impl Fn(D) -> D2 + Sync,
    ) -> Self::Storage<D2> {
        src.map_bounded(len, f)
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
        src_map: &MappingValue,
        dst_map: &MappingValue,
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
        lhs_map: &MappingValue,
        rhs_map: &MappingValue,
        pre_reduce: &MappingValue,
        out: &MappingValue,
    ) -> Self::Storage<D> {
        // `BufStorage` is a bare buffer with no layout of its own, so it reads operand strides from
        // `lhs_map`/`rhs_map` (the same reason its `transpose` above takes `src_map`/`dst_map`).
        BufStorage::contraction(lhs, rhs, lhs_map, rhs_map, pre_reduce, out)
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
        src_map: &MappingValue,
        dst_map: &MappingValue,
    ) -> Self::Storage<D> {
        storage.transmute(src_map, dst_map)
    }

    async fn to_hbm<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        HbmTensor::new(host.inner().transpose(true), 0)
    }

    async fn from_hbm<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        hbm.inner().transpose(true).into()
    }
}
