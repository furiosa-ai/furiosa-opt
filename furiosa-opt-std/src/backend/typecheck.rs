use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::cast::ContractionCast;
use crate::scalar::{MaterializableScalar, Scalar};
use crate::storage::PhantomStorage;
use crate::tensor::Tensor;
use crate::tensor::memory::{HbmTensor, HostTensor};

use crate::backend::Backend;

/// Typecheck backend: shape/mapping-only, no host-side computation.
///
/// All per-operation overrides live on [`PhantomStorage`]'s inherent methods. The Backend
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
    type Storage<D: Scalar> = PhantomStorage<D>;

    fn from_vec<D: Scalar>(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        PhantomStorage::from_vec(mapping, data)
    }

    fn uninit<D: Scalar>(mapping: &MappingValue) -> Self::Storage<D> {
        PhantomStorage::uninit(mapping)
    }

    fn into_vec<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<D> {
        storage.into_vec(mapping)
    }

    fn map<D: MaterializableScalar, D2: Scalar>(
        src: &Self::Storage<D>,
        f: impl Fn(D) -> D2 + Sync,
    ) -> Self::Storage<D2> {
        src.map(f)
    }

    fn map_bounded<D: Scalar, D2: Scalar>(
        src: &Self::Storage<D>,
        _len: usize,
        f: impl Fn(D) -> D2 + Sync,
    ) -> Self::Storage<D2> {
        // `PhantomStorage` carries no values (see `PhantomStorage::map`), so the explicit `len` this
        // exists to honor for a real buffer is moot here; delegate straight to the ordinary phantom map.
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
        _src_map: &MappingValue,
        _dst_map: &MappingValue,
        allow_broadcast: bool,
    ) {
        // `PhantomStorage` holds no buffer, so it ignores `*_map` and runs only the structural check.
        dst.transpose::<Src, Dst>(src, src_offset, dst_offset, allow_broadcast);
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
        _lhs_map: &MappingValue,
        _rhs_map: &MappingValue,
        _pre_reduce: &MappingValue,
        out: &MappingValue,
    ) -> Self::Storage<D> {
        // `PhantomStorage` holds no values, so it neither widens nor folds; see its doc.
        lhs.contraction(rhs, out)
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
        _src_map: &MappingValue,
        _dst_map: &MappingValue,
    ) -> Self::Storage<D> {
        // `PhantomStorage` holds no buffer; a relabel is purely type-level.
        storage
    }

    async fn to_hbm<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        _host: &HostTensor<D, Element, Self>,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        HbmTensor::new(Tensor::empty(), 0)
    }

    async fn from_hbm<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        _hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        Tensor::empty().into()
    }
}
