use furiosa_mapping::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::Error;
use crate::scalar::{MaterializableScalar, Scalar};
use crate::storage::BufStorage;
use crate::tensor::memory::{HbmTensor, HostTensor};

use super::{Function, HostBuf};
use crate::backend::Backend;
use crate::cast::{ContractionAccumulator, ContractionCast};
use crate::context::{Dma, DmaContext};
use crate::runtime::Topology;
use furiosa_opt_rt::Image;

/// NPU backend.
///
/// Host storage uses packed CPU memory; an HBM tensor's bytes live on the device and its host
/// side holds none.
#[derive(Debug, Clone, Copy)]
pub struct Npu;

/// An opened device with the device functions loaded on it so far. A `#[device]` fn loads once per
/// device and launches through the same handle afterwards.
pub struct Device {
    device: Arc<furiosa_opt_rt::Device>,
    functions: tokio::sync::Mutex<HashMap<(&'static str, Vec<usize>), Arc<furiosa_opt_rt::Function>>>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Device").finish_non_exhaustive()
    }
}

impl Device {
    pub(crate) fn inner(&self) -> &Arc<furiosa_opt_rt::Device> {
        &self.device
    }

    /// The device function for registry entry `path`/`key`, loaded from `bytes` on first use.
    pub(crate) async fn function(
        &self,
        path: &'static str,
        key: &[usize],
        bytes: &[u8],
    ) -> Result<Arc<furiosa_opt_rt::Function>, Error> {
        let mut functions = self.functions.lock().await;
        if let Some(function) = functions.get(&(path, key.to_vec())) {
            return Ok(Arc::clone(function));
        }
        let image = Image::parse(bytes).map_err(furiosa_opt_rt::FunctionError::Image)?;
        let function = Arc::new(furiosa_opt_rt::Function::load(&self.device, &image).await?);
        functions.insert((path, key.to_vec()), Arc::clone(&function));
        Ok(function)
    }
}

impl Backend for Npu {
    type Device = Arc<Device>;
    type Storage<D: Scalar> = BufStorage<D, HostBuf>;

    /// Opens a device of `chips` chips among those `FURIOSA_VISIBLE_CHIPS` names, or any when
    /// it is unset. The runtime itself reads no environment; this is where the process's policy
    /// meets it.
    fn open(Topology { chips, pes }: Topology) -> Result<Self::Device, Error> {
        let visible = match std::env::var("FURIOSA_VISIBLE_CHIPS") {
            Ok(spec) if !spec.trim().is_empty() => Some(
                spec.split(',')
                    .map(|id| {
                        id.trim().parse::<u8>().map_err(|_| {
                            Error::Chips(format!("FURIOSA_VISIBLE_CHIPS: `{id}` is not a chip index (0-255)"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Ok(_) | Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::Chips("FURIOSA_VISIBLE_CHIPS is not valid Unicode".into()));
            }
        };
        let mut builder = furiosa_opt_rt::Device::builder((chips, pes));
        if let Some(visible) = visible {
            builder = builder.among(visible);
        }
        Ok(Arc::new(Device {
            device: Arc::new(builder.open()?),
            functions: tokio::sync::Mutex::new(HashMap::new()),
        }))
    }

    fn from_vec<D: Scalar>(_mapping: &Mapping, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        BufStorage::from_vec(data)
    }

    fn from_buf<D: MaterializableScalar>(_mapping: &Mapping, buf: Vec<u8>) -> Self::Storage<D> {
        BufStorage::from_buf(buf)
    }

    /// An unplaced handle; the compiled function places its own buffers, so this never names one.
    fn alloc_hbm<D: Scalar, Chip: M, Element: M>() -> HbmTensor<D, Chip, Element, Self> {
        HbmTensor::unbacked()
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
        dma: &DmaContext<{ Dma::Pcie }, Self>,
    ) -> Result<HbmTensor<D, Chip, Element2, Self>, Error> {
        Function::write(dma, host).await
    }

    async fn from_hbm<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
    ) -> Result<HostTensor<D, Element2, Self>, Error> {
        Function::read(dma, hbm).await
    }

    async fn to_hbm_into<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        hbm: &mut HbmTensor<D, Chip, Element2, Self>,
    ) -> Result<(), Error> {
        Function::write_into(dma, host, hbm).await
    }

    async fn from_hbm_into<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        host: &mut HostTensor<D, Element2, Self>,
    ) -> Result<(), Error> {
        Function::read_into(dma, hbm, host).await
    }

    fn pin<D: MaterializableScalar>(storage: Self::Storage<D>) -> Result<Self::Storage<D>, Error> {
        storage.into_inner().pinned().map(BufStorage::from_inner)
    }
}
