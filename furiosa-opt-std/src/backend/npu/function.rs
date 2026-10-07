use super::registry;
use super::{HostBuf, Npu};

use furiosa_mapping::M;
use std::sync::Arc;

use furiosa_opt_rt::{self as rt, Buffer, View};

use crate::Error;
use crate::context::{Dma, DmaContext};
use crate::prelude::HostTensor;
use crate::scalar::{MaterializableScalar, RuntimeScalar, Scalar};
use crate::storage::BufStorage;
use crate::tensor::Tensor;
use crate::tensor::memory::{HbmScalar, HbmTensor};

trait HbmBuffer {
    fn buffer(&self) -> Option<&Buffer>;
    fn place(&mut self, buffer: Buffer);
}

impl<D: Scalar, Chip: M, Element: M> HbmBuffer for HbmTensor<D, Chip, Element, Npu> {
    fn buffer(&self) -> Option<&Buffer> {
        HbmTensor::owner(self)
    }

    fn place(&mut self, buffer: Buffer) {
        HbmTensor::own(self, buffer);
    }
}

impl<D: RuntimeScalar> HbmBuffer for HbmScalar<D, Npu> {
    fn buffer(&self) -> Option<&Buffer> {
        HbmScalar::buffer(self)
    }

    fn place(&mut self, buffer: Buffer) {
        HbmScalar::place(self, buffer);
    }
}

/// A device function loaded on the NPU backend's device: what a `#[device]` fn launches through.
pub struct Function(Arc<furiosa_opt_rt::Function>);

impl std::fmt::Debug for Function {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Function").finish_non_exhaustive()
    }
}

impl Function {
    /// Runs the function on `inputs`/`outputs`. With `FURIOSA_OPT_PROFILE` at `info` or finer and a
    /// `tracing` subscriber on `span::npu`, every span the image names is reported with its cycles.
    /// A profile that cannot be had, for a value that is no level or a request the image refuses,
    /// is warned about and the function runs unprofiled.
    pub async fn run(&self, inputs: &[Buffer], outputs: &[Buffer]) -> Result<(), Error> {
        let profile_level = || {
            let spec = std::env::var("FURIOSA_OPT_PROFILE").ok()?;
            match spec.parse::<log::Level>() {
                Ok(level) if level >= log::Level::Info => Some(level),
                Ok(_) => None,
                Err(_) => {
                    log::warn!("FURIOSA_OPT_PROFILE is {spec:?}, not a log level; running unprofiled");
                    None
                }
            }
        };
        let profiled = profile_level()
            .filter(|_| tracing::enabled!(target: "span::npu", tracing::Level::INFO))
            .and_then(|level| match self.0.profiled(level) {
                Ok(profiled) => Some(profiled),
                Err(why) => {
                    log::warn!("profiling at {level} is refused, running unprofiled: {why}");
                    None
                }
            });
        match profiled {
            Some(profiled) => {
                let spans = profiled.launch(inputs, outputs)?.wait().await?;
                for span in spans {
                    tracing::info_span!(
                        target: "span::npu",
                        "NPU",
                        cat = "NPU",
                        name = span.name,
                        // A complete event carries its own extent, so a timeline reads device
                        // cycles, not the host clock; a cluster counts its own, so each is a row.
                        ph = "Complete",
                        ts = span.begin,
                        dur = span.end - span.begin,
                        tid = span.cluster,
                        begin_cycle = span.begin,
                        end_cycle = span.end,
                    );
                }
            }
            None => self.0.launch(inputs, outputs)?.wait().await?,
        }
        Ok(())
    }
}

impl Function {
    /// Allocates `size` bytes on every chip of the function's device.
    pub(super) fn alloc(&self, size: usize) -> Result<Buffer, Error> {
        Ok(self.0.device().alloc(rt::image::Memory::Dram, size)?)
    }

    /// Bytes one chip holds of an HBM tensor spread over `Chip` chips: the device binds per
    /// device. (`Chip` is the DSL's name for the device axis.)
    pub(super) fn device_bytes<Chip: M>(total: usize) -> usize {
        assert_ne!(Chip::SIZE, 0, "an HBM tensor must span at least one chip");
        assert!(
            total.is_multiple_of(Chip::SIZE),
            "HBM tensor bytes must divide evenly across chips"
        );
        total / Chip::SIZE
    }

    /// Writes a host tensor into a fresh device allocation.
    pub(super) async fn write<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        dma: &DmaContext<{ Dma::Pcie }, Npu>,
        host: &HostTensor<D, Element, Npu>,
    ) -> Result<HbmTensor<D, Chip, Element2, Npu>, Error> {
        let mut hbm = HbmTensor::unbacked();
        Self::write_into(dma, host, &mut hbm).await?;
        Ok(hbm)
    }

    /// Writes a host tensor into `hbm`, allocating on the device only when `hbm` is unplaced.
    pub(super) async fn write_into<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        dma: &DmaContext<{ Dma::Pcie }, Npu>,
        host: &HostTensor<D, Element, Npu>,
        hbm: &mut HbmTensor<D, Chip, Element2, Npu>,
    ) -> Result<(), Error> {
        let device = dma.device().inner();
        let bytes_per_chip = Self::device_bytes::<Chip>(D::size_in_bytes_from_length(Element::SIZE));
        let buffer = Self::buffer_for_write(device, hbm, bytes_per_chip)?;
        let bytes: &[u8] = host.storage().inner().as_ref();
        device
            .write([(bytes, Self::all_chip_view::<Chip>(device, &buffer)?)])
            .await?;
        Ok(())
    }

    /// Writes one value into the local HBM allocation of every chip in the group.
    pub(super) async fn write_scalar<D: RuntimeScalar>(
        dma: &DmaContext<{ Dma::Pcie }, Npu>,
        value: D,
    ) -> Result<HbmScalar<D, Npu>, Error> {
        let mut hbm = HbmScalar::from_value(value);
        Self::write_scalar_into(dma, value, &mut hbm).await?;
        Ok(hbm)
    }

    /// Updates a scalar allocation on every chip in the group.
    pub(super) async fn write_scalar_into<D: RuntimeScalar>(
        dma: &DmaContext<{ Dma::Pcie }, Npu>,
        value: D,
        hbm: &mut HbmScalar<D, Npu>,
    ) -> Result<(), Error> {
        let device = dma.device().inner();
        let value_bytes = HbmScalar::<D, Npu>::encode(value);
        let buffer = Self::buffer_for_write(device, hbm, value_bytes.len())?;
        let all_chip_bytes = value_bytes.repeat(device.ranks().count());
        device.write([(all_chip_bytes.as_slice(), buffer.on_all())]).await?;
        hbm.set_value(value);
        Ok(())
    }

    /// Reads a device tensor into fresh host memory.
    pub(super) async fn read<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        dma: &DmaContext<{ Dma::Pcie }, Npu>,
        hbm: &HbmTensor<D, Chip, Element, Npu>,
    ) -> Result<HostTensor<D, Element2, Npu>, Error> {
        let host = vec![0; D::size_in_bytes_from_length(Element2::SIZE)];
        let mut host: HostTensor<D, Element2, Npu> =
            Tensor::from_inner(BufStorage::<D, HostBuf>::from_inner(HostBuf::from(host))).into();
        Self::read_into(dma, hbm, &mut host).await?;
        Ok(host)
    }

    /// Reads a device tensor into `host`'s own memory; pinned memory receives the DMA directly.
    pub(super) async fn read_into<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        dma: &DmaContext<{ Dma::Pcie }, Npu>,
        hbm: &HbmTensor<D, Chip, Element, Npu>,
        host: &mut HostTensor<D, Element2, Npu>,
    ) -> Result<(), Error> {
        let device = dma.device().inner();
        let buffer = hbm.owner().ok_or(Error::Unplaced)?;
        let view = Self::all_chip_view::<Chip>(device, buffer)?;
        device.read([(view, host.storage_mut().inner_mut().as_mut())]).await?;
        Ok(())
    }
}

impl Function {
    fn buffer_for_write(
        device: &rt::Device,
        target: &mut impl HbmBuffer,
        bytes_per_chip: usize,
    ) -> Result<Buffer, Error> {
        if let Some(buffer) = target.buffer() {
            return Ok(buffer.clone());
        }
        let buffer = device.alloc(rt::image::Memory::Dram, bytes_per_chip)?;
        target.place(buffer.clone());
        Ok(buffer)
    }

    fn all_chip_view<Chip: M>(device: &rt::Device, buffer: &Buffer) -> Result<View, Error> {
        let devices = device.ranks().count();
        Self::ensure_device_extent(Chip::SIZE, devices)?;
        Ok(buffer.on_all())
    }

    fn ensure_device_extent(chip_extent: usize, devices: usize) -> Result<(), Error> {
        if chip_extent != devices {
            return Err(Error::Chips(format!(
                "HBM chip extent {chip_extent} does not match the device group size {devices}"
            )));
        }
        Ok(())
    }
}

/// The device function for registry entry `path`/`key` on `device`, loaded on first use.
pub async fn function(device: &crate::Device<Npu>, path: &'static str, key: &[usize]) -> Result<Function, Error> {
    let uncompiled = || Error::Uncompiled {
        path,
        key: key.to_vec(),
    };
    let (_, bytes) = registry::entries(path)
        .find(|(entry, ..)| entry.iter().copied().eq(key.iter().map(|&key| key as u64)))
        .ok_or_else(uncompiled)?;
    if bytes.is_empty() {
        return Err(uncompiled());
    }
    Ok(Function(device.pdma.device().function(path, key, bytes).await?))
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::Function;

    #[test]
    fn rejects_invalid_partitions() {
        assert_eq!(Function::device_bytes::<m![2]>(64), 32);
        assert!(std::panic::catch_unwind(|| Function::device_bytes::<m![0]>(64)).is_err());
        assert!(std::panic::catch_unwind(|| Function::device_bytes::<m![2]>(63)).is_err());
    }

    #[test]
    fn rejects_a_smaller_chip_extent_instead_of_replicating_it() {
        assert!(Function::ensure_device_extent(1, 4).is_err());
        assert!(Function::ensure_device_extent(4, 4).is_ok());
    }
}
