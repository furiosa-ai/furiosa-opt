use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::buffer::{Allocator, Buffer, View};
use crate::{Error, Result};

mod dma;
pub(crate) mod engine;
use engine::{Cluster, Engine, PeRange};

/// The chips a process opened: functions load and launch through it, and it carries every
/// allocation and host transfer.
pub struct Device {
    pub(crate) engine: Arc<Engine>,
    pub(crate) allocator: Arc<Mutex<Allocator>>,
    /// Numbers each load, so a cluster can tell a new image at a reused address from the one it
    /// already holds.
    pub(crate) tokens: std::sync::atomic::AtomicU64,
    pes: u8,
}

/// A chip's position in its device, from 0 up to [`Device::chips`]: what a transfer names a
/// chip by, not the chip's id on the host. A device of chip 3 alone has rank 0 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChipRank(usize);

/// The chips a runtime opens, each contributing one cluster of `pes` PEs (1, 2 or 4) or, for 8,
/// both clusters. Which chips is [`Builder::among`]'s to narrow: the first window whose lowest PEs
/// no other process holds wins, and ascending chip order fixes the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    /// Number of chips.
    pub chips: u8,
    /// Number of PEs per chip.
    pub pes: u8,
}

impl From<(u8, u8)> for Topology {
    fn from((chips, pes): (u8, u8)) -> Self {
        Self { chips, pes }
    }
}

/// How a device is opened; see [`Device::builder`].
pub struct Builder {
    topology: Topology,
    timeout: Duration,
    among: Option<Vec<u8>>,
}

/// How long a cluster lets one launch run before failing it, unless the builder says otherwise. The
/// driver fails an unanswered doorbell at 60 s, so a timeout stays well under that.
const TIMEOUT: Duration = Duration::from_secs(5);

impl Device {
    /// Opens the chips `topology` names with the default options; see [`Device::builder`].
    pub fn open(topology: impl Into<Topology>) -> Result<Self> {
        Self::builder(topology).open()
    }

    /// Prepares to open the chips `topology` names, bringing up the firmware image on every
    /// cluster once [`Builder::open`] runs.
    pub fn builder(topology: impl Into<Topology>) -> Builder {
        Builder {
            topology: topology.into(),
            timeout: TIMEOUT,
            among: None,
        }
    }
}

impl Builder {
    /// How long a cluster lets one launch run before failing it: 5 s unless set, in whole seconds
    /// rounded up. [`Builder::open`] refuses zero and a limit past the driver's own deadline.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The chips a device may be chosen from; every exposed chip unless set.
    /// Whoever decides that policy, an environment variable or a scheduler, decides it out here.
    pub fn among(mut self, chips: impl IntoIterator<Item = u8>) -> Self {
        self.among = Some(chips.into_iter().collect());
        self
    }

    pub fn open(self) -> Result<Device> {
        let Topology { chips, pes } = self.topology;
        {
            let placements = PeRange::placements(pes)?;
            // The chips the host exposes, narrowed to `among` when set.
            let exposed = || -> Result<Vec<u8>> {
                let mut chips = std::fs::read_dir(DEVICE_DIR)
                    .map_err(|why| Error::Topology(why.to_string()))?
                    .filter_map(|entry| {
                        let name = entry.ok()?.file_name();
                        name.to_str()?
                            .strip_prefix("npu")?
                            .strip_suffix("mgmt")?
                            .parse::<u8>()
                            .ok()
                    })
                    .collect::<Vec<_>>();
                chips.sort_unstable();
                chips.dedup();
                Ok(chips)
            };
            let mut available = exposed()?;
            if let Some(among) = &self.among {
                available.retain(|chip| among.contains(chip));
            }
            if chips == 0 || chips as usize > available.len() {
                return Err(Error::Topology(format!(
                    "invalid NPU chip count {chips}: {chips} chip(s) requested, {} present",
                    available.len(),
                )));
            }
            // Only PEs another process holds are worth passing over; any other failure is a
            // fault this process must see.
            let mut reserved = None;
            for (window, placement) in available
                .windows(chips as usize)
                .flat_map(|window| placements.iter().map(move |placement| (window, placement)))
            {
                match self.open_chips(window, placement, pes) {
                    Ok(device) => return Ok(device),
                    Err(Error::Reserved(chip)) => {
                        log::debug!("npu{window:?} {placement:?} is not free: npu{chip} holds it");
                        reserved = Some(Error::Reserved(chip));
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(reserved.unwrap_or_else(|| Error::Topology("no candidate chips".into())))
        }
    }

    /// Opens exactly `chips`, each contributing the clusters at `placement`, `pes` PEs in all.
    fn open_chips(&self, chips: &[u8], placement: &[PeRange], pes: u8) -> Result<Device> {
        let clusters = Cluster::across(chips, placement)?;
        let allocator = Arc::new(Mutex::new(Allocator::new(0..0)));
        let (engine, free) = Engine::open(clusters, IMAGES, Arc::clone(&allocator), self.timeout)?;
        *allocator
            .lock()
            .map_err(|_| Error::Memory("the allocator is poisoned".into()))? = Allocator::new(free);
        Ok(Device {
            engine: Arc::new(engine),
            allocator,
            tokens: std::sync::atomic::AtomicU64::new(0),
            pes,
        })
    }
}

#[cfg(test)]
impl Device {
    /// A device of `chips` with `pes` PEs each and no hardware behind it; see [`Engine::stub`].
    pub(crate) fn stub(chips: &[u8], pes: u8) -> Self {
        let allocator = Arc::new(Mutex::new(Allocator::new(0..4096)));
        Self {
            engine: Arc::new(Engine::stub(chips, Arc::clone(&allocator))),
            allocator,
            tokens: std::sync::atomic::AtomicU64::new(0),
            pes,
        }
    }
}

/// The two images embedded in the host binary and run by every cluster.
const IMAGES: engine::Images = engine::Images {
    boot: include_bytes!(env!("FURIOSA_OPT_BOOTLOADER")),
    device: include_bytes!(env!("FURIOSA_OPT_DEVICE_IMAGE")),
};

const DEVICE_DIR: &str = "/dev/rngd";

impl Device {
    /// Chips in the device; a [`Buffer::on_all`] view moves this many buffers' worth of host bytes.
    pub fn chips(&self) -> usize {
        self.engine.chips()
    }

    /// PEs of each chip the device holds; a function loads only when compiled for as many.
    pub fn pes(&self) -> u8 {
        self.pes
    }

    /// The chip at position `index` of the device, for a [`Buffer::on`] view.
    pub fn rank(&self, index: usize) -> Result<ChipRank> {
        (index < self.chips()).then_some(ChipRank(index)).ok_or_else(|| {
            Error::Topology(format!(
                "chip rank {index} is outside a device of {} chips",
                self.chips()
            ))
        })
    }

    /// Every chip of the device in order, for one [`Buffer::on`] view each.
    pub fn ranks(&self) -> impl Iterator<Item = ChipRank> + use<> {
        (0..self.chips()).map(ChipRank)
    }

    /// Allocates `len` bytes of device memory at one address on every chip.
    pub fn alloc(&self, len: usize) -> Result<Buffer> {
        Buffer::alloc(&self.allocator, len)
            .ok_or_else(|| Error::Memory(format!("device memory has no room left for {len} bytes")))
    }

    /// Copies each host slice into its view: `(bytes, buffer.on(rank))` fills one chip,
    /// `(bytes, buffer.on_all())` deals one buffer's worth per chip in chip order. Every pair moves
    /// at once. Panics when a slice is not exactly its view's size, as `copy_from_slice` does; a
    /// view whose buffer or rank is another device's is [`Error::ForeignBuffer`]. Slices of
    /// [`crate::Pinned`] memory need no pinning for the transfer; others are pinned for its duration.
    pub async fn write(&self, pairs: impl IntoIterator<Item = (&[u8], View)>) -> Result<()> {
        self.engine.write(pairs).await
    }

    /// Copies each view into its host slice, with the same terms as [`Self::write`].
    pub async fn read(&self, pairs: impl IntoIterator<Item = (View, &mut [u8])>) -> Result<()> {
        self.engine.read(pairs).await
    }
}
