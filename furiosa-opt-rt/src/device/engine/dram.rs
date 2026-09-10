use std::ops::Range;
use std::os::fd::AsRawFd;

use super::DRAM_BASE;
use super::mmio::{PES, node, open_rw};
use crate::device::dma::DmaQueue;
use crate::{Error, Result};

/// Where a chip's memory window may start in a cluster's address space; the bootloader tells
/// which by finding the firmware image at it.
const CANDIDATES: [u64; 3] = [0, 256 << 20, 512 << 20];

const PAGE: usize = 4096;

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct BarInfo {
    phys: u64,
    size: u64,
}

nix::ioctl_readwrite!(available_dram, b'N', 0x02, BarInfo);

/// One chip's memory window and the share of it the device's PEs own, an eighth per PE: the
/// firmware image at the share's top, everything below it for the allocator.
pub(crate) struct Dram {
    chip: u8,
    len: usize,
    share: Range<usize>,
    placed: Staged,
    /// Where a cluster sees this window, once a boot found the image in it.
    at: Option<u64>,
}

/// Where the firmware image is staged in the window, and the token that tells this staging apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Staged {
    pub(crate) offset: usize,
    /// Chosen per staging and folded into the image's first word, so a boot that finds an image
    /// left by an earlier runtime at the same address does not run it.
    pub(crate) token: u64,
}

impl Dram {
    /// The share of `chip`'s memory the PE bitmask `pes` owns, with `image` placed at its top;
    /// [`Self::write`] puts the image there.
    pub(crate) fn open(chip: u8, pes: u8, image: &[u8], token: u64) -> Result<Self> {
        let bar = open_rw(&node(chip, "bar4"))?;
        let mut usable = BarInfo::default();
        // SAFETY: `usable` is a valid writable `BarInfo` for this ioctl.
        unsafe { available_dram(bar.as_raw_fd(), &mut usable) }
            .map_err(|why| Error::Device(format!("asking for the usable window: {why}")))?;
        let available = usable.size;
        let len = usize::try_from(available)
            .map_err(|_| Error::Memory(format!("a {available} byte window is more than this host addresses")))?
            & !(PAGE - 1);
        let per_pe = (len / usize::from(PES)) & !(PAGE - 1);
        let (first, past) = (
            pes.trailing_zeros() as usize,
            usize::from(PES) - pes.leading_zeros() as usize,
        );
        let share = first * per_pe..past * per_pe;
        let offset = share
            .end
            .saturating_sub(PAGE)
            .checked_sub(image.len())
            .filter(|offset| *offset >= share.start)
            .ok_or_else(|| Error::Memory(format!("{} bytes do not fit above the share's floor", image.len())))?
            & !(PAGE - 1);
        Ok(Self {
            chip,
            len,
            share,
            placed: Staged { offset, token },
            at: None,
        })
    }

    /// Writes `image` where it is staged, as a cluster seeing this window at `candidate` addresses
    /// it, with the token folded into the opening word so a boot cannot mistake a stale image.
    pub(crate) fn write(&self, dma: &mut DmaQueue, candidate: u64, image: &[u8]) -> Result<()> {
        let (magic, rest) = image
            .split_at_checked(8)
            .ok_or_else(|| Error::Firmware("firmware image is too short".into()))?;
        let magic = u64::from_le_bytes(magic.try_into().expect("eight bytes"));
        let bytes = [&(magic ^ self.placed.token).to_le_bytes()[..], rest].concat();
        dma.write(self.chip, DRAM_BASE + candidate + self.placed.offset as u64, &bytes)
    }

    pub(crate) fn chip(&self) -> u8 {
        self.chip
    }

    pub(crate) fn placed(&self) -> Staged {
        self.placed
    }

    /// Where a cluster may see this window: the located address once one cluster booted, every
    /// candidate before.
    pub(crate) fn candidates(&self) -> Vec<u64> {
        self.at.map_or(CANDIDATES.to_vec(), |at| vec![at])
    }

    /// Where a cluster sees this window, once a boot found the image staged in it.
    pub(crate) fn window(&self) -> Option<u64> {
        self.at
    }

    /// Records where a cluster sees this window, once a boot found the image staged in it.
    pub(crate) fn located(&mut self, candidate: u64) {
        self.at = Some(candidate);
    }

    /// What an arg table names `offset` of this window by. Device memory is addressed from its
    /// own start there, not from where a chip sees it.
    pub(crate) fn arg(&self, offset: usize) -> Result<u64> {
        let at = self
            .at
            .ok_or_else(|| Error::Memory("this window has not been located yet".into()))?;
        if offset > self.len {
            return Err(Error::Memory(format!(
                "offset {offset} falls outside the {} byte window",
                self.len
            )));
        }
        Ok(at + offset as u64)
    }

    /// The offsets of this window the allocator may hand out: the share below the staged image.
    pub(crate) fn free(&self) -> Range<usize> {
        self.share.start..self.placed.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_args_from_the_located_window() {
        let mut dram = Dram {
            chip: 0,
            len: 4096,
            share: 0..4096,
            placed: Staged { offset: 0, token: 0 },
            at: None,
        };
        assert!(dram.arg(0).is_err());

        dram.located(256 << 20);

        assert_eq!(dram.arg(16).expect("inside the window"), (256 << 20) + 16);
        assert!(dram.arg(4097).is_err());
    }
}
