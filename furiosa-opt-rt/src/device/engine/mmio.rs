use std::fs::{File, OpenOptions};
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use nix::sys::mman::{MapFlags, ProtFlags, mmap, munmap};

use crate::{Error, Result};

pub(super) const Q_BASE: u64 = 0x40_0000;
pub(super) const Q_STRIDE: u64 = 1024;
pub(super) const DOORBELL: u64 = 0x80_0000;
pub(super) const TAIL0: u64 = 0x800;
pub(super) const HEAD0: u64 = 0x900;
pub(super) const PE_WINDOW: u64 = 512 << 20;
pub(super) const PES: u8 = 8;
pub(super) const BAR2_LEN: usize = (DOORBELL + 0x1000 - Q_BASE) as usize;
pub(super) const CQ_REGION: i64 = 1 << 16;
pub(super) const CQ_LEN: usize = 4096;

/// PEs one cluster fuses at most; a chip has two such clusters.
pub const CLUSTER_PES: u8 = 4;
pub const CLUSTERS_PER_DEVICE: u32 = (PES / CLUSTER_PES) as u32;

pub(crate) fn node(chip: u8, name: &str) -> String {
    format!("/dev/rngd/npu{chip}{name}")
}

pub(crate) fn open_rw(path: &str) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|why| Error::Device(format!("{path}: {why}")))
}

pub(crate) struct Mapping {
    pub(crate) at: NonNull<u8>,
    pub(crate) len: usize,
}

// SAFETY: Category 9. A mapping moves only with its owning link;
// the link never shares a mapping reference between threads.
unsafe impl Send for Mapping {}

impl Mapping {
    pub(crate) fn new(file: &File, offset: i64, len: usize) -> Result<Self> {
        let len =
            NonZeroUsize::new(len).ok_or_else(|| Error::Device("a mapping of no bytes addresses nothing".into()))?;
        // SAFETY: the driver maps device memory here and the length is checked against the region's own size.
        let at = unsafe {
            mmap(
                None,
                len,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                file,
                offset,
            )
        }
        .map_err(|why| Error::Device(format!("mapping at {offset:#x} failed: {why}")))?;
        Ok(Self {
            at: at.cast(),
            len: len.get(),
        })
    }
}

impl Mapping {
    /// Writes `bytes` at `offset`: whole words from the first aligned address, since word stores
    /// post whole over the uncached mapping, and the ragged ends bytewise. Panics past the mapping.
    pub(crate) fn store(&self, offset: usize, bytes: &[u8]) {
        assert!(
            offset.checked_add(bytes.len()).is_some_and(|end| end <= self.len),
            "{} bytes at {offset:#x} fall outside a {:#x} byte mapping",
            bytes.len(),
            self.len
        );
        let target = self.at.as_ptr().wrapping_add(offset);
        let head = head(target.addr(), bytes.len());
        let (words, tail) = bytes[head..].as_chunks::<WORD>();
        // SAFETY: the range is inside the mapping, checked above, and `head` moves the word stores
        // to an aligned address.
        unsafe {
            for (index, byte) in bytes[..head].iter().enumerate() {
                target.add(index).write_volatile(*byte);
            }
            for (index, word) in words.iter().enumerate() {
                target
                    .add(head)
                    .cast::<u64>()
                    .add(index)
                    .write_volatile(u64::from_ne_bytes(*word));
            }
            for (index, byte) in tail.iter().enumerate() {
                target.add(head + words.len() * WORD + index).write_volatile(*byte);
            }
        }
    }
}

const WORD: usize = size_of::<u64>();

/// Bytes at `addr` before the first word-aligned address, at most `len`.
const fn head(addr: usize, len: usize) -> usize {
    let head = addr.wrapping_neg() % WORD;
    if head < len { head } else { len }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this is the mapping made in `new` and nothing else unmaps it.
        let _ = unsafe { munmap(self.at.cast(), self.len) };
    }
}

/// The PEs one cluster fuses on a chip: `count` (1, 2 or 4) from `lead`, a multiple of `count`,
/// exactly the driver's fusion nodes. The lead PE's queue and control window are the cluster's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PeRange {
    lead: u8,
    count: u8,
}

impl PeRange {
    pub fn new(lead: u8, count: u8) -> Result<Self> {
        if !matches!(count, 1 | 2 | 4) || !lead.is_multiple_of(count) || lead + count > PES {
            return Err(Error::Topology(format!(
                "a cluster fuses 1, 2 or 4 aligned PEs of {PES}, not {count} from pe{lead}"
            )));
        }
        Ok(Self { lead, count })
    }

    /// Every way `pes` PEs sit on a chip, lowest first: one aligned slice of that size, or for
    /// eight, both four-PE clusters.
    pub(crate) fn placements(pes: u8) -> Result<Vec<Vec<Self>>> {
        match pes {
            1 | 2 | 4 => (0..PES / pes)
                .map(|slice| Self::new(slice * pes, pes).map(|range| vec![range]))
                .collect(),
            8 => Ok(vec![
                (0..PES / CLUSTER_PES)
                    .map(|slice| Self::new(slice * CLUSTER_PES, CLUSTER_PES))
                    .collect::<Result<_>>()?,
            ]),
            _ => Err(Error::Topology(format!(
                "invalid NPU PE count {pes}: expected 1, 2, 4 or 8"
            ))),
        }
    }

    /// The doorbell queue of this cluster on the chip ranked `member` in its device.
    pub(super) fn queue(self, member: u8) -> u64 {
        u64::from(member) * u64::from(PES) + u64::from(self.lead)
    }

    pub(super) fn window(self) -> u64 {
        u64::from(self.lead) * PE_WINDOW
    }

    /// The driver's bitmask of these PEs.
    pub(super) fn mask(self) -> u8 {
        ((1 << self.count) - 1) << self.lead
    }

    pub(super) fn node(self) -> String {
        match self.count {
            1 => format!("pe{}", self.lead),
            count => format!("pe{}-{}", self.lead, self.lead + count - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_store_starts_aligned() {
        assert_eq!(head(0x1000, 16), 0);
        assert_eq!(head(0x1001, 16), 7);
        assert_eq!(head(0x100f, 16), 1);
        assert_eq!(head(0x1001, 4), 4);
    }

    #[test]
    fn unaligned_store_lands_whole() {
        let file = tempfile::tempfile().expect("a file to map");
        file.set_len(4096).expect("a page to map");
        let mapping = Mapping::new(&file, 0, 4096).expect("mapping");
        let bytes = (1..=19).collect::<Vec<u8>>();

        mapping.store(1, &bytes);

        let mut landed = vec![0; 32];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut landed, 0).expect("read back");
        assert_eq!(&landed[1..20], bytes.as_slice());
        assert_eq!((landed[0], landed[20]), (0, 0));
    }

    #[test]
    fn a_store_past_the_mapping_panics() {
        let file = tempfile::tempfile().expect("a file to map");
        file.set_len(4096).expect("a page to map");
        let mapping = Mapping::new(&file, 0, 4096).expect("mapping");

        assert!(std::panic::catch_unwind(|| mapping.store(4090, &[0; 8])).is_err());
    }

    #[test]
    fn routes_cluster_addresses_within_bar() {
        for (lead, count, mask, queue, window, node) in [
            (0, 4, 0b0000_1111, 0, 0, "pe0-3"),
            (4, 4, 0b1111_0000, 4, 0x8000_0000, "pe4-7"),
            (2, 2, 0b0000_1100, 2, 0x4000_0000, "pe2-3"),
            (5, 1, 0b0010_0000, 5, 0xa000_0000, "pe5"),
        ] {
            let cluster = PeRange::new(lead, count).expect("cluster");
            assert_eq!(
                (cluster.mask(), cluster.queue(0), cluster.window(), cluster.node()),
                (mask, queue, window, node.to_owned())
            );
        }
        let doorbell = (DOORBELL - Q_BASE) as usize;
        for member in 0..furiosa_opt_ipc::MAX_GROUP as u8 {
            for lead in 0..PES {
                let queue = PeRange::new(lead, 1).expect("cluster").queue(member) as usize;
                assert!(doorbell + HEAD0 as usize + 4 * queue + 4 <= BAR2_LEN, "{member}/{lead}");
                assert!(Q_STRIDE as usize * queue < BAR2_LEN, "{member}/{lead}");
            }
        }
        assert_eq!(BAR2_LEN, 0x401000);
    }

    #[test]
    fn rejects_misaligned_and_oversized_ranges() {
        for (lead, count) in [(1, 2), (2, 4), (3, 1 + 2), (8, 1), (6, 4), (0, 8)] {
            assert!(PeRange::new(lead, count).is_err(), "pe{lead} x{count}");
        }
    }

    #[test]
    fn clusters_fill_aligned_slices() {
        let leads = |pes| {
            PeRange::placements(pes)
                .expect("placements")
                .into_iter()
                .map(|placement| placement.into_iter().map(|range| range.lead).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(leads(1), [[0], [1], [2], [3], [4], [5], [6], [7]]);
        assert_eq!(leads(2), [[0], [2], [4], [6]]);
        assert_eq!(leads(4), [[0], [4]]);
        assert_eq!(leads(8), [[0, 4]]);
        for pes in [0, 3, 5, 6, 7, 9] {
            assert!(PeRange::placements(pes).is_err(), "{pes}");
        }
    }
}
