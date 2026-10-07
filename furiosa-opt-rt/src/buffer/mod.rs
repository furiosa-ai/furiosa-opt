use std::ops::Range;
use std::sync::Arc;

use crate::device::ChipRank;

mod allocator;
use crate::image::Memory;
pub use allocator::AllocError;
pub(crate) use allocator::{Allocations, Reservation};

/// A range of one device allocation. Clones and slices share the allocation, which is released
/// when the last of them drops.
pub struct Buffer {
    offset: usize,
    len: usize,
    allocation: Arc<Allocation>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("at", &self.addr())
            .field("len", &self.len)
            .finish()
    }
}

impl Buffer {
    pub fn memory(&self) -> Memory {
        self.allocation.memory
    }

    /// The bytes `range` of this buffer on every device, as a buffer of its own sharing the
    /// allocation. Panics past the end, as slicing does.
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "buffer range {range:?} is outside 0..{}",
            self.len
        );
        Self {
            offset: self.offset + range.start,
            len: range.end - range.start,
            allocation: Arc::clone(&self.allocation),
        }
    }

    /// This buffer as it is on the chip ranked `rank` in its device: one buffer's worth of bytes,
    /// for a transfer.
    pub fn on(&self, rank: ChipRank) -> View {
        View {
            buffer: self.clone(),
            chip: Some(rank),
        }
    }

    /// This buffer as it is on every chip, in chip order, for a transfer: the host side holds
    /// one buffer's worth per chip.
    pub fn on_all(&self) -> View {
        View {
            buffer: self.clone(),
            chip: None,
        }
    }

    pub(crate) fn belongs_to(&self, allocations: &Arc<Allocations>) -> bool {
        Arc::ptr_eq(&self.allocation.allocations, allocations)
    }

    pub fn addr(&self) -> usize {
        match self.memory() {
            Memory::Dram => self.allocation.at + self.offset,
            Memory::Sram => furiosa_opt_abi::dm::virtual_addr(self.allocation.at as u64) as usize + self.offset,
        }
    }

    pub(crate) fn bind(&self, window: u64) -> u64 {
        match self.memory() {
            Memory::Dram => window + self.addr() as u64,
            Memory::Sram => self.addr() as u64,
        }
    }

    pub fn size(&self) -> usize {
        self.len
    }

    pub(crate) fn same_base(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation) && self.offset == other.offset
    }

    pub(crate) fn overlaps(&self, len: usize, other: &Self, other_len: usize) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
            && len != 0
            && other_len != 0
            && self.offset < other.offset + other_len
            && other.offset < self.offset + len
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            offset: self.offset,
            len: self.len,
            allocation: Arc::clone(&self.allocation),
        }
    }
}

/// One side of a transfer: a buffer's bytes on one chip, or on every chip in chip order.
#[derive(Clone, Debug)]
pub struct View {
    pub(crate) buffer: Buffer,
    pub(crate) chip: Option<ChipRank>,
}

struct Allocation {
    at: usize,
    memory: Memory,
    allocations: Arc<Allocations>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.allocations.release(self.memory, self.at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_outlives_parent() {
        let allocations = Arc::new(Allocations::new(0..512));
        let buffer = allocations.alloc(Memory::Dram, 256).expect("buffer");
        let tail = buffer.slice(128..256);

        drop(buffer);
        let next = allocations.alloc(Memory::Dram, 256).expect("next buffer");

        assert_eq!((tail.addr(), tail.size()), (128, 128));
        assert_ne!(next.addr(), tail.addr() - 128);
    }

    #[test]
    fn slice_rejects_ranges_past_the_buffer() {
        let allocations = Arc::new(Allocations::new(0..512));
        let buffer = allocations.alloc(Memory::Dram, 256).expect("buffer");

        assert!(std::panic::catch_unwind(|| buffer.slice(0..257)).is_err());
    }

    #[test]
    fn identifies_allocator_provenance() {
        let allocations = Arc::new(Allocations::new(0..256));
        let foreign = Arc::new(Allocations::new(0..256));
        let buffer = allocations.alloc(Memory::Dram, 1).expect("buffer");

        assert!(buffer.belongs_to(&allocations));
        assert!(!buffer.belongs_to(&foreign));
    }

    #[test]
    fn resident_releases_allocation() {
        let allocations = Arc::new(Allocations::new(0..256));
        let resident = allocations.alloc(Memory::Sram, 1).unwrap();
        let address = resident.addr();
        assert_eq!(resident.memory(), Memory::Sram);
        assert!(resident.belongs_to(&allocations));
        drop(resident);
        assert_eq!(allocations.alloc(Memory::Sram, 1).unwrap().addr(), address);
    }

    #[test]
    fn resolves_each_memory() {
        let allocations = Arc::new(Allocations::new(256..512));
        let dram = allocations.alloc(Memory::Dram, 128).unwrap().slice(16..32);
        let sram = allocations.alloc(Memory::Sram, 128).unwrap().slice(16..32);
        for window in [0, 4096] {
            assert_eq!(dram.bind(window), window + 272);
            assert_eq!(sram.bind(window), sram.addr() as u64);
        }
        drop(sram);
        let top = allocations.alloc(Memory::Sram, 256).unwrap();
        let end = top.slice(256..256);
        assert_eq!(end.addr(), top.addr() + 256);
        assert_eq!(end.size(), 0);
    }
}
