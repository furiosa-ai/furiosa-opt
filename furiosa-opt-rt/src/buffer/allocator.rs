use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use furiosa_opt_abi::dm;

use super::{Allocation, Buffer, Memory};

/// Allocation and launch failures preserve the memory kind and requested extent.
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    #[error("{memory:?} has no contiguous space for {requested} bytes")]
    OutOfMemory { memory: Memory, requested: usize },
    #[error("a {requested} byte SRAM allocation starts at {start}, below running temporaries at {end}")]
    Busy { requested: usize, start: usize, end: usize },
    #[error("kernel temporaries end at {end}, above live SRAM allocations at {start}")]
    Collision { end: usize, start: usize },
    #[error("the allocator is poisoned by an earlier panic")]
    Poisoned,
}

/// One device's independently synchronized DRAM and SRAM allocations.
pub(crate) struct Allocations {
    dram: Mutex<Allocator>,
    sram: Mutex<Sram>,
}

struct Allocator {
    range: Range<usize>,
    live: BTreeMap<usize, usize>,
}

struct Sram {
    allocator: Allocator,
    // Equal prefixes can overlap, so each end retains its reservation count.
    reserved: BTreeMap<usize, usize>,
}

impl Allocator {
    /// Hands out `range`, narrowed to whole aligned units.
    fn new(range: Range<usize>) -> Self {
        let start = range.start.next_multiple_of(dm::ALIGNMENT as usize);
        let end = range.end / dm::ALIGNMENT as usize * dm::ALIGNMENT as usize;
        Self {
            range: start.min(end)..end,
            live: BTreeMap::new(),
        }
    }

    fn gaps(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.live
            .iter()
            .map(|(&at, &len)| at..at + len)
            .chain(std::iter::once(self.range.end..self.range.end))
            .scan(self.range.start, |end, occupied| {
                let gap = *end..occupied.start;
                *end = occupied.end;
                Some(gap)
            })
    }

    fn start(&self) -> usize {
        self.live.first_key_value().map_or(self.range.end, |(&at, _)| at)
    }
}

impl Allocations {
    pub(crate) fn new(dram: Range<usize>) -> Self {
        Self {
            dram: Mutex::new(Allocator::new(dram)),
            sram: Mutex::new(Sram {
                allocator: Allocator::new(dm::RESIDENT_BASE as usize..dm::RESIDENT_END as usize),
                reserved: BTreeMap::new(),
            }),
        }
    }

    pub(crate) fn memory(&self) -> Result<crate::Memory, AllocError> {
        let allocator = self.dram.lock().map_err(|_| AllocError::Poisoned)?;
        let (available, largest) = allocator.gaps().fold((0, 0), |(available, largest), gap| {
            (available + gap.len(), largest.max(gap.len()))
        });
        Ok(crate::Memory {
            capacity: allocator.range.len(),
            available,
            largest,
        })
    }

    pub(crate) fn alloc(self: &Arc<Self>, memory: Memory, requested: usize) -> Result<Buffer, AllocError> {
        let len = requested
            .max(1)
            .checked_next_multiple_of(dm::ALIGNMENT as usize)
            .ok_or(AllocError::OutOfMemory { memory, requested })?;
        let at = match memory {
            Memory::Dram => {
                let mut allocator = self.dram.lock().map_err(|_| AllocError::Poisoned)?;
                let at = allocator
                    .gaps()
                    .find(|gap| gap.end - gap.start >= len)
                    .map(|gap| gap.start)
                    .ok_or(AllocError::OutOfMemory { memory, requested })?;
                allocator.live.insert(at, len);
                at
            }
            Memory::Sram => {
                let mut sram = self.sram.lock().map_err(|_| AllocError::Poisoned)?;
                let at = sram
                    .allocator
                    .gaps()
                    .filter(|gap| gap.end - gap.start >= len)
                    .last()
                    .map(|gap| gap.end - len)
                    .ok_or(AllocError::OutOfMemory { memory, requested })?;
                if let Some((&end, _)) = sram.reserved.last_key_value()
                    && at < end
                {
                    return Err(AllocError::Busy {
                        requested,
                        start: at,
                        end,
                    });
                }
                sram.allocator.live.insert(at, len);
                at
            }
        };
        Ok(Buffer {
            offset: 0,
            len: requested,
            allocation: Arc::new(Allocation {
                at,
                memory,
                allocations: Arc::clone(self),
            }),
        })
    }

    pub(crate) fn reserve(self: &Arc<Self>, end: usize) -> Result<Reservation, AllocError> {
        let mut sram = self.sram.lock().map_err(|_| AllocError::Poisoned)?;
        let start = sram.allocator.start();
        if end > start {
            return Err(AllocError::Collision { end, start });
        }
        *sram.reserved.entry(end).or_default() += 1;
        Ok(Reservation {
            allocations: Arc::clone(self),
            end,
        })
    }

    pub(super) fn release(&self, memory: Memory, at: usize) {
        // Cleanup remains possible after poison; new operations never recover it.
        match memory {
            Memory::Dram => {
                self.dram
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .live
                    .remove(&at);
            }
            Memory::Sram => {
                self.sram
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .allocator
                    .live
                    .remove(&at);
            }
        }
    }
}

/// A temporary SRAM prefix, retained until its device work finishes.
pub(crate) struct Reservation {
    allocations: Arc<Allocations>,
    end: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut sram = self
            .allocations
            .sram
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(count) = sram.reserved.get_mut(&self.end) {
            if *count > 1 {
                *count -= 1;
            } else {
                sram.reserved.remove(&self.end);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_aligned_capacity() {
        let allocations = Allocations::new(1000..4097);
        let memory = allocations.memory().unwrap();
        assert_eq!(memory.capacity, 3072);
        assert_eq!(memory.available, 3072);
        assert_eq!(memory.largest, 3072);
    }

    #[test]
    fn reports_fragmentation() {
        let allocations = Arc::new(Allocations::new(1024..5120));
        let first = allocations.alloc(Memory::Dram, 1024).unwrap();
        let second = allocations.alloc(Memory::Dram, 1024).unwrap();
        drop(first);
        let memory = allocations.memory().unwrap();
        assert_eq!(memory.capacity, 4096);
        assert_eq!(memory.available, 3072);
        assert_eq!(memory.largest, 2048);
        assert!(allocations.alloc(Memory::Dram, memory.largest + 1).is_err());
        assert_eq!(
            allocations.alloc(Memory::Dram, memory.largest).unwrap().addr(),
            second.addr() + 1024
        );
    }

    #[test]
    fn reports_coalesced_space() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let first = allocations.alloc(Memory::Dram, 1024).unwrap();
        let second = allocations.alloc(Memory::Dram, 1024).unwrap();
        drop(first);
        drop(second);
        assert_eq!(allocations.memory().unwrap().available, 4096);
        assert_eq!(allocations.memory().unwrap().largest, 4096);
    }

    #[test]
    fn reports_exhaustion() {
        let allocations = Arc::new(Allocations::new(0..256));
        let buffer = allocations.alloc(Memory::Dram, 256).unwrap();
        let memory = allocations.memory().unwrap();
        assert_eq!(memory.capacity, buffer.size());
        assert_eq!(memory.available, 0);
        assert_eq!(memory.largest, 0);
    }

    #[test]
    fn allocates_dram_upward() {
        let allocations = Arc::new(Allocations::new(1000..2048));
        let first = allocations.alloc(Memory::Dram, 256).unwrap();
        assert_eq!(first.addr(), 1024);
        assert!(matches!(
            allocations.alloc(Memory::Dram, 1024),
            Err(AllocError::OutOfMemory { .. })
        ));
        assert_eq!(allocations.alloc(Memory::Dram, 768).unwrap().addr(), 1280);
        drop(first);
        assert_eq!(allocations.alloc(Memory::Dram, 256).unwrap().addr(), 1024);
    }

    #[test]
    fn reuses_sram_gaps() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let top = allocations.alloc(Memory::Sram, 256).unwrap();
        let middle = allocations.alloc(Memory::Sram, 1).unwrap();
        let bottom = allocations.alloc(Memory::Sram, 257).unwrap();
        assert_eq!(top.addr(), dm::virtual_addr(dm::RESIDENT_END - 256) as usize);
        assert_eq!(bottom.addr(), middle.addr() - 512);
        let hole = middle.addr();
        drop(middle);
        let larger = allocations.alloc(Memory::Sram, 512).unwrap();
        assert_eq!(larger.addr(), bottom.addr() - 512);
        assert_eq!(allocations.alloc(Memory::Sram, 1).unwrap().addr(), hole);
    }

    #[test]
    fn rejects_oversized_requests() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let live = allocations.alloc(Memory::Sram, 1).unwrap();
        for requested in [dm::RESIDENT_END as usize + 1, usize::MAX] {
            assert!(
                matches!(allocations.alloc(Memory::Sram, requested), Err(AllocError::OutOfMemory { requested: value, .. }) if value == requested)
            );
        }
        assert_eq!(allocations.alloc(Memory::Sram, 1).unwrap().addr(), live.addr() - 256);
    }

    #[test]
    fn rejects_colliding_launches() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let live = allocations.alloc(Memory::Sram, 256).unwrap();
        let start = dm::physical_addr(live.addr() as u64) as usize;
        assert!(allocations.reserve(start).is_ok());
        assert!(
            matches!(allocations.reserve(start + 1), Err(AllocError::Collision { end, start: value }) if end == start + 1 && value == start)
        );
    }

    #[test]
    fn retains_each_launch_boundary() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let end = dm::RESIDENT_END as usize;
        let lower = allocations.reserve(end - 512).unwrap();
        let first = allocations.reserve(end).unwrap();
        let second = allocations.reserve(end).unwrap();
        drop(first);
        assert!(matches!(
            allocations.alloc(Memory::Sram, 1),
            Err(AllocError::Busy { .. })
        ));
        drop(second);
        let live = allocations.alloc(Memory::Sram, 512).unwrap();
        assert!(matches!(
            allocations.alloc(Memory::Sram, 1),
            Err(AllocError::Busy { .. })
        ));
        drop(lower);
        assert_eq!(allocations.alloc(Memory::Sram, 1).unwrap().addr(), live.addr() - 256);
    }

    #[test]
    fn poison_isolates_memory() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let dram = allocations.alloc(Memory::Dram, 1).unwrap();
        let sram = allocations.alloc(Memory::Sram, 1).unwrap();
        let running = allocations.reserve(256).unwrap();
        let _ = std::panic::catch_unwind(|| {
            let sram = allocations.sram.lock().unwrap();
            assert!(sram.reserved.is_empty());
        });
        drop((dram, sram, running));
        assert!(allocations.alloc(Memory::Dram, 1).is_ok());
        assert!(matches!(allocations.alloc(Memory::Sram, 1), Err(AllocError::Poisoned)));
        assert!(matches!(allocations.reserve(0), Err(AllocError::Poisoned)));
        let sram = allocations.sram.lock().err().unwrap().into_inner();
        assert!(sram.allocator.live.is_empty());
        assert!(sram.reserved.is_empty());
    }

    #[test]
    fn serializes_colliding_requests() {
        for _ in 0..64 {
            let allocations = Arc::new(Allocations::new(0..4096));
            let barrier = std::sync::Barrier::new(2);
            std::thread::scope(|scope| {
                let allocation = scope.spawn(|| {
                    barrier.wait();
                    allocations.alloc(Memory::Sram, 256)
                });
                barrier.wait();
                let reservation = allocations.reserve(dm::RESIDENT_END as usize);
                let allocation = allocation.join().unwrap();
                assert_ne!(allocation.is_ok(), reservation.is_ok());
                match (&allocation, &reservation) {
                    (Ok(_), Err(AllocError::Collision { .. })) | (Err(AllocError::Busy { .. }), Ok(_)) => {}
                    _ => panic!("conflicting requests must have exactly one winner"),
                }
            });
            assert!(allocations.reserve(dm::RESIDENT_END as usize).is_ok());
        }
    }
}
