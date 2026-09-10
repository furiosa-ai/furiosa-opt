use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::device::ChipRank;

mod allocator;
pub(crate) use allocator::Allocator;

/// A range of one device allocation. Clones and slices share the allocation, which is released
/// when the last of them drops.
pub struct Buffer {
    at: usize,
    len: usize,
    allocation: Arc<Allocation>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("at", &self.at)
            .field("len", &self.len)
            .finish()
    }
}

impl Buffer {
    pub(crate) fn alloc(memory: &Arc<Mutex<Allocator>>, len: usize) -> Option<Self> {
        let at = memory.lock().ok()?.alloc(len)?;
        Some(Self {
            at,
            len,
            allocation: Arc::new(Allocation {
                at,
                len,
                memory: Arc::clone(memory),
            }),
        })
    }

    /// The bytes `range` of this buffer on every chip, as a buffer of its own sharing the
    /// allocation. Panics past the end, as slicing does.
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "buffer range {range:?} is outside 0..{}",
            self.len
        );
        Self {
            at: self.at + range.start,
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

    pub(crate) fn belongs_to(&self, memory: &Arc<Mutex<Allocator>>) -> bool {
        Arc::ptr_eq(&self.allocation.memory, memory)
    }

    pub fn addr(&self) -> usize {
        self.at
    }

    pub fn size(&self) -> usize {
        self.len
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            at: self.at,
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
    len: usize,
    memory: Arc<Mutex<Allocator>>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Ok(mut memory) = self.memory.lock() {
            memory.release(self.at, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_outlives_parent() {
        let allocator = Arc::new(Mutex::new(Allocator::new(0..512)));
        let buffer = Buffer::alloc(&allocator, 256).expect("buffer");
        let tail = buffer.slice(128..256);

        drop(buffer);
        let next = Buffer::alloc(&allocator, 256).expect("next buffer");

        assert_eq!((tail.addr(), tail.size()), (128, 128));
        assert_ne!(next.addr(), tail.addr() - 128);
    }

    #[test]
    fn slice_rejects_ranges_past_the_buffer() {
        let allocator = Arc::new(Mutex::new(Allocator::new(0..512)));
        let buffer = Buffer::alloc(&allocator, 256).expect("buffer");

        assert!(std::panic::catch_unwind(|| buffer.slice(0..257)).is_err());
    }

    #[test]
    fn identifies_allocator_provenance() {
        let allocator = Arc::new(Mutex::new(Allocator::new(0..256)));
        let foreign = Arc::new(Mutex::new(Allocator::new(0..256)));
        let buffer = Buffer::alloc(&allocator, 1).expect("buffer");

        assert!(buffer.belongs_to(&allocator));
        assert!(!buffer.belongs_to(&foreign));
    }
}
