//! Page-locked host memory, as `cudaMallocHost` gives it: a transfer from or into it needs no
//! per-call pinning. The runtime recognizes slices of it by address, so a plain `&pinned[a..b]`
//! takes the fast path with nothing said at the call.

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut, Range};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::RwLock;

use dma_heap::{Heap, HeapKind};
use memmap2::MmapMut;

use crate::{Error, Result};

/// Page-locked host memory holding a `[T]`.
#[derive(Debug)]
pub struct Pinned<T: ?Sized> {
    /// Kept open for the mapping's lifetime: the driver takes the dma-buf by this descriptor.
    _fd: OwnedFd,
    map: MmapMut,
    /// Bytes the caller asked for; the mapping is page-rounded past them.
    len: usize,
    marker: std::marker::PhantomData<T>,
}

/// The dma-buf behind a pinned range: its descriptor and where the range starts in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Backing {
    pub(crate) fd: RawFd,
    pub(crate) offset: usize,
}

/// Every live pinned mapping by its start address, so a slice's address finds its dma-buf.
static REGISTRY: RwLock<BTreeMap<usize, (RawFd, usize)>> = RwLock::new(BTreeMap::new());

impl<T: bytemuck::Pod> Pinned<[T]> {
    /// `len` zeroed elements of page-locked memory.
    pub fn zeroed(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| Error::Transfer(format!("{len} elements of {} bytes cannot be pinned", size_of::<T>())))?;
        let fd = Heap::new(HeapKind::System)
            .and_then(|heap| heap.allocate(bytes))
            .map_err(|why| Error::Transfer(format!("allocating {bytes} bytes of pinned memory: {why}")))?;
        // SAFETY: the dma-buf is this process's own and nothing else maps it.
        let map = unsafe { MmapMut::map_mut(&fd) }
            .map_err(|why| Error::Transfer(format!("mapping {bytes} bytes of pinned memory: {why}")))?;
        REGISTRY
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(map.as_ptr().addr(), (fd.as_raw_fd(), bytes));
        Ok(Self {
            _fd: fd,
            map,
            len: bytes,
            marker: std::marker::PhantomData,
        })
    }
}

impl<T: bytemuck::Pod> Deref for Pinned<[T]> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        bytemuck::cast_slice(&self.map[..self.len])
    }
}

impl<T: bytemuck::Pod> DerefMut for Pinned<[T]> {
    fn deref_mut(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(&mut self.map[..self.len])
    }
}

impl<T: ?Sized> Drop for Pinned<T> {
    fn drop(&mut self) {
        REGISTRY
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&self.map.as_ptr().addr());
    }
}

/// The dma-buf holding `bytes`, when a pinned allocation contains all of them.
pub(crate) fn backing(bytes: Range<usize>) -> Option<Backing> {
    let registry = REGISTRY.read().unwrap_or_else(|poison| poison.into_inner());
    let (&start, &(fd, len)) = registry.range(..=bytes.start).next_back()?;
    (bytes.end <= start + len).then_some(Backing {
        fd,
        offset: bytes.start - start,
    })
}
