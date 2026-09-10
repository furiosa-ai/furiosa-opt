//! Host memory a tensor on the NPU backend lives in: pageable by default, pinned once the caller
//! asks, since a pinned buffer is the DMA's direct source or destination.

use furiosa_opt_rt::Pinned;

use crate::Error;

/// Host bytes of an NPU-backend tensor.
#[derive(Debug)]
pub enum HostBuf {
    /// Ordinary heap memory; a transfer pins its pages for the duration.
    Pageable(Vec<u8>),
    /// Page-locked memory the DMA reads or writes directly.
    Pinned(Pinned<[u8]>),
}

impl HostBuf {
    /// The same bytes in pinned memory; already pinned bytes stay where they are.
    pub fn pinned(self) -> Result<Self, Error> {
        match self {
            Self::Pageable(bytes) => {
                let mut pinned = Pinned::<[u8]>::zeroed(bytes.len())?;
                pinned.copy_from_slice(&bytes);
                Ok(Self::Pinned(pinned))
            }
            pinned @ Self::Pinned(_) => Ok(pinned),
        }
    }
}

impl From<Vec<u8>> for HostBuf {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Pageable(bytes)
    }
}

impl From<HostBuf> for Vec<u8> {
    fn from(buf: HostBuf) -> Self {
        match buf {
            HostBuf::Pageable(bytes) => bytes,
            HostBuf::Pinned(pinned) => pinned.to_vec(),
        }
    }
}

impl AsRef<[u8]> for HostBuf {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Pageable(bytes) => bytes,
            Self::Pinned(pinned) => pinned,
        }
    }
}

impl AsMut<[u8]> for HostBuf {
    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Pageable(bytes) => bytes,
            Self::Pinned(pinned) => pinned,
        }
    }
}

impl Clone for HostBuf {
    /// A copy keeps the memory kind: the caller who pinned once expects every copy to transfer
    /// the same way. Pinned memory that cannot be had panics, as `Vec::clone` does without heap.
    fn clone(&self) -> Self {
        match self {
            Self::Pageable(bytes) => Self::Pageable(bytes.clone()),
            Self::Pinned(pinned) => Self::from(pinned.to_vec()).pinned().expect("pinned memory"),
        }
    }
}
