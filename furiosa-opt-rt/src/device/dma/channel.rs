//! One PDMA channel command: a scatter list the driver moves in one direction, from page-locked
//! memory it already holds by its dma-buf, or from pageable memory it pins for the command.

use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};

use crate::device::engine::{node, open_rw};
use crate::{Error, Result};

pub(crate) const CAPS_DMABUF: u32 = 1;
const URING_CMD_SCATTER_READ: u32 = 0;
const URING_CMD_SCATTER_WRITE: u32 = 1;
const URING_CMD_DMABUF_READ: u32 = 3;
const URING_CMD_DMABUF_WRITE: u32 = 4;

const CHANNELS_PER_DIRECTION: u8 = 4;
const READER_CHANNEL_BASE: u8 = CHANNELS_PER_DIRECTION;

/// One contiguous piece of a scatter list: `src` and `dst` are host and chip addresses by direction, and for
/// page-locked memory the host side is an offset into its dma-buf.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Segment {
    pub(crate) src: u64,
    pub(crate) dst: u64,
    pub(crate) len: u64,
}

const _: () = assert!(std::mem::size_of::<Segment>() == 24);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Writer,
    Reader,
}

/// Where a command's host bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Host {
    /// Any process memory, by address; the driver pins its pages for the command.
    Pageable,
    /// Page-locked memory the driver takes by its dma-buf descriptor.
    Pinned(RawFd),
}

/// One command for one channel: every descriptor moves in `direction` between `host` memory and
/// `chip`. The descriptors are read by the driver when the command is submitted.
pub(crate) struct Command {
    pub(crate) chip: u8,
    pub(crate) direction: Direction,
    pub(crate) host: Host,
    pub(crate) scatter: Vec<Segment>,
}

impl Command {
    pub(crate) fn opcode(&self) -> u32 {
        match (self.host, self.direction) {
            (Host::Pageable, Direction::Writer) => URING_CMD_SCATTER_WRITE,
            (Host::Pageable, Direction::Reader) => URING_CMD_SCATTER_READ,
            (Host::Pinned(_), Direction::Writer) => URING_CMD_DMABUF_WRITE,
            (Host::Pinned(_), Direction::Reader) => URING_CMD_DMABUF_READ,
        }
    }

    /// The sixteen command bytes: `{count: u64, desc_addr: u64}` for pageable memory, and
    /// `{fd: i32, count: u32, desc_addr: u64}` for a dma-buf.
    pub(crate) fn payload(&self) -> [u8; 16] {
        let mut payload = [0; 16];
        let descriptors = (self.scatter.as_ptr().expose_provenance() as u64).to_ne_bytes();
        match self.host {
            Host::Pageable => {
                payload[..8].copy_from_slice(&(self.scatter.len() as u64).to_ne_bytes());
            }
            Host::Pinned(fd) => {
                payload[..4].copy_from_slice(&fd.to_ne_bytes());
                payload[4..8].copy_from_slice(&(self.scatter.len() as u32).to_ne_bytes());
            }
        }
        payload[8..].copy_from_slice(&descriptors);
        payload
    }
}

nix::ioctl_none!(get_caps, b'N', 0x00);

pub(crate) struct Channel(File);

impl Channel {
    pub(crate) fn open(chip: u8, direction: Direction, slot: u8) -> Result<Self> {
        Ok(Self(open_rw(&Self::path(chip, direction, slot)?)?))
    }

    pub(crate) fn path(chip: u8, direction: Direction, slot: u8) -> Result<String> {
        if slot >= CHANNELS_PER_DIRECTION {
            return Err(Error::Transfer(format!(
                "PDMA {direction:?} slot {slot} is outside 0..{CHANNELS_PER_DIRECTION}"
            )));
        }
        let channel = match direction {
            Direction::Writer => slot,
            Direction::Reader => READER_CHANNEL_BASE + slot,
        };
        Ok(node(chip, &format!("ch{channel}")))
    }

    pub(crate) fn caps(&self) -> Result<u32> {
        // SAFETY: `self` owns the channel fd and this ioctl takes no userspace payload.
        unsafe { get_caps(self.0.as_raw_fd()) }
            .map_err(|why| Error::Transfer(format!("asking PDMA capabilities: {why}")))
            .and_then(|bits| {
                u32::try_from(bits).map_err(|_| Error::Transfer(format!("PDMA returned negative capabilities {bits}")))
            })
    }

    pub(crate) fn fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcode_follows_host_memory_and_direction() {
        let command = |host, direction| Command {
            chip: 0,
            direction,
            host,
            scatter: vec![],
        };

        assert_eq!(
            command(Host::Pageable, Direction::Writer).opcode(),
            URING_CMD_SCATTER_WRITE
        );
        assert_eq!(
            command(Host::Pinned(3), Direction::Reader).opcode(),
            URING_CMD_DMABUF_READ
        );
        assert_eq!(
            &command(Host::Pinned(3), Direction::Reader).payload()[..8],
            &[3, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}
