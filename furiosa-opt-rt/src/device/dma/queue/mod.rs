use std::collections::{BTreeMap, BTreeSet};

use super::channel::{CAPS_DMABUF, Channel, Command, Direction, Host, Segment};
use crate::{Error, Result};

mod uring;
use uring::{Cmd, Ring};

const DEPTH: u32 = 64;

/// The PDMA channels of a device's chips behind one ring. A command goes to the next free channel
/// of its chip and direction, in rotation; the commands of one transfer complete together, under
/// the ticket their submission returned.
pub(crate) struct DmaQueue {
    ring: Ring,
    channels: BTreeMap<u8, Channels>,
    pending: BTreeMap<u64, Pending>,
    tickets: u64,
}

struct Channels {
    writers: Vec<Channel>,
    readers: Vec<Channel>,
    writer: usize,
    reader: usize,
}

/// One transfer whose completions are still arriving.
struct Pending {
    remaining: usize,
    failed: Option<Error>,
    /// Kept until the last completion: a queued entry points into its command's scatter list.
    _commands: Vec<Command>,
}

impl DmaQueue {
    pub(crate) fn open(chips: impl IntoIterator<Item = u8>) -> Result<Self> {
        let channels = chips
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|chip| Channels::open(chip).map(|channels| (chip, channels)))
            .collect::<Result<_>>()?;
        Ok(Self {
            ring: Ring::open(DEPTH).map_err(|why| Error::Transfer(format!("opening io_uring: {why}")))?,
            channels,
            pending: BTreeMap::new(),
            tickets: 0,
        })
    }

    /// Submits every command of one transfer, without waiting, and returns the ticket
    /// [`Self::complete`] answers under once all of them are done. A batch the ring has no room
    /// for fails with nothing queued.
    pub(crate) fn submit(&mut self, commands: Vec<Command>) -> Result<u64> {
        let ticket = self.tickets;
        let cmds = commands
            .iter()
            .map(|command| {
                let channels = self
                    .channels
                    .get_mut(&command.chip)
                    .ok_or_else(|| Error::Transfer(format!("no DMA channels for chip {}", command.chip)))?;
                Ok(Cmd {
                    fd: channels.next(command.direction).fd(),
                    opcode: command.opcode(),
                    payload: command.payload(),
                    tag: ticket,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.tickets = self.tickets.wrapping_add(1);
        // Owned before the first entry is queued: the ring points into the scatter lists from then on.
        let pending = self.pending.entry(ticket).insert_entry(Pending {
            remaining: commands.len(),
            failed: None,
            _commands: commands,
        });
        if pending.get().remaining == 0 {
            return Ok(ticket);
        }
        match self.ring.submit(cmds.into_iter()) {
            Ok(()) => Ok(ticket),
            Err(why) if why.kind() == std::io::ErrorKind::WouldBlock => {
                pending.remove();
                Err(Error::Transfer(format!("submitting DMA commands: {why}")))
            }
            // The batch is queued and only a dead poller fails past that, which never runs it: the
            // caller may go, and the commands stay held until the ring closes.
            Err(why) => Err(Error::Transfer(format!("the DMA ring is broken: {why}"))),
        }
    }

    /// Takes every completion that has arrived and yields each transfer whose commands have all
    /// completed: its ticket and its result, the first failure when any command failed.
    pub(crate) fn complete(&mut self) -> impl Iterator<Item = (u64, Result<()>)> + '_ {
        for (ticket, result) in self.ring.completions() {
            let Some(pending) = self.pending.get_mut(&ticket) else {
                continue;
            };
            pending.remaining -= 1;
            if let Err(why) = result {
                pending
                    .failed
                    .get_or_insert_with(|| Error::Transfer(format!("DMA command failed: {why}")));
            }
        }
        self.pending
            .extract_if(.., |_, pending| pending.remaining == 0)
            .map(|(ticket, pending)| (ticket, pending.failed.map_or(Ok(()), Err)))
    }

    /// Writes `bytes` to `addr` on `chip` and waits for it. Only for the firmware image going in at
    /// bring-up, before any caller has a transfer in flight whose completion the wait could take.
    pub(crate) fn write(&mut self, chip: u8, addr: u64, bytes: &[u8]) -> Result<()> {
        assert!(self.pending.is_empty(), "a synchronous write needs the queue to itself");
        let ticket = self.submit(vec![Command {
            chip,
            direction: Direction::Writer,
            host: Host::Pageable,
            scatter: vec![Segment {
                src: bytes.as_ptr().addr() as u64,
                dst: addr,
                len: bytes.len() as u64,
            }],
        }])?;
        loop {
            if let Some((_, result)) = self.complete().find(|(done, _)| *done == ticket) {
                return result;
            }
            std::hint::spin_loop();
        }
    }
}

impl Channels {
    fn open(chip: u8) -> Result<Self> {
        let open = |direction| {
            (0..4)
                .map(|slot| Channel::open(chip, direction, slot))
                .collect::<Result<Vec<_>>>()
        };
        let writers = open(Direction::Writer)?;
        let readers = open(Direction::Reader)?;
        for channel in writers.iter().chain(&readers) {
            if channel.caps()? & CAPS_DMABUF == 0 {
                return Err(Error::Transfer(format!(
                    "PDMA channel {} lacks dma-buf support",
                    channel.fd()
                )));
            }
        }
        Ok(Self {
            writers,
            readers,
            writer: 0,
            reader: 0,
        })
    }

    fn next(&mut self, direction: Direction) -> &Channel {
        match direction {
            Direction::Writer => {
                let channel = &self.writers[self.writer];
                self.writer = (self.writer + 1) % self.writers.len();
                channel
            }
            Direction::Reader => {
                let channel = &self.readers[self.reader];
                self.reader = (self.reader + 1) % self.readers.len();
                channel
            }
        }
    }
}
