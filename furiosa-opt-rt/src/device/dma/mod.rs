//! The DMA thread's state: the device, its queue, and the loop that answers callers as their
//! transfers complete while taking new jobs.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::oneshot;

use crate::Error;

use super::engine::{Chip, Cluster, Images, Job, Launcher, Reserved};

mod channel;
mod queue;

pub(super) use channel::{Command, Direction, Host, Segment};
pub(super) use queue::DmaQueue;

type DeviceResult<T> = std::result::Result<T, Error>;

/// How long the thread keeps spinning for the next job after the last one finished.
const IDLE_SPIN: Duration = Duration::from_millis(1);

pub(super) struct Worker {
    reserved: Reserved,
    dma: DmaQueue,
    /// Who awaits each transfer in flight, by its ticket.
    replies: BTreeMap<u64, oneshot::Sender<DeviceResult<()>>>,
}

impl Worker {
    /// Opens the queue first: the device stages its firmware image through it.
    pub(super) fn open(clusters: Vec<Cluster>, images: Images, timeout: Duration) -> DeviceResult<(Self, Launcher)> {
        let mut dma = DmaQueue::open(clusters.iter().map(|cluster| cluster.chip()))?;
        let (reserved, launcher) = Reserved::open(clusters, images, timeout, &mut dma)?;
        Ok((
            Self {
                reserved,
                dma,
                replies: BTreeMap::new(),
            },
            launcher,
        ))
    }

    pub(super) fn free(&self) -> DeviceResult<std::ops::Range<usize>> {
        self.reserved.free()
    }

    pub(super) fn chips(&self) -> DeviceResult<Vec<Chip>> {
        self.reserved.chips()
    }

    /// Takes jobs and drives the transfers in flight until every sender has gone and nothing is
    /// left in flight. Spins while anything is in flight, and for a short while after going idle,
    /// so a caller issuing back-to-back transfers never pays a thread wake-up; only a longer idle
    /// parks the thread.
    pub(super) fn run(mut self, mut jobs: tokio::sync::mpsc::Receiver<Job>) {
        let mut idle_since = None;
        loop {
            let idle = self.replies.is_empty();
            let park = idle && idle_since.get_or_insert_with(Instant::now).elapsed() > IDLE_SPIN;
            let next = if park {
                match jobs.blocking_recv() {
                    Some(job) => Some(job),
                    None => return,
                }
            } else {
                match jobs.try_recv() {
                    Ok(job) => Some(job),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) if idle => return,
                    Err(TryRecvError::Disconnected) => None,
                }
            };
            let took = next.is_some();
            if let Some(job) = next {
                idle_since = None;
                job(&mut self);
            }
            for (ticket, result) in self.dma.complete() {
                if let Some(reply) = self.replies.remove(&ticket) {
                    let _ = reply.send(result);
                }
            }
            if !took {
                std::hint::spin_loop();
            }
        }
    }

    /// Submits every command of one transfer at once; `reply` is answered when all complete.
    pub(super) fn dma(&mut self, commands: Vec<Command>, reply: oneshot::Sender<DeviceResult<()>>) {
        match self.dma.submit(commands) {
            Ok(ticket) => {
                self.replies.insert(ticket, reply);
            }
            Err(why) => {
                let _ = reply.send(Err(why));
            }
        }
    }
}
