use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

use crate::Error;
use crate::buffer::View;
use crate::device::ChipRank;
use furiosa_opt_ipc::ProfileRecord;

use super::dma::{Command, Direction, Host, Segment, Worker};

type DeviceResult<T> = std::result::Result<T, Error>;

mod dram;
mod link;
mod mmio;
mod reserved;
mod ring;
use mmio::Mapping;
pub(crate) use mmio::PeRange;
pub(super) use mmio::{node, open_rw};
pub(crate) use reserved::{Cluster, Images, Launch, Launcher, Reserved, Ticket};

/// Where device memory starts in a cluster's address space, and in the PDMA's.
pub(super) const DRAM_BASE: u64 = furiosa_opt_abi::DEVICE_MEMORY.start;

/// Work the DMA thread runs against the chips.
pub(super) type Job = Box<dyn FnOnce(&mut Worker) + Send>;

/// Where a job reports its result, now or when the device finishes.
type Reply<T> = oneshot::Sender<DeviceResult<T>>;

/// Queued work waiting for the DMA thread; enough for every caller a process can reasonably
/// keep in flight before a launch or transfer must finish.
const QUEUE_DEPTH: usize = 64;

/// How long a caller spins on an unanswered launch or transfer before yielding its task: a
/// device usually answers within tens of microseconds, and a wake-up would cost more.
const SPIN: Duration = Duration::from_micros(200);

/// Writes up to this many bytes post through the memory window instead of a DMA. A DMA costs
/// about 25 µs before it moves a byte; posting costs about 3 µs up to here, then the uncached
/// mapping's word-sized writes throttle to PCIe (4 KiB takes as long as the DMA).
const MMIO_WRITE_LIMIT: usize = 2 << 10;

/// Where device memory starts in the chip's fourth BAR.
const WINDOW_DRAM: u64 = 256 << 20;

const PAGE: usize = 4096;

/// One chip's memory as the PDMA addresses it: `base` is where the device's window starts.
#[derive(Clone, Copy)]
pub(crate) struct Chip {
    pub(crate) index: u8,
    pub(crate) base: u64,
}

/// A device from the host's side. Launches go straight from the calling thread to the
/// clusters' queues, one caller at a time, and are answered through completion rings the
/// caller polls itself. Transfers complete through one io_uring, so a dedicated thread drives
/// those; callers queue transfers there and await the result, waiting for room when the queue
/// is full.
pub(crate) struct Engine {
    allocator: Arc<Mutex<crate::buffer::Allocator>>,
    /// In dense order: a view's chip index is an index here.
    chips: Vec<Chip>,
    /// Each chip's memory as the host can map it, in the same order.
    windows: Vec<std::fs::File>,
    launcher: Mutex<Launcher>,
    sender: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl Engine {
    pub(super) fn open(
        clusters: Vec<Cluster>,
        images: Images,
        allocator: Arc<Mutex<crate::buffer::Allocator>>,
        timeout: Duration,
    ) -> DeviceResult<(Self, Range<usize>)> {
        let (worker, launcher) = Worker::open(clusters, images, timeout)?;
        let free = worker.free()?;
        let chips = worker.chips()?;
        let windows = chips
            .iter()
            .map(|chip| open_rw(&node(chip.index, "bar4")))
            .collect::<DeviceResult<Vec<_>>>()?;
        let (sender, jobs) = mpsc::channel(QUEUE_DEPTH);
        let thread = std::thread::spawn(move || worker.run(jobs));
        Ok((
            Self {
                allocator,
                chips,
                windows,
                launcher: Mutex::new(launcher),
                sender: Some(sender),
                thread: Some(thread),
            },
            free,
        ))
    }

    /// Chips in the device.
    pub(super) fn chips(&self) -> usize {
        self.chips.len()
    }

    /// Submits one launch to every cluster and returns its ticket. Launches run in submission
    /// order; transfers proceed alongside.
    pub(crate) fn submit(&self, launch: &Launch<'_>) -> DeviceResult<Ticket> {
        self.launcher
            .lock()
            .map_err(|_| Error::Device("the launch path is poisoned".into()))?
            .submit(launch)
    }

    /// Collects `ticket`'s answer without awaiting, for a launch dropped unwaited. A poisoned
    /// launch path or a device error ends the wait: neither leaves an answer to collect.
    pub(crate) fn drain(&self, ticket: &mut Ticket) {
        while let Ok(Ok(false)) = self.launcher.lock().map(|mut launcher| launcher.poll(ticket)) {
            std::hint::spin_loop();
        }
    }

    /// Waits for every cluster to answer `ticket` and returns their profile records: one entry per
    /// cluster when the launch was profiled, none otherwise. The ticket stays the caller's, so a
    /// wait dropped part-way can still be drained.
    pub(crate) async fn wait(&self, ticket: &mut Ticket) -> DeviceResult<Vec<Vec<ProfileRecord>>> {
        let started = Instant::now();
        loop {
            let done = self
                .launcher
                .lock()
                .map_err(|_| Error::Device("the launch path is poisoned".into()))?
                .poll(ticket)?;
            if done {
                return Ok(ticket.records());
            }
            if started.elapsed() < SPIN {
                std::hint::spin_loop();
            } else {
                tokio::task::yield_now().await;
            }
        }
    }

    /// Copies each host slice into its view. Every pair moves at once; the call returns when all
    /// have landed. Panics when a slice is not its view's size or a buffer is another runtime's.
    ///
    /// A write of up to [`MMIO_WRITE_LIMIT`] bytes posts straight through the device's memory
    /// window from this thread instead of taking the DMA path.
    pub(super) async fn write(&self, pairs: impl IntoIterator<Item = (&[u8], View)>) -> DeviceResult<()> {
        let pairs: Vec<(&[u8], View)> = pairs.into_iter().collect();
        if pairs.iter().map(|(bytes, _)| bytes.len()).sum::<usize>() <= MMIO_WRITE_LIMIT {
            return pairs.into_iter().try_for_each(|(bytes, view)| self.post(bytes, &view));
        }
        let commands = self.commands(
            Direction::Writer,
            pairs
                .into_iter()
                .map(|(bytes, view)| (view, bytes.as_ptr_range().start.addr()..bytes.as_ptr_range().end.addr())),
        )?;
        self.transfer(commands).await
    }

    /// Copies each view into its host slice, with the same terms as [`Self::write`].
    pub(super) async fn read(&self, pairs: impl IntoIterator<Item = (View, &mut [u8])>) -> DeviceResult<()> {
        let commands = self.commands(
            Direction::Reader,
            pairs
                .into_iter()
                .map(|(view, bytes)| (view, bytes.as_ptr_range().start.addr()..bytes.as_ptr_range().end.addr())),
        )?;
        self.transfer(commands).await
    }

    /// Writes `bytes` into `view` through the memory window, one chip at a time.
    fn post(&self, bytes: &[u8], view: &View) -> DeviceResult<()> {
        let chips = self.pieces(view, bytes.len())?;
        let size = view.buffer.size();
        for (piece, index) in chips.enumerate() {
            let offset = WINDOW_DRAM + view.buffer.addr() as u64;
            let page = offset & !(PAGE as u64 - 1);
            Mapping::new(&self.windows[index], page as i64, (offset - page) as usize + size)?
                .store((offset - page) as usize, &bytes[piece * size..(piece + 1) * size]);
        }
        Ok(())
    }

    /// The chips `view` covers, as indices into the device. Another device's buffer or rank is
    /// refused; `host` bytes of the wrong size panic, as a wrong-length `copy_from_slice` would.
    fn pieces(&self, view: &View, host: usize) -> DeviceResult<Range<usize>> {
        if !view.buffer.belongs_to(&self.allocator) {
            return Err(Error::ForeignBuffer);
        }
        let chips = match view.chip {
            Some(ChipRank(rank)) if rank < self.chips.len() => rank..rank + 1,
            Some(_) => return Err(Error::ForeignBuffer),
            None => 0..self.chips.len(),
        };
        assert_eq!(
            host,
            view.buffer.size() * chips.len(),
            "{host} host bytes do not match a {} byte buffer on {} chips",
            view.buffer.size(),
            chips.len()
        );
        Ok(chips)
    }

    async fn transfer(&self, commands: Vec<Command>) -> DeviceResult<()> {
        self.enqueue(move |worker, reply| worker.dma(commands, reply))
            .await?
            .await
    }

    /// One transfer's segments gathered by channel: every segment between one chip and one kind
    /// of host memory goes in one command, and a view over every chip takes one buffer per chip.
    fn commands(
        &self,
        direction: Direction,
        pairs: impl IntoIterator<Item = (View, Range<usize>)>,
    ) -> DeviceResult<Vec<Command>> {
        let mut segments: BTreeMap<(u8, Host), Vec<Segment>> = BTreeMap::new();
        for (view, host) in pairs {
            let chips = self.pieces(&view, host.len())?;
            let size = view.buffer.size();
            for (piece, index) in chips.enumerate() {
                let chip = self.chips[index];
                let host = host.start + piece * size..host.start + (piece + 1) * size;
                let (kind, host) = match crate::pinned::backing(host.clone()) {
                    Some(backing) => (Host::Pinned(backing.fd), backing.offset as u64),
                    None => (Host::Pageable, host.start as u64),
                };
                let address = chip.base + view.buffer.addr() as u64;
                let (src, dst) = match direction {
                    Direction::Writer => (host, address),
                    Direction::Reader => (address, host),
                };
                segments.entry((chip.index, kind)).or_default().push(Segment {
                    src,
                    dst,
                    len: size as u64,
                });
            }
        }
        Ok(segments
            .into_iter()
            .map(|((chip, host), scatter)| Command {
                chip,
                direction,
                host,
                scatter,
            })
            .collect())
    }

    /// Queues `job` for the DMA thread with the reply it answers through. Waits for queue
    /// room, so a caller that outruns the chips slows down instead of failing; the returned
    /// transfer resolves when the job answers.
    async fn enqueue<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut Worker, Reply<T>) + Send + 'static,
    ) -> DeviceResult<Transfer<T>> {
        let (reply, receiver) = oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Device("the device thread has shut down".into()))?
            .send(Box::new(move |worker| job(worker, reply)))
            .await
            .map_err(|_| Error::Device("the device thread stopped".into()))?;
        Ok(Transfer(receiver))
    }
}

/// The caller's handle to one transfer in flight, resolving when the DMA thread answers. Dropping
/// it early waits anyway: the worker keeps using the host memory the caller lent until then.
pub(crate) struct Transfer<T>(oneshot::Receiver<DeviceResult<T>>);

impl<T> Drop for Transfer<T> {
    fn drop(&mut self) {
        while let Err(oneshot::error::TryRecvError::Empty) = self.0.try_recv() {
            std::thread::yield_now();
        }
    }
}

impl<T> std::future::Future for Transfer<T> {
    type Output = DeviceResult<T>;

    /// Spins for [`SPIN`] before parking, so a transfer the device finishes at once does not pay a
    /// cross-thread wake-up on top.
    fn poll(mut self: std::pin::Pin<&mut Self>, context: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        let stopped = || Error::Device("the device thread stopped before answering".into());
        let started = Instant::now();
        loop {
            match self.0.try_recv() {
                Ok(received) => return std::task::Poll::Ready(received),
                Err(oneshot::error::TryRecvError::Closed) => return std::task::Poll::Ready(Err(stopped())),
                Err(oneshot::error::TryRecvError::Empty) if started.elapsed() < SPIN => std::hint::spin_loop(),
                Err(oneshot::error::TryRecvError::Empty) => {
                    return std::pin::Pin::new(&mut self.0)
                        .poll(context)
                        .map(|received| received.map_err(|_| stopped())?);
                }
            }
        }
    }
}

#[cfg(test)]
impl Engine {
    /// An engine over `chips` with no hardware behind it, for what never reaches a chip.
    pub(crate) fn stub(chips: &[u8], allocator: Arc<Mutex<crate::buffer::Allocator>>) -> Self {
        Self {
            allocator,
            chips: chips.iter().map(|&index| Chip { index, base: 0 }).collect(),
            windows: Vec::new(),
            launcher: Mutex::default(),
            sender: None,
            thread: None,
        }
    }
}

impl Drop for Engine {
    /// Closing the queue lets the thread finish every job already queued before it exits.
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use super::*;
    use crate::buffer::{Allocator, Buffer};

    fn addresses(bytes: &[u8]) -> Range<usize> {
        bytes.as_ptr_range().start.addr()..bytes.as_ptr_range().end.addr()
    }

    #[test]
    fn dropping_drains_queued_work_first() {
        let (sender, mut jobs) = mpsc::channel::<Job>(1);
        let (done, complete) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            while jobs.blocking_recv().is_some() {
                let _ = done.send(());
            }
        });
        let (mut engine, _) = engine(&[]);
        engine.sender = Some(sender);
        engine.thread = Some(thread);

        engine
            .sender
            .as_ref()
            .expect("worker sender")
            .try_send(Box::new(|_| {}))
            .expect("queued work");
        drop(engine);

        complete
            .recv()
            .expect("the worker drained the job before drop returned");
    }

    fn engine(chips: &[u8]) -> (Engine, Buffer) {
        let allocator = Arc::new(Mutex::new(Allocator::new(0..256)));
        let buffer = Buffer::alloc(&allocator, 16).expect("device buffer");
        (Engine::stub(chips, allocator), buffer)
    }

    #[test]
    fn two_clusters_one_command() {
        let (engine, buffer) = engine(&[1]);
        let host = [0u8; 16];

        let commands = engine
            .commands(Direction::Writer, [(buffer.on_all(), addresses(&host))])
            .expect("commands");

        assert_eq!(engine.chips(), 1);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].chip, 1);
        assert_eq!(commands[0].host, Host::Pageable);
        assert_eq!(commands[0].scatter[0].src, host.as_ptr().addr() as u64);
    }

    #[test]
    fn all_chip_view_orders_buffers() {
        let (engine, buffer) = engine(&[1, 2]);
        let host = [0u8; 32];

        let commands = engine
            .commands(Direction::Reader, [(buffer.on_all(), addresses(&host))])
            .expect("commands");

        assert_eq!(commands.iter().map(|command| command.chip).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(commands[1].scatter[0].dst, host.as_ptr().addr() as u64 + 16);
        assert_eq!(commands[1].scatter[0].src, buffer.addr() as u64);
    }

    #[test]
    fn pairs_to_one_chip_share_one_command() {
        let (engine, buffer) = engine(&[1, 2]);
        let host = [0u8; 16];

        let commands = engine
            .commands(
                Direction::Writer,
                [
                    (buffer.on(ChipRank(1)), addresses(&host)),
                    (buffer.on(ChipRank(1)), addresses(&host)),
                ],
            )
            .expect("commands");

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].chip, 2);
        assert_eq!(commands[0].scatter.len(), 2);
    }

    #[test]
    fn a_host_slice_of_the_wrong_size_panics() {
        let (engine, buffer) = engine(&[1, 2]);
        let host = [0u8; 16];

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                engine.commands(Direction::Writer, [(buffer.on_all(), addresses(&host))])
            }))
            .is_err()
        );
    }

    #[test]
    fn rejects_foreign_rank() {
        let (engine, buffer) = engine(&[3]);
        let host = [0u8; 16];

        assert!(matches!(
            engine.commands(Direction::Writer, [(buffer.on(ChipRank(1)), addresses(&host))]),
            Err(Error::ForeignBuffer)
        ));
        assert_eq!(
            engine
                .commands(Direction::Writer, [(buffer.on(ChipRank(0)), addresses(&host))])
                .expect("commands")[0]
                .chip,
            3
        );
    }

    #[test]
    fn rejects_foreign_buffer() {
        let (engine, _) = engine(&[1]);
        let other = Arc::new(Mutex::new(Allocator::new(0..256)));
        let foreign = Buffer::alloc(&other, 16).expect("foreign buffer");
        let host = [0u8; 16];

        assert!(matches!(
            engine.commands(Direction::Writer, [(foreign.on_all(), addresses(&host))]),
            Err(Error::ForeignBuffer)
        ));
    }
}
