use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use crate::{Error, Result};
use furiosa_opt_abi::bootloader;
use furiosa_opt_ipc::entry::Entry;
use furiosa_opt_ipc::{Identity, LAUNCHED_WORDS, MAX_RESPONSE_WORDS, MAX_SUBMISSION_WORDS, Message, Request, Response};
use nix::errno::Errno;

use super::mmio::{BAR2_LEN, CQ_LEN, CQ_REGION, DOORBELL, HEAD0, Q_BASE, Q_STRIDE, TAIL0};
use super::mmio::{Mapping, node, open_rw};
use super::{DRAM_BASE, PeRange, ring};

const DRAM_FIRMWARE: u64 = 512 << 20;
const DRAM_LEN: u64 = (48 << 30) - DRAM_FIRMWARE;
const WINDOW_DRAM: u8 = 0;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct PeFw {
    pe_ids: u32,
    size: u32,
    data: u64,
}

nix::ioctl_readwrite!(copy_pe_fw, b'N', 0x04, PeFw);

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Window {
    kind: u8,
    pes: u8,
    base: u64,
    size: u64,
}

nix::ioctl_readwrite!(set_window, b'N', 0x03, Window);

pub(crate) struct Link {
    submission: ring::Submission,
    completion: ring::Completion,
    _bar2: Mapping,
    _page: Mapping,
    _cluster: File,
    next: u32,
    answers: [Answer; IN_FLIGHT],
    /// How long the device may leave a request unanswered before it fails.
    timeout: Duration,
}

/// Requests one link keeps in flight at most; a submitter past this many waits for an answer
/// to be collected.
pub(crate) const IN_FLIGHT: usize = 16;

/// Where the completions answering one request accumulate. Each launch keeps its words, so a
/// launch answer without profile records never grows them after the first.
struct Answer {
    id: Option<u32>,
    words: Vec<u64>,
}

impl Answer {
    /// The words of the complete frame, once every completion carrying it has arrived; the last
    /// completion pads the frame out to its width.
    fn frame(&self) -> Result<Option<&[u64]>> {
        let Some(&first) = self.words.first() else {
            return Ok(None);
        };
        let words = furiosa_opt_ipc::Header::read(first)
            .map_err(|why| Error::Device(format!("the device's answer to {:?} is unreadable: {why}", self.id)))?
            .words as usize;
        if words > MAX_RESPONSE_WORDS {
            return Err(Error::Device(format!(
                "the device's answer to {:?} spans {words} words",
                self.id
            )));
        }
        Ok(self.words.get(..words))
    }
}

impl Link {
    /// Takes the cluster's PEs, loads the signed bootloader and opens the queues; `boot` brings up
    /// the firmware next. A request unanswered for `timeout` fails; held PEs give `Reserved`.
    pub(super) fn open(identity: &Identity, pes: PeRange, boot: &[u8], timeout: Duration) -> Result<Self> {
        let chip = identity.chip();
        // Staging refuses PEs in use, so it goes first; the memory window must be set before the
        // node opens, which starts the PEs on the staged bootloader.
        Self::stage(chip, pes, boot)?;
        Self::map_dram(chip, pes)?;
        let cluster_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(node(chip, &pes.node()))
            .map_err(|why| Error::reserving(chip, Errno::from_raw(why.raw_os_error().unwrap_or(0))))?;
        let bars = open_rw(&node(chip, "bar2"))?;
        let queue = pes.queue(identity.rank());
        let bar2 = Mapping::new(&bars, (pes.window() + Q_BASE) as i64, BAR2_LEN)?;
        let page = Mapping::new(&cluster_file, CQ_REGION, CQ_LEN).map_err(|why| {
            Error::Device(format!(
                "{why}; the driver must offer the completion-queue mapping (NPU_PE_MMAP_REGION_CQ)"
            ))
        })?;
        let (submission, completion) = Self::queues(&bar2, &page, queue)?;
        Ok(Self {
            submission,
            completion,
            _bar2: bar2,
            _page: page,
            _cluster: cluster_file,
            next: 1,
            answers: std::array::from_fn(|_| Answer {
                id: None,
                words: Vec::with_capacity(LAUNCHED_WORDS),
            }),
            timeout,
        })
    }

    fn map_dram(chip: u8, pes: PeRange) -> Result<()> {
        let mgmt = open_rw(&node(chip, "mgmt"))?;
        let mut window = Window {
            kind: WINDOW_DRAM,
            pes: pes.mask(),
            base: DRAM_BASE + DRAM_FIRMWARE,
            size: DRAM_LEN,
        };
        // SAFETY: the request reads `window` and writes back only its own result.
        unsafe { set_window(mgmt.as_raw_fd(), &mut window) }
            .map(|_| ())
            .map_err(|why| Error::Device(format!("npu{chip} {} addressing memory: {why}", pes.node())))
    }

    /// Hands the signed bootloader to the driver, which verifies it and starts the cluster's PEs
    /// on it once their node opens.
    fn stage(chip: u8, pes: PeRange, image: &[u8]) -> Result<()> {
        let size = u32::try_from(image.len()).map_err(|_| Error::Firmware("bootloader exceeds 4 GiB".into()))?;
        let channel = open_rw(&node(chip, "ch0"))?;
        let mut desc = PeFw {
            pe_ids: u32::from(pes.mask()),
            size,
            data: image.as_ptr() as u64,
        };
        // SAFETY: `desc` names this process's own buffer for the length it states.
        unsafe { copy_pe_fw(channel.as_raw_fd(), &mut desc) }
            .map(|_| ())
            .map_err(|why| Error::reserving(chip, why))
    }

    fn queues(bar2: &Mapping, page: &Mapping, queue: u64) -> Result<(ring::Submission, ring::Completion)> {
        let base = bar2.at.as_ptr();
        let doorbell = (DOORBELL - Q_BASE) as usize;
        // SAFETY: offsets are within the mappings and the boot initialized the addressed queue.
        let (submission, completion) = unsafe {
            (
                ring::Submission::new(
                    base.add((Q_STRIDE * queue) as usize).cast(),
                    base.add(doorbell + (TAIL0 + 4 * queue) as usize).cast(),
                    base.add(doorbell + (HEAD0 + 4 * queue) as usize).cast(),
                ),
                ring::Completion::new(page.at.as_ptr()),
            )
        };
        let why = |why| Error::Device(format!("the cluster's rings are not in a usable state: {why}"));
        Ok((submission.map_err(why)?, completion.map_err(why)?))
    }

    /// Asks the bootloader to bring up the firmware image staged at `addr`, `len` bytes under
    /// `token`, and reports its answer. Speaks the boot contract, not the runtime's protocol:
    /// nothing else has run yet.
    pub(crate) fn boot(&mut self, addr: u64, len: u64, token: u64) -> Result<bootloader::Code> {
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        let frame = [
            bootloader::header::pack(id, bootloader::REQUEST_WORDS as u8),
            addr,
            len,
            token,
        ];
        self.submission
            .submit(&frame)
            .map_err(|why| Error::Device(format!("submitting the boot request: {why}")))?;
        let deadline = Instant::now() + self.timeout;
        loop {
            match self.completion.take() {
                Some(Ok((answered, words))) if answered == id => {
                    return bootloader::Code::from_word(words[0])
                        .ok_or_else(|| Error::Firmware(format!("the bootloader answered with {}", words[0])));
                }
                Some(Ok(_)) => {}
                Some(Err(why)) => return Err(Error::Firmware(format!("the bootloader's answer is unreadable: {why}"))),
                None if Instant::now() >= deadline => {
                    return Err(Error::Firmware(format!(
                        "the bootloader left the request unanswered for {}s",
                        self.timeout.as_secs()
                    )));
                }
                None => std::hint::spin_loop(),
            }
        }
    }

    /// Submits `request` and returns its id. Waits, until `deadline`, for room in the queue and
    /// for a free answer slot; the submission is committed once the queue's tail moves. Only a
    /// full queue is waited on: a ring in an unusable state fails at once.
    pub(crate) fn submit(&mut self, request: &Request<'_>, deadline: Instant) -> Result<u32> {
        let id = self.next;
        let mut frame = [0; 1 + MAX_SUBMISSION_WORDS];
        let words = request
            .write(&mut frame[1..])
            .map_err(|why| Error::Device(format!("encoding request {id}: {why}")))?;
        frame[0] =
            Entry::submission(id, words).map_err(|why| Error::Device(format!("submitting request {id}: {why}")))?;
        let frame = &frame[..1 + words];
        let slot = id as usize % IN_FLIGHT;
        loop {
            if self.answers[slot].id.is_none() {
                match self.submission.submit(frame) {
                    Ok(()) => break,
                    Err(ring::Error::Full { .. }) => {}
                    Err(why) => return Err(Error::Device(format!("submitting request {id}: {why}"))),
                }
            }
            if Instant::now() >= deadline {
                return Err(Error::Device(format!(
                    "the device left the queue full for {}s",
                    self.timeout.as_secs()
                )));
            }
            self.collect()?;
            std::hint::spin_loop();
        }
        self.answers[slot].id = Some(id);
        self.next = self.next.wrapping_add(1);
        Ok(id)
    }

    /// The answer to `id` once it is complete. Completions arrive in request order; those for
    /// other requests wait in their slots for their own callers.
    pub(crate) fn poll(&mut self, id: u32) -> Result<Option<Response>> {
        self.collect()?;
        let answer = &mut self.answers[id as usize % IN_FLIGHT];
        if answer.id != Some(id) {
            return Ok(None);
        }
        let Some(frame) = answer.frame()? else {
            return Ok(None);
        };
        let response = Response::read(frame)
            .map_err(|why| Error::Device(format!("the device's answer to {id} is unreadable: {why}")));
        answer.id = None;
        answer.words.clear();
        response.map(Some)
    }

    /// Gives up on `id`: collects its answer if the device sends one before `deadline`, and frees
    /// its slot either way, so a failed launch does not hold the slot forever.
    pub(crate) fn abandon(&mut self, id: u32, deadline: Instant) {
        while Instant::now() < deadline {
            match self.poll(id) {
                Ok(None) => std::hint::spin_loop(),
                Ok(Some(_)) | Err(_) => break,
            }
        }
        let answer = &mut self.answers[id as usize % IN_FLIGHT];
        if answer.id == Some(id) {
            answer.id = None;
            answer.words.clear();
        }
    }

    /// Takes every completion the device has published into the slot of the request it answers.
    fn collect(&mut self) -> Result<()> {
        while let Some(entry) = self.completion.take() {
            let (id, words) = entry.map_err(|why| Error::Device(format!("a completion is unreadable: {why}")))?;
            let answer = &mut self.answers[id as usize % IN_FLIGHT];
            if answer.id != Some(id) || answer.frame()?.is_some() {
                // An answer to a request nobody awaits any more; `submit` gave it up on timeout.
                continue;
            }
            answer.words.extend_from_slice(&words);
        }
        Ok(())
    }

    /// Submits `request` and spins for its answer.
    pub(crate) fn call(&mut self, request: &Request<'_>) -> Result<Response> {
        let deadline = Instant::now() + self.timeout;
        let id = self.submit(request, deadline)?;
        loop {
            if let Some(response) = self.poll(id)? {
                return Ok(response);
            }
            if Instant::now() >= deadline {
                return Err(Error::Device(format!(
                    "the device left request {id} unanswered for {}s",
                    self.timeout.as_secs(),
                )));
            }
            std::hint::spin_loop();
        }
    }
}
