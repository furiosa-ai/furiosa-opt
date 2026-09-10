//! An io_uring that carries driver commands and nothing else: a command is a descriptor, an
//! opcode, sixteen payload bytes and a tag; a completion is the tag and the driver's result. What
//! the opcodes and payloads mean is the caller's business.

use std::os::fd::RawFd;

use io_uring::{IoUring, opcode::UringCmd16, types::Fd};

/// One `uring_cmd` for the driver behind `fd`, answered under `tag`.
pub(super) struct Cmd {
    pub(super) fd: RawFd,
    pub(super) opcode: u32,
    pub(super) payload: [u8; 16],
    pub(super) tag: u64,
}

/// How long the kernel's submission poller keeps spinning after the last command.
const SQPOLL_IDLE: std::time::Duration = std::time::Duration::from_millis(100);

pub(super) struct Ring(IoUring);

impl Ring {
    /// A ring that holds `depth` commands in flight at most. The kernel polls its submission
    /// queue, so handing it a command costs no system call while transfers keep coming; the
    /// poller sleeps after [`SQPOLL_IDLE`] of quiet.
    pub(super) fn open(depth: u32) -> std::io::Result<Self> {
        IoUring::builder()
            .setup_sqpoll(SQPOLL_IDLE.as_millis() as u32)
            .build(depth)
            .map(Self)
    }

    /// Queues every command for the kernel's poller without waiting: `WouldBlock` with nothing
    /// queued when the batch does not fit; any other error means the ring itself is broken.
    pub(super) fn submit(&mut self, cmds: impl ExactSizeIterator<Item = Cmd>) -> std::io::Result<()> {
        let mut queue = self.0.submission();
        if queue.len() + cmds.len() > queue.capacity() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!(
                    "{} commands do not fit the ring's {} free slots",
                    cmds.len(),
                    queue.capacity() - queue.len()
                ),
            ));
        }
        for cmd in cmds {
            let entry = UringCmd16::new(Fd(cmd.fd), cmd.opcode)
                .cmd(cmd.payload)
                .build()
                .user_data(cmd.tag);
            // SAFETY: a `uring_cmd` entry owns its sixteen payload bytes; nothing else is borrowed.
            unsafe { queue.push(&entry) }.expect("room was checked");
        }
        drop(queue);
        // The poller takes whatever the queue holds; this call only wakes it when it went idle,
        // and the count it returns is the queue's length then, not a batch to wait for.
        self.0.submit().map(drop)
    }

    /// Every completion that has arrived: its tag and the driver's result, an `errno` when
    /// negative.
    pub(super) fn completions(&mut self) -> impl Iterator<Item = (u64, std::io::Result<i32>)> + '_ {
        self.0.completion().map(|entry| {
            let result = entry.result();
            let result = if result < 0 {
                Err(std::io::Error::from_raw_os_error(-result))
            } else {
                Ok(result)
            };
            (entry.user_data(), result)
        })
    }
}
