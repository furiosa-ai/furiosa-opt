//! One cluster's two rings as the host sees them through its mappings: the submission ring it
//! produces into and the completion ring it consumes, each one side of a
//! [`furiosa_opt_abi::ring::Ring`]. Every register read is a PCIe round trip, so a side that
//! remembers its own index and the space it last saw is what keeps a launch to one write.

use std::task::Poll;

use furiosa_opt_abi::reg::Reg;
use furiosa_opt_abi::ring::{Consumer, Producer, Ring};
use furiosa_opt_ipc::entry::Entry;
use furiosa_opt_ipc::{COMPLETION_RING, REPLY_WORDS, SUBMISSION_RING, TransportError};

pub struct Submission(Producer);

impl Submission {
    /// # Safety
    ///
    /// The pointers must map one PE queue and its tail and head registers.
    pub unsafe fn new(entries: *mut u64, tail: *mut u32, head: *mut u32) -> Result<Self, Error> {
        // SAFETY: the caller's.
        let ring = unsafe { Ring::new(Reg::new(entries), Reg::new(tail), Reg::new(head), SUBMISSION_RING) };
        Ok(Self(Producer::new(ring)?))
    }

    /// Writes `words` and moves the tail, or reports the room the ring had when they did not fit;
    /// `Ok` means the device can see them.
    pub fn submit(&mut self, words: &[u64]) -> Result<(), Error> {
        match self.0.write_all(words) {
            Poll::Ready(result) => Ok(result?),
            Poll::Pending => Err(Error::Full {
                need: words.len(),
                free: self.0.space()?,
            }),
        }
    }
}

pub struct Completion(Consumer);

impl Completion {
    /// # Safety
    ///
    /// `page` must map one cluster completion page.
    pub unsafe fn new(page: *mut u8) -> Result<Self, Error> {
        // SAFETY: the caller's; the page opens with the tail and head registers, then the entries.
        let ring = unsafe {
            Ring::new(
                Reg::new(page.add(MEM).cast()),
                Reg::new(page.add(TAIL).cast()),
                Reg::new(page.add(HEAD).cast()),
                COMPLETION_RING,
            )
        };
        Ok(Self(Consumer::new(ring)?))
    }

    /// The next completion: the id it answers and its payload words.
    pub fn take(&mut self) -> Option<Result<(u32, [u64; REPLY_WORDS]), Error>> {
        let mut words = [0u64; REPLY_WORDS + 1];
        match self.0.copy(0, &mut words) {
            Ok(()) => {}
            Err(furiosa_opt_abi::ring::Error::Words) => return None,
            Err(error) => return Some(Err(error.into())),
        }
        if let Err(error) = self.0.consume(words.len()) {
            return Some(Err(error.into()));
        }
        Some(
            Entry::from_word(words[0])
                .and_then(|header| {
                    (usize::from(header.words) == REPLY_WORDS)
                        .then_some(header.id)
                        .ok_or(TransportError::Words(header.words))
                })
                .map(|id| (id, std::array::from_fn(|index| words[1 + index])))
                .map_err(Error::Frame),
        )
    }
}

const TAIL: usize = 0;
const HEAD: usize = 4;
const MEM: usize = 8;

#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    #[error(transparent)]
    Frame(#[from] TransportError),
    #[error("the queue has room for {free} words, not {need}")]
    Full { need: usize, free: usize },
    #[error("a ring register holds an index past the ring")]
    Index,
    #[error("a ring was asked for more words than it holds")]
    Words,
}

impl From<furiosa_opt_abi::ring::Error> for Error {
    fn from(error: furiosa_opt_abi::ring::Error) -> Self {
        match error {
            furiosa_opt_abi::ring::Error::Index => Self::Index,
            furiosa_opt_abi::ring::Error::Words => Self::Words,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::cell::UnsafeCell;

    use super::*;

    #[test]
    fn full_ring_reports_free_space() {
        let entries = UnsafeCell::new([0u64; SUBMISSION_RING]);
        let tail = UnsafeCell::new(0);
        let head = UnsafeCell::new(1);
        // SAFETY: the test keeps the mapped ring and registers alive for `submission`.
        let mut submission = unsafe { Submission::new(entries.get().cast(), tail.get(), head.get()) }.unwrap();

        assert_eq!(submission.submit(&[7]), Err(Error::Full { need: 1, free: 0 }));
        assert_eq!(unsafe { *tail.get() }, 0);

        // The device consumes; the retried submission writes the words and moves the tail.
        unsafe { *head.get() = 2 };
        assert_eq!(submission.submit(&[7]), Ok(()));
        assert_eq!(unsafe { (*tail.get(), (*entries.get())[0]) }, (1, 7));
    }

    #[test]
    fn broken_index_beats_full() {
        let entries = UnsafeCell::new([0u64; SUBMISSION_RING]);
        let tail = UnsafeCell::new(0);
        let head = UnsafeCell::new(0);
        // SAFETY: as above.
        let mut submission = unsafe { Submission::new(entries.get().cast(), tail.get(), head.get()) }.unwrap();
        assert_eq!(submission.submit(&[1; SUBMISSION_RING - 1]), Ok(()));

        unsafe { *head.get() = SUBMISSION_RING as u32 };

        assert_eq!(submission.submit(&[7]), Err(Error::Index));
    }

    #[test]
    fn a_completion_ring_hands_out_whole_entries() {
        let mut page = vec![0u8; MEM + COMPLETION_RING * 8];
        let entries = page[MEM..].as_mut_ptr().cast::<u64>();
        // SAFETY: the test owns the page.
        unsafe {
            entries.write(Entry::new(9, REPLY_WORDS as u8).to_word());
            entries.add(1).write(1);
            entries.add(2).write(2);
            entries.add(3).write(3);
            page[TAIL..TAIL + 4].copy_from_slice(&4u32.to_ne_bytes());
        }
        // SAFETY: as above.
        let mut completion = unsafe { Completion::new(page.as_mut_ptr()) }.unwrap();
        assert_eq!(completion.take(), Some(Ok((9, [1, 2, 3]))));
        assert_eq!(completion.take(), None);
        assert_eq!(u32::from_ne_bytes(page[HEAD..HEAD + 4].try_into().unwrap()), 4);
    }
}
