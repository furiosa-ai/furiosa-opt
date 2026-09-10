//! The driver's ring entries. Every submission and completion opens with one [`Entry`] word; the
//! frame that follows is this crate's own.

/// Driver-owned transport opcode.
pub const OPCODE: u8 = 1;

/// Submission ring capacity shared with the driver.
pub const SUBMISSION_RING: usize = 127;

/// Completion ring capacity shared with the driver.
pub const COMPLETION_RING: usize = 255;

/// Payload words in each completion.
pub const REPLY_WORDS: usize = 3;

/// Maximum payload words in one submission.
pub const MAX_SUBMISSION_WORDS: usize = 1008 / 8 - 1;

/// The header word of one ring entry: which request it belongs to and how many payload words
/// follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub id: u32,
    /// Payload words following the header.
    pub words: u8,
}

#[derive(thiserror_core::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Unsupported driver opcode.
    #[error("entry names opcode {0}, expected {OPCODE}")]
    Opcode(u8),
    /// Completion request ID mismatch.
    #[error("completion answers request {found}, expected {expected}")]
    Id { expected: u32, found: u32 },
    /// Invalid completion payload width.
    #[error("completion carries {0} words, expected {REPLY_WORDS}")]
    Words(u8),
    /// Submission exceeds [`MAX_SUBMISSION_WORDS`].
    #[error("frame spans {0} words, more than {MAX_SUBMISSION_WORDS}")]
    TooLong(usize),
}

impl Entry {
    // Driver layout: `{id, reserved, payload count, opcode}` in little-endian order.
    const WORDS_SHIFT: u32 = 48;
    const OPCODE_SHIFT: u32 = 56;

    pub fn new(id: u32, words: u8) -> Self {
        Self { id, words }
    }

    pub fn to_word(self) -> u64 {
        u64::from(self.id) | (u64::from(self.words) << Self::WORDS_SHIFT) | (u64::from(OPCODE) << Self::OPCODE_SHIFT)
    }

    pub fn from_word(word: u64) -> Result<Self, Error> {
        let opcode = (word >> Self::OPCODE_SHIFT) as u8;
        if opcode != OPCODE {
            return Err(Error::Opcode(opcode));
        }
        Ok(Self {
            id: word as u32,
            words: (word >> Self::WORDS_SHIFT) as u8,
        })
    }

    /// The header word opening a submission of `words` payload words.
    pub fn submission(id: u32, words: usize) -> Result<u64, Error> {
        if words > MAX_SUBMISSION_WORDS {
            return Err(Error::TooLong(words));
        }
        Ok(Self::new(id, words as u8).to_word())
    }

    /// Checks that `word` opens a completion answering `id`; [`REPLY_WORDS`] payload words follow.
    pub fn check_completion(word: u64, id: u32) -> Result<(), Error> {
        let entry = Self::from_word(word)?;
        if entry.id != id {
            return Err(Error::Id {
                expected: id,
                found: entry.id,
            });
        }
        if usize::from(entry.words) != REPLY_WORDS {
            return Err(Error::Words(entry.words));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transports_a_reply() {
        let entry = Entry::new(4, REPLY_WORDS as u8).to_word();
        assert_eq!(Entry::new(0x0102_0304, 3).to_word(), 0x0103_0000_0102_0304);
        assert_eq!(Entry::check_completion(entry, 4), Ok(()));
        assert_eq!(Entry::submission(9, 2), Ok(Entry::new(9, 2).to_word()));
    }

    #[test]
    fn rejects_malformed_entries() {
        for opcode in [0, 0x80, 0xff] {
            assert_eq!(
                Entry::from_word((u64::from(opcode) << Entry::OPCODE_SHIFT) | 7),
                Err(Error::Opcode(opcode))
            );
        }
        assert_eq!(
            Entry::submission(0, MAX_SUBMISSION_WORDS + 1),
            Err(Error::TooLong(MAX_SUBMISSION_WORDS + 1))
        );
        assert_eq!(
            Entry::check_completion(Entry::new(4, REPLY_WORDS as u8).to_word(), 5),
            Err(Error::Id { expected: 5, found: 4 })
        );
        assert_eq!(
            Entry::check_completion(Entry::new(4, 2).to_word(), 4),
            Err(Error::Words(2))
        );
    }
}
