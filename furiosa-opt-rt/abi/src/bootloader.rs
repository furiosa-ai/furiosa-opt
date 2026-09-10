//! The contract with the signed bootloader. Signed once, it answers one request, so nothing here
//! may change without a re-signing.
//!
//! - The request is a driver queue entry: a header word, then the firmware image's `addr`, `len`
//!   and a `token` the host chose for this staging.
//! - The token masks [`IMAGE_MAGIC`], so a stale image at a reused address is never booted.
//! - Landing copies and authenticates the same bytes; only the bound digest authorizes entry.
//! - The answer is one completion whose first payload word is a [`Code`].

/// Where a firmware image runs in the scratchpad, above the bootloader.
pub const IMAGE_BASE: usize = 0x1_0000;
/// The most bytes a firmware image may span.
pub const IMAGE_LIMIT: usize = 0xe_b000 - IMAGE_BASE;
/// The first word of every firmware image as built, `RESIDENT` in ASCII; a staging xors its token in.
pub const IMAGE_MAGIC: u64 = u64::from_be_bytes(*b"RESIDENT");
/// Payload words in the request after its header: `addr`, `len`, `token`.
pub const REQUEST_WORDS: usize = 3;
/// Payload words in every completion; the kernel driver fixes this width.
pub const REPLY_WORDS: usize = 3;

/// The first payload word of the bootloader's answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Code {
    /// The image is in place and about to run.
    Booted = 0,
    /// `addr..addr + len` is outside device memory or the image limit.
    BadRange = 1,
    /// The bytes at `addr` do not open with [`IMAGE_MAGIC`] xor the token.
    NoImage = 2,
    /// The request had another shape; nothing else is served before the firmware image runs.
    NotBoot = 3,
    /// The image does not match the firmware digest bound into this signed bootloader.
    BadDigest = 4,
}

impl Code {
    pub const fn from_word(word: u64) -> Option<Self> {
        match word {
            0 => Some(Self::Booted),
            1 => Some(Self::BadRange),
            2 => Some(Self::NoImage),
            3 => Some(Self::NotBoot),
            4 => Some(Self::BadDigest),
            _ => None,
        }
    }
}

/// The kernel driver's queue entry header: `{id, reserved, payload words, opcode}` little-endian.
pub mod header {
    const OPCODE: u64 = 1;
    const WORDS_SHIFT: u32 = 48;
    const OPCODE_SHIFT: u32 = 56;

    pub const fn pack(id: u32, words: u8) -> u64 {
        id as u64 | (words as u64) << WORDS_SHIFT | OPCODE << OPCODE_SHIFT
    }

    /// `(id, payload words)` of a header carrying the driver's opcode.
    pub const fn unpack(word: u64) -> Option<(u32, u8)> {
        if word >> OPCODE_SHIFT != OPCODE {
            return None;
        }
        Some((word as u32, (word >> WORDS_SHIFT) as u8))
    }
}

/// Whether `addr..addr + len` can hold a firmware image: inside device memory and the limit.
pub const fn in_range(addr: u64, len: u64) -> bool {
    const DEVICE_START: u64 = 0xc0_0000_0000;
    const DEVICE_END: u64 = 0xd0_0000_0000;
    len >= 16
        && len <= IMAGE_LIMIT as u64
        && len.is_multiple_of(8)
        && addr.is_multiple_of(8)
        && addr >= DEVICE_START
        && addr <= DEVICE_END - len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_code_values() {
        assert_eq!(Code::from_word(0), Some(Code::Booted));
        assert_eq!(Code::from_word(1), Some(Code::BadRange));
        assert_eq!(Code::from_word(2), Some(Code::NoImage));
        assert_eq!(Code::from_word(3), Some(Code::NotBoot));
        assert_eq!(Code::from_word(4), Some(Code::BadDigest));
        assert_eq!(Code::from_word(5), None);
    }

    #[test]
    fn header_roundtrips_and_rejects_other_opcodes() {
        assert_eq!(header::unpack(header::pack(0x0102_0304, 3)), Some((0x0102_0304, 3)));
        assert_eq!(header::pack(0x0102_0304, 3), 0x0103_0000_0102_0304);
        assert_eq!(header::unpack(7), None);
    }

    #[test]
    fn accepts_only_device_memory_that_fits() {
        assert!(in_range(0xc0_0000_0000, 16));
        assert!(in_range(0xd0_0000_0000 - 16, 16));
        assert!(!in_range(0xd0_0000_0000 - 8, 16), "past the end");
        assert!(!in_range(0xbf_ffff_fff8, 16), "below the start");
        assert!(!in_range(0xc0_0000_0004, 16), "unaligned");
        assert!(!in_range(0xc0_0000_0000, IMAGE_LIMIT as u64 + 8), "over the limit");
    }
}
