//! Landing a staged firmware image in the scratchpad under its signed digest.

#![no_std]

use furiosa_opt_abi::bootloader::{Code, IMAGE_MAGIC};
use sha2::{Digest, Sha256};

/// A staged firmware image and the digest that authorizes it.
pub struct Image<'a> {
    pub words: usize,
    pub token: u64,
    pub digest: &'a [u8; 32],
}

impl Image<'_> {
    /// Copies and authenticates each source word once; only [`Code::Booted`] authorizes entry.
    /// A failed digest may leave destination bytes and returns [`Code::BadDigest`].
    pub fn land(self, mut read: impl FnMut(usize) -> u64, mut write: impl FnMut(usize, u64)) -> Code {
        if self.words < 2 {
            return Code::BadRange;
        }
        let first = read(0);
        if first != IMAGE_MAGIC ^ self.token {
            return Code::NoImage;
        }

        let mut digest = Sha256::new();
        for word in 0..self.words {
            let staged = if word == 0 { first } else { read(word) };
            let image = if word == 0 { staged ^ self.token } else { staged };
            write(word, image);
            digest.update(image.to_le_bytes());
        }
        if digest.finalize().as_slice() == self.digest {
            Code::Booted
        } else {
            Code::BadDigest
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: u64 = 0x1234_5678_9abc_def0;
    const IMAGE: [u64; 3] = [IMAGE_MAGIC, 0x0102_0304_0506_0708, 0x1112_1314_1516_1718];

    fn digest(words: &[u64]) -> [u8; 32] {
        let mut digest = Sha256::new();
        for word in words {
            digest.update(word.to_le_bytes());
        }
        digest.finalize().into()
    }

    fn stage() -> [u64; 3] {
        let mut staged = IMAGE;
        staged[0] ^= TOKEN;
        staged
    }

    #[test]
    fn lands_bound_image() {
        let staged = stage();
        let mut reads = [0; 3];
        let mut destination = [0; 3];

        let digest = digest(&IMAGE);
        let code = Image {
            words: staged.len(),
            token: TOKEN,
            digest: &digest,
        }
        .land(
            |word| {
                reads[word] += 1;
                staged[word]
            },
            |word, value| destination[word] = value,
        );

        assert_eq!(code, Code::Booted);
        assert_eq!(reads, [1; 3]);
        assert_eq!(destination, IMAGE);
    }

    #[test]
    fn replaces_rejected_image() {
        let mut staged = stage();
        staged[1] ^= 1;
        let mut destination = [0; 3];

        let digest = digest(&IMAGE);
        let image = || Image {
            words: IMAGE.len(),
            token: TOKEN,
            digest: &digest,
        };
        let code = image().land(|word| staged[word], |word, value| destination[word] = value);

        assert_eq!(code, Code::BadDigest);
        assert_ne!(destination, IMAGE);

        staged[1] ^= 1;
        let code = image().land(|word| staged[word], |word, value| destination[word] = value);

        assert_eq!(code, Code::Booted);
        assert_eq!(destination, IMAGE);
    }
}
