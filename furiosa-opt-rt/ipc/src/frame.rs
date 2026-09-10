//! Frames. A frame is a [`Header`] and the bincode encoding of one message, padded to whole
//! words; both use fixed-width little-endian integers. A launch encodes into the caller's buffer
//! with its arguments on a word boundary, where the device reads them in place, and its answer
//! encodes into the device's buffer: neither side allocates.

use furiosa_opt_abi::ABI_VERSION;

use super::entry::MAX_SUBMISSION_WORDS;
use super::{Error, MAX_RESPONSE_WORDS, Request, Response};

const WORD_BYTES: usize = size_of::<u64>();

/// Words in a launch answer that carries no profile records: one completion entry's worth.
pub const LAUNCHED_WORDS: usize = 3;
/// Where a launch's arguments start in the frame: after the header, the variant and the count.
const LAUNCH_ARGS_AT: usize = size_of::<Header>() + 4 + 8;
const _: () = assert!(LAUNCH_ARGS_AT.is_multiple_of(WORD_BYTES));
/// Arguments one launch frame carries at most, beside its function and profile request.
pub const MAX_LAUNCH_ARGS: usize = (MAX_SUBMISSION_WORDS * WORD_BYTES - LAUNCH_ARGS_AT - 24 - 6) / WORD_BYTES;

/// How every frame opens: the ABI it speaks and the words it spans, header included. Half a word,
/// so the message starts on the other half.
#[derive(bincode::Encode, bincode::Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u16,
    pub words: u16,
}

const HEADER_BYTES: usize = size_of::<Header>();
const _: () = assert!(HEADER_BYTES == 4);
const _: () = assert!(MAX_RESPONSE_WORDS <= u16::MAX as usize);

impl Header {
    /// The header in a frame's first word, if it speaks this ABI.
    pub fn read(word: u64) -> Result<Self, Error> {
        let (header, _) = bincode::decode_from_slice::<Self, _>(&word.to_ne_bytes()[..HEADER_BYTES], config())
            .map_err(|_| Error::Decode)?;
        if u32::from(header.version) != ABI_VERSION {
            return Err(Error::Version(u32::from(header.version)));
        }
        Ok(header)
    }
}

/// Both messages: written as a frame into a caller's words, read back from exactly those words.
pub trait Message<'a>: bincode::Encode + bincode::BorrowDecode<'a, ()> + Sized {
    /// Writes the frame into `out` and returns the words it spans.
    fn write(&self, out: &mut [u64]) -> Result<usize, Error> {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
        let (header, message) = bytes.split_at_mut_checked(HEADER_BYTES).ok_or(Error::Truncated)?;
        let len = bincode::encode_into_slice(self, message, config()).map_err(|_| Error::Truncated)?;
        let words = (HEADER_BYTES + len).div_ceil(WORD_BYTES);
        message[len..words * WORD_BYTES - HEADER_BYTES].fill(0);
        let version = ABI_VERSION as u16;
        let words = u16::try_from(words).map_err(|_| Error::Truncated)?;
        bincode::encode_into_slice(Header { version, words }, header, config()).map_err(|_| Error::Truncated)?;
        Ok(usize::from(words))
    }

    /// The message in exactly the frame `words`.
    fn read(words: &'a [u64]) -> Result<Self, Error> {
        let header = Header::read(*words.first().ok_or(Error::Truncated)?)?;
        if words.len() != usize::from(header.words) {
            return Err(Error::Trailing);
        }
        let bytes: &[u8] = bytemuck::cast_slice(words);
        let (message, used) =
            bincode::borrow_decode_from_slice(&bytes[HEADER_BYTES..], config()).map_err(|_| Error::Decode)?;
        if bytes[HEADER_BYTES + used..].iter().any(|byte| *byte != 0) {
            return Err(Error::Padding);
        }
        Ok(message)
    }
}

impl<'a> Message<'a> for Request<'a> {}
impl Message<'_> for Response {}

/// Fixed-width little-endian integers: a word on the wire is a word in the struct. The limit is
/// the largest frame either side accepts.
const fn config() -> bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
    bincode::config::Limit<{ MAX_RESPONSE_WORDS * WORD_BYTES }>,
> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
        .with_limit()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::{
        Args, Identity, PROFILE_CAPACITY, PROFILE_CHUNK_CAPACITY, PROFILE_TOTAL_CAPACITY, ProfileRecord,
        ProfileRequest, Records, Staged,
    };

    use super::*;

    fn words<'a, T: Message<'a>>(message: &T, limit: usize) -> Vec<u64> {
        let mut out = vec![0; limit];
        let len = message.write(&mut out).expect("message fits");
        out.truncate(len);
        out
    }

    fn launch(args: &[u64], profile: Option<ProfileRequest>) -> Request<'_> {
        Request::Launch {
            args: Args::from(args),
            function: Staged {
                addr: 0xc0_0000_1000,
                len: 44_000,
                token: 3,
            },
            profile,
        }
    }

    fn initialize(members: Vec<u8>, chip: u8) -> Request<'static> {
        Request::Initialize {
            identity: Identity::new(members, chip).expect("identity"),
            column: 1,
            timeout_secs: 5,
            log_level: 0,
        }
    }

    #[test]
    fn writes_args_where_the_device_reads_them() {
        let args = [7, 8, 9];
        let request = launch(&args, Some(ProfileRequest::new(4, 16).expect("profile")));
        let frame = words(&request, MAX_SUBMISSION_WORDS);
        assert_eq!(&frame[LAUNCH_ARGS_AT / WORD_BYTES..][..args.len()], &args);
        let decoded = Request::read(&frame).expect("decodes");
        assert_eq!(decoded, request);
        let Request::Launch { args: decoded, .. } = decoded else {
            unreachable!()
        };
        assert_eq!(decoded.0.as_ptr(), frame[LAUNCH_ARGS_AT / WORD_BYTES..].as_ptr());
        assert_eq!(Header::read(frame[0]).expect("header").words as usize, frame.len());
    }

    #[test]
    fn fills_a_submission_at_the_argument_limit() {
        let args = [0; MAX_LAUNCH_ARGS];
        let frame = words(
            &launch(&args, Some(ProfileRequest::new(3, 1).expect("profile"))),
            MAX_SUBMISSION_WORDS,
        );
        assert_eq!(frame.len(), MAX_SUBMISSION_WORDS);
        assert!(
            launch(&[0; MAX_LAUNCH_ARGS + 1], None)
                .write(&mut [0; MAX_SUBMISSION_WORDS + 8])
                .is_err()
        );
    }

    #[test]
    fn unprofiled_launch_one_completion() {
        let response = Response::Launched {
            code: -3,
            records: Records::default(),
        };
        let frame = words(&response, MAX_RESPONSE_WORDS);
        assert_eq!(frame.len(), LAUNCHED_WORDS);
        assert_eq!(Response::read(&frame).expect("decodes"), response);
    }

    #[test]
    fn answers_a_full_profile_in_one_response() {
        let records = (0..PROFILE_CHUNK_CAPACITY)
            .map(|chunk| ProfileRecord {
                chunk,
                hits: vec![u64::from(chunk); (PROFILE_TOTAL_CAPACITY / PROFILE_CHUNK_CAPACITY) as usize],
            })
            .collect();
        let response = Response::Launched {
            code: 0,
            records: Records::new(records).expect("records fit"),
        };
        let frame = words(&response, MAX_RESPONSE_WORDS);
        assert_eq!(frame.len(), MAX_RESPONSE_WORDS);
        assert_eq!(Response::read(&frame).expect("decodes"), response);
    }

    #[test]
    fn initialize_round_trips() {
        let request = initialize(vec![1, 3], 3);
        assert_eq!(
            Request::read(&words(&request, MAX_SUBMISSION_WORDS)).expect("decodes"),
            request
        );
        let response = Response::Initialized {
            chip: 3,
            rank: 1,
            members: 2,
        };
        assert_eq!(
            Response::read(&words(&response, MAX_RESPONSE_WORDS)).expect("decodes"),
            response
        );
    }

    #[test]
    fn rejects_frames_that_misstate_themselves() {
        let initialize = initialize(vec![0], 0);
        let mut version = words(&initialize, MAX_SUBMISSION_WORDS);
        version[0] ^= 1;
        let mut trailing = words(&initialize, MAX_SUBMISSION_WORDS);
        trailing.push(0);
        let mut padding = words(
            &Response::Failed {
                code: 0,
                message: "x".into(),
            },
            MAX_RESPONSE_WORDS,
        );
        let last = padding.len() - 1;
        padding[last] |= 1 << 56;
        let mut short = words(&launch(&[1, 2], None), MAX_SUBMISSION_WORDS);
        short.pop();
        let mut bad_identity = words(&initialize, MAX_SUBMISSION_WORDS);
        bad_identity[1] = 0;
        for (label, result, expected) in [
            (
                "version",
                Request::read(&version).map(|_| ()),
                Error::Version(ABI_VERSION ^ 1),
            ),
            ("trailing", Request::read(&trailing).map(|_| ()), Error::Trailing),
            ("padding", Response::read(&padding).map(|_| ()), Error::Padding),
            ("short launch", Request::read(&short).map(|_| ()), Error::Trailing),
            ("identity", Request::read(&bad_identity).map(|_| ()), Error::Decode),
            (
                "profile",
                ProfileRequest::new(3, PROFILE_CAPACITY + 1).map(|_| ()),
                Error::Profile,
            ),
        ] {
            assert_eq!(result, Err(expected), "{label}");
        }
    }
}
