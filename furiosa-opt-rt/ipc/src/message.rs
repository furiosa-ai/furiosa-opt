use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use bincode2::de::read::BorrowReader;
use bincode2::de::{BorrowDecoder, Decoder};
use bincode2::error::DecodeError;
use bincode2::{BorrowDecode, Decode};

use super::MAX_GROUP;

/// Raw records held by the firmware's TUC profile queue.
pub const PROFILE_CAPACITY: u32 = 8192;
/// Raw profile hits retained across one launch.
pub const PROFILE_TOTAL_CAPACITY: u32 = 8192;
/// Profile records retained across one launch.
pub const PROFILE_CHUNK_CAPACITY: u32 = 256;

/// Largest response frame: a launch answer carrying every profile record it can, each record
/// its chunk, its hit count and its hits.
pub const MAX_RESPONSE_WORDS: usize = super::frame::LAUNCHED_WORDS
    + (PROFILE_CHUNK_CAPACITY as usize * (4 + 8)).div_ceil(8)
    + PROFILE_TOTAL_CAPACITY as usize;

/// Where a device function's image is staged in device memory. `token` tells this staging from
/// any earlier one at the same address, so the firmware never mistakes a new image for the one
/// it already holds.
#[derive(bincode2::Decode, bincode2::Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staged {
    pub addr: u64,
    pub len: u64,
    pub token: u64,
}

/// What one launch records: the task library's profile level, `Info` (3) through `Trace` (5),
/// and how many hits each chunk keeps, at most [`PROFILE_CAPACITY`]. Exists only in that range.
#[derive(bincode2::Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileRequest {
    level: u8,
    depth: u32,
}

impl ProfileRequest {
    pub fn new(level: u8, depth: u32) -> Result<Self, Error> {
        if !(3..=5).contains(&level) || depth == 0 || depth > PROFILE_CAPACITY {
            return Err(Error::Profile);
        }
        Ok(Self { level, depth })
    }

    pub fn level(self) -> u8 {
        self.level
    }

    pub fn depth(self) -> u32 {
        self.depth
    }
}

impl<Context> Decode<Context> for ProfileRequest {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Self::new(u8::decode(decoder)?, u32::decode(decoder)?).map_err(|_| DecodeError::Other("profile request"))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for ProfileRequest {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Decode::decode(decoder)
    }
}

/// Raw hardware profile hits captured for one task chunk.
#[derive(bincode2::Decode, bincode2::Encode, Debug, Clone, PartialEq, Eq)]
pub struct ProfileRecord {
    pub chunk: u32,
    pub hits: Vec<u64>,
}

/// One launch's records, within what a response carries: [`PROFILE_CHUNK_CAPACITY`] records,
/// [`PROFILE_CAPACITY`] hits each and [`PROFILE_TOTAL_CAPACITY`] in all. Exists only within.
#[derive(bincode2::Encode, Debug, Clone, PartialEq, Eq, Default)]
pub struct Records(Vec<ProfileRecord>);

impl Records {
    pub fn new(records: Vec<ProfileRecord>) -> Result<Self, Error> {
        if records.len() > PROFILE_CHUNK_CAPACITY as usize
            || records
                .iter()
                .any(|record| record.hits.len() > PROFILE_CAPACITY as usize)
            || records.iter().map(|record| record.hits.len()).sum::<usize>() > PROFILE_TOTAL_CAPACITY as usize
        {
            return Err(Error::Profile);
        }
        Ok(Self(records))
    }
}

impl core::ops::Deref for Records {
    type Target = [ProfileRecord];

    fn deref(&self) -> &[ProfileRecord] {
        &self.0
    }
}

impl From<Records> for Vec<ProfileRecord> {
    fn from(records: Records) -> Self {
        records.0
    }
}

impl<Context> Decode<Context> for Records {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Self::new(Vec::decode(decoder)?).map_err(|_| DecodeError::Other("profile records"))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Records {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Decode::decode(decoder)
    }
}

/// A chip and the ordered group it belongs to: at most [`MAX_GROUP`] chips, ascending,
/// one of them this one. Exists only in that shape.
#[derive(bincode2::Encode, Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    members: Vec<u8>,
    chip: u8,
}

impl Identity {
    pub fn new(members: Vec<u8>, chip: u8) -> Result<Self, Error> {
        if members.is_empty()
            || members.len() > MAX_GROUP
            || !members.windows(2).all(|pair| pair[0] < pair[1])
            || !members.iter().all(|&member| usize::from(member) < MAX_GROUP)
            || !members.contains(&chip)
        {
            return Err(Error::Identity);
        }
        Ok(Self { members, chip })
    }

    /// A group of this chip alone.
    pub fn single(chip: u8) -> Result<Self, Error> {
        Self::new(vec![chip], chip)
    }

    /// Group members in driver rank order.
    pub fn members(&self) -> &[u8] {
        &self.members
    }

    /// Physical chip index.
    pub fn chip(&self) -> u8 {
        self.chip
    }

    /// Rank within the group.
    pub fn rank(&self) -> u8 {
        self.members
            .iter()
            .position(|&member| member == self.chip)
            .expect("a constructed identity is one of its own members") as u8
    }

    /// How many chips the group has.
    pub fn size(&self) -> u8 {
        self.members.len() as u8
    }
}

impl<Context> Decode<Context> for Identity {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Self::new(Vec::decode(decoder)?, u8::decode(decoder)?).map_err(|_| DecodeError::Other("identity"))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Identity {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Decode::decode(decoder)
    }
}

/// Host to device request. A launch's arguments come first in the frame, where they sit on a
/// word boundary and the device reads them in place.
#[derive(bincode2::BorrowDecode, bincode2::Encode, Debug, Clone, PartialEq, Eq)]
pub enum Request<'a> {
    /// Runs the device function staged at `function` with the supplied args.
    Launch {
        args: Args<'a>,
        function: Staged,
        profile: Option<ProfileRequest>,
    },
    /// Gives the cluster its `column` among the group's clusters, which selects its task code in
    /// every image, its launch limit, and its `log::LevelFilter` index (0 off, 5 trace).
    Initialize {
        identity: Identity,
        column: u8,
        timeout_secs: u32,
        log_level: u8,
    },
}

/// A launch's arguments, at most [`super::MAX_LAUNCH_ARGS`], borrowed from the caller's buffer on
/// the way out and from the frame on the way in: neither side copies them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Args<'a>(pub &'a [u64]);

impl<'a> From<&'a [u64]> for Args<'a> {
    fn from(args: &'a [u64]) -> Self {
        Self(args)
    }
}

impl core::ops::Deref for Args<'_> {
    type Target = [u64];

    fn deref(&self) -> &[u64] {
        self.0
    }
}

impl bincode2::Encode for Args<'_> {
    fn encode<E: bincode2::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode2::error::EncodeError> {
        if self.0.len() > super::MAX_LAUNCH_ARGS {
            return Err(bincode2::error::EncodeError::Other("too many launch arguments"));
        }
        self.0.encode(encoder)
    }
}

impl<'a, 'de: 'a, Context> BorrowDecode<'de, Context> for Args<'a> {
    /// The words as they lie in the frame; they start on a word boundary, which the frame layout
    /// guarantees and [`super::frame`] pins.
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = u64::decode(decoder)? as usize;
        if len > super::MAX_LAUNCH_ARGS {
            return Err(DecodeError::Other("too many launch arguments"));
        }
        let bytes = decoder.borrow_reader().take_bytes(len * size_of::<u64>())?;
        bytemuck::try_cast_slice(bytes)
            .map(Self)
            .map_err(|_| DecodeError::Other("launch arguments are not word aligned"))
    }
}

/// Device to host response.
#[derive(bincode2::Decode, bincode2::Encode, Debug, Clone, PartialEq, Eq)]
pub enum Response {
    /// A launch failed and said why: a panicked task's assertion and backtrace, or a request the firmware
    /// could not decode.
    Failed { code: i32, message: String },
    /// A launch finished.
    Launched { code: i32, records: Records },
    /// Published chip identity.
    Initialized { chip: u8, rank: u8, members: u8 },
}

#[derive(thiserror_core::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Frame ended in the middle of a field, or claimed more than it carries.
    #[error("frame ended before the field did")]
    Truncated,
    /// Frame was written against another ABI revision.
    #[error("frame speaks ABI version {0}, expected {expected}", expected = furiosa_opt_abi::ABI_VERSION)]
    Version(u32),
    #[error("frame contains an invalid bincode payload")]
    Decode,
    #[error("frame includes complete words beyond its byte length")]
    Trailing,
    #[error("frame padding is nonzero")]
    Padding,
    #[error("a group of no more than {MAX_GROUP} ascending chips must contain its own")]
    Identity,
    #[error("a profile request is level 3 to 5 and depth 1 to {PROFILE_CAPACITY}, within the response's capacity")]
    Profile,
}
