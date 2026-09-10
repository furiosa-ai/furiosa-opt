//! Wire protocol between the host runtime and the firmware. This is
//! transport implementation, replaceable without changing the call contract.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate bincode2 as bincode;
extern crate thiserror_core as thiserror;

pub mod entry;
mod frame;
mod message;
pub mod status;

pub use entry::{COMPLETION_RING, Error as TransportError, MAX_SUBMISSION_WORDS, REPLY_WORDS, SUBMISSION_RING};
pub use frame::{Header, LAUNCHED_WORDS, MAX_LAUNCH_ARGS, Message};
pub use message::{
    Args, Error, Identity, MAX_RESPONSE_WORDS, PROFILE_CAPACITY, PROFILE_CHUNK_CAPACITY, PROFILE_TOTAL_CAPACITY,
    ProfileRecord, ProfileRequest, Records, Request, Response, Staged,
};

/// Maximum chips in one group and the chip-to-rank table width.
pub const MAX_GROUP: usize = 8;
