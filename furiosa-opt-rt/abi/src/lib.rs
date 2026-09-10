//! What the compiler, the host runtime and the device agree on: the device function image and how the
//! bootloader brings the firmware image up.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
extern crate bincode2 as bincode;
extern crate thiserror_core as thiserror;

pub mod bootloader;
/// Needs `alloc`: an image owns its tables.
#[cfg(feature = "alloc")]
pub mod image;
pub mod reg;
pub mod ring;

#[cfg(feature = "alloc")]
pub use image::Image;

/// Revision of the shared host-device ABI.
pub const ABI_VERSION: u32 = 1;

/// PEs a device has, in clusters of [`CLUSTER_PES`]. A function uses a prefix of a device's PEs
/// and has one task column per cluster that prefix touches.
pub const CHIP_PES: u8 = 8;
pub const CLUSTER_PES: u8 = 4;

/// Where a cluster sees device memory in its address space; the PDMA addresses it the same way.
pub const DEVICE_MEMORY: core::ops::Range<u64> = 0xc0_0000_0000..0xd0_0000_0000;
