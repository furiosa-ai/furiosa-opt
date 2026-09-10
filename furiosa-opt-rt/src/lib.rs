//! Host runtime for device functions: opens a device's chips, loads their images on them and
//! launches them on device buffers. `DESIGN.md` says why it is shaped this way.

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("invalid NPU topology: {0}")]
    Topology(String),
    #[error("profiling configuration is invalid: {0}")]
    Profile(String),
    #[error("firmware error: {0}")]
    Firmware(String),
    #[error("device error: {0}")]
    Device(String),
    #[error("npu{0} is already reserved by another process")]
    Reserved(u8),
    #[error("the launch timeout is invalid: {0}")]
    Timeout(String),
    #[error("memory error: {0}")]
    Memory(String),
    #[error("transfer error: {0}")]
    Transfer(String),
    #[error("a launch failed earlier and the device takes no more: {0}")]
    Poisoned(Box<Error>),
    #[error("a buffer or a chip rank belongs to another device")]
    ForeignBuffer,
    #[error(transparent)]
    Function(#[from] FunctionError),
}

pub type Result<T> = std::result::Result<T, Error>;

mod buffer;
mod device;
mod function;
mod pinned;

pub use buffer::{Buffer, View};
pub use device::{Builder, ChipRank, Device, Topology};
pub use function::{Function, FunctionError, Launch, Profiled, Span, Trace};
pub use furiosa_opt_abi::image::{self, Image};
pub use pinned::Pinned;
