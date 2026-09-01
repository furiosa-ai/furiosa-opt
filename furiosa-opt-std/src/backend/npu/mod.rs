mod backend;
mod convert;
mod ffi;
mod kernel;
mod output;
mod registry;

pub use backend::Npu;
pub use convert::ExtendBuffers;
pub use ffi::NpuDesc;
pub(crate) use ffi::bind_device;
pub(crate) use kernel::CpuBuffer;
pub use kernel::{Buffer, Kernel, Kernels, kernel};
pub use output::{KernelOutput, KernelOutputDestination};
pub use registry::Key;
