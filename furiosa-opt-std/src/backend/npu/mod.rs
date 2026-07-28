mod backend;
mod convert;
mod ffi;
mod kernel;

pub use backend::Npu;
pub use convert::{ExtendBuffers, KernelOutput};
pub use ffi::NpuDesc;
pub(crate) use ffi::bind_device;
pub(crate) use kernel::CpuBuffer;
pub use kernel::{Buffer, Kernel};
