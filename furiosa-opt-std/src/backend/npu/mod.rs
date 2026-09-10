mod backend;
mod bind;
mod function;
mod host;
mod output;
mod registry;

pub use backend::Npu;
pub use function::{Function, function};
pub use host::HostBuf;
pub use registry::Key;

#[doc(hidden)]
pub mod __private {
    pub use super::output::DeviceOutput;
}
