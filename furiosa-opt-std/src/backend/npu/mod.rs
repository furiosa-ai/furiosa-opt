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

impl crate::Device<Npu> {
    /// Available HBM per chip after runtime reservations and live allocations.
    pub fn memory(&self) -> Result<furiosa_opt_rt::Memory, crate::Error> {
        Ok(self.pdma.device().inner().memory()?)
    }
}

#[doc(hidden)]
pub mod __private {
    pub use super::output::DeviceOutput;
}
