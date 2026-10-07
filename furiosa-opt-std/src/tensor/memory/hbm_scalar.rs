use furiosa_opt_macro::primitive;
use furiosa_opt_rt::Buffer;

use crate::Error;
use crate::backend::Backend;
use crate::context::{Dma, DmaContext};
use crate::runtime::CurrentBackend;
use crate::scalar::{RuntimeScalar, Scalar};

/// One scalar value replicated in the local HBM of every chip in the launch group.
#[primitive(HbmScalar)]
pub struct HbmScalar<D: RuntimeScalar, B: Backend = CurrentBackend> {
    value: D,
    buffer: Option<Buffer>,
    _backend: std::marker::PhantomData<B>,
}

impl<D: RuntimeScalar, B: Backend> std::fmt::Debug for HbmScalar<D, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HbmScalar").field("buffer", &self.buffer).finish()
    }
}

impl<D: RuntimeScalar, B: Backend> HbmScalar<D, B> {
    /// Uploads one host value as an HBM scalar.
    pub async fn from_host(value: D, dma: &mut DmaContext<{ Dma::Pcie }, B>) -> Result<Self, Error> {
        B::to_hbm_scalar(value, dma).await
    }

    /// Updates this scalar through its existing HBM allocation.
    pub async fn write(&mut self, value: D, dma: &mut DmaContext<{ Dma::Pcie }, B>) -> Result<(), Error> {
        B::to_hbm_scalar_into(value, dma, self).await
    }

    /// Returns the scalar's packed HBM image.
    pub fn to_buf(&self) -> Vec<u8> {
        Self::encode(self.value)
    }

    /// Loads the value into an SPM scalar every chip and cluster can read.
    #[primitive(HbmScalar::to_spm)]
    pub fn to_spm(&self, _dma: &mut DmaContext<{ Dma::Tensor }, B>) -> D {
        self.value
    }
}

impl<D: RuntimeScalar, B: Backend> HbmScalar<D, B> {
    pub(crate) fn from_value(value: D) -> Self {
        Self {
            value,
            buffer: None,
            _backend: std::marker::PhantomData,
        }
    }

    pub(crate) fn place(&mut self, buffer: Buffer) {
        self.buffer = Some(buffer);
    }

    pub(crate) fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    pub(crate) fn set_value(&mut self, value: D) {
        self.value = value;
    }

    pub(crate) fn encode(value: D) -> Vec<u8> {
        D::Storage::to_buf(&[value.into_storage()])
    }
}
