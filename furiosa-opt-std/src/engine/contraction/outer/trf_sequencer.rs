//! TRF Sequencer: TRF-side transpose for [`super::contract_outer`].
//!
//! Transposes the TRF tensor into the joint computation mapping. Tiling-axis
//! validation lives on the stream side (see [`super::stream_adapter`]).

use furiosa_mapping::*;

use crate::runtime::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::memory::TrfTensor;

pub(super) fn contract_outer<
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    TrfElement: M,
    OutTime: M,
    OutPacket: M,
    B: Backend,
>(
    trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Lane, TrfElement, B>,
) -> Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { OutTime }, { OutPacket }], B> {
    trf_tensor.inner.transpose(true)
}
