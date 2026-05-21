use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::tensor::memory::{Address, HbmTensor, HostTensor};
use crate::tensor::raw::RawTensor;

/// Backend for tensor operations.
///
/// `Simulation`: interprets operations via mapping expressions on the host CPU (default).
/// `Emulation`: host-side `BufRawTensor` storage for the future Cpu+Buffer interpreter; today
/// every value-producing method is `todo!()` placeholder.
/// `Npu`: host-side code owns native staging buffers and performs DMA in `to_hbm` / `from_hbm`.
/// It does not interpret tensor math on the host; storage-level methods (on `BufRawTensor`)
/// are `todo!()` placeholders shared with the Emulation backend.
/// `Typecheck`: shape/mapping validation only, no value-level loops on host
/// (`--cfg backend="typecheck"`). Returns empty (phantom) tensors throughout.
///
/// `Backend` selects the concrete storage type (`Self::RawTensor<D>`) and supplies only the
/// genuinely cross-tensor, backend-specific protocols: DMA (`to_hbm` / `from_hbm`). Per-operation
/// methods (zip_with, write_scatter, map, reduce, apply_branch_operands, …) live on
/// [`crate::tensor::raw::RawTensor`] — they only need RawTensor-level primitives
/// (`read_index` / `write_index` / `uninit_from_axes`), not Backend state.
pub trait Backend: Sized + 'static {
    /// Backend storage type. Per-operation tensor behavior lives on the corresponding
    /// [`RawTensor`] impl.
    type RawTensor<D: Scalar>: RawTensor<D>;

    /// Transfer host tensor to HBM. Host-only — invoked from test/setup code over PCIe DMA, never
    /// from device kernel MIR — so no `#[primitive(...)]` annotation is needed.
    fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        address: Address,
    ) -> impl std::future::Future<Output = HbmTensor<D, Chip, Element2, Self>>;

    /// Transfer HBM tensor to host. (Renamed from `to_host` for symmetry with `to_hbm`.) Host-only
    /// — see the `to_hbm` note above.
    fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> impl std::future::Future<Output = HostTensor<D, Element2, Self>>;
}
