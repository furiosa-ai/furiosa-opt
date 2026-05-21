//! Fetch Engine: DM → Tensor Unit stream.
//!
//! Reads a tensor from a `BufTensor` in data memory and streams it into the
//! Tensor Unit pipeline as a `FetchTensor`.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::cast::FetchCast;
use crate::context::*;
use crate::engine::CanApplyFetch;
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::tu::{Position, TuTensor};

/// Output packet must be `FETCH_ALIGN_BYTES`-byte aligned.
const FETCH_ALIGN_BYTES: usize = 8;

/// After the fetch engine.
#[derive(Debug)]
pub struct PositionFetch;

impl Position for PositionFetch {}

/// Tensor streamed after the fetch engine.
pub type FetchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetch, D, Chip, Cluster, Slice, Time, Packet, B>;

// ANCHOR: fetch_impl
impl<'l, const T: Tu, P: CanApplyFetch, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Performs the fetch operation.
    #[primitive(TuTensor::fetch)]
    pub fn fetch<D2: Scalar, Time2: M, Packet2: M>(
        self,
    ) -> FetchTensor<'l, T, D2, Chip, Cluster, Slice, Time2, Packet2, B>
    where
        D: FetchCast<D2>,
    {
        verify_fetch::<D2, Cluster, Slice, Packet2>();
        FetchTensor::new(self.ctx, self.inner.map(|v| v.map(|v| v.cast())).transpose(true))
    }
}
// ANCHOR_END: fetch_impl

/// Validates Fetch engine constraints: Cluster size = 2, Slice size = 256,
/// output packet is `FETCH_ALIGN_BYTES`-byte aligned.
fn verify_fetch<D2: Scalar, Cluster: M, Slice: M, Packet2: M>() {
    assert_eq!(Cluster::SIZE, 2, "Cluster size must be 2, got {}", Cluster::SIZE);
    assert_eq!(Slice::SIZE, 256, "Slice size must be 256, got {}", Slice::SIZE);
    let packet_bytes = D2::size_in_bytes_from_length(Packet2::SIZE);
    assert_eq!(
        packet_bytes % FETCH_ALIGN_BYTES,
        0,
        "Fetch output packet must be {FETCH_ALIGN_BYTES}-byte aligned, got {packet_bytes} bytes.",
    );
}
