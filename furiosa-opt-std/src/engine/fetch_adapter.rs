//! Fetch Adapter: per-element transforms after the Fetch Engine.
//!
//! Each stage is exposed as a method on `FetchTensor`, applied in the
//! hardware pipeline order:
//!
//! - `fetch_mask::<OutTime, OutPacket>()` → `FetchMaskTensor`
//! - `fetch_table_lookup::<OutD>()` → `FetchTableLookupTensor`
//! - `fetch_cast::<OutD>()` → `FetchCastTensor` (type casting, with optional zero-point subtraction)
//!
//! Each stage is optional. A `FetchTensor` can flow directly into Switch
//! or Collect without any adapter call. Each downstream stage accepts every
//! earlier stage's tensor as input (mask can chain into table-lookup,
//! table-lookup into cast, and so on).
//!
//! `fetch_cast` is fully implemented. `fetch_mask` and
//! `fetch_table_lookup` are stubs whose `verify_*()` helpers `todo!()` at
//! runtime. The masking config (`FetchMaskConfig`) is not a kernel-author
//! argument: `fetch_mask` takes no runtime parameter, and the compiler
//! derives the config from the difference between the input and output
//! mappings.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::cast::FetchCast;
use crate::context::*;
use crate::engine::{CanApplyFetchCast, CanApplyFetchMask, CanApplyFetchTableLookup};
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::tu::{Position, TuTensor};

/// Compiler-derived configuration for the Fetch Adapter's masking stage.
///
/// This is **not** a kernel-author argument. `fetch_mask` takes no
/// runtime parameter; the compiler derives this config from the
/// difference between the input and output mappings (the `#` → `#{0}`
/// pad-kind change carried by `OutTime` / `OutPacket`). See the book
/// chapter `computing-tensors/fetch-adapter.md` (the "Masking" section)
/// for what each field means.
#[derive(Debug, Clone, Default)]
pub struct FetchMaskConfig {
    /// Sequencer loop axis whose rightmost iteration carries the pad.
    /// Recorded as an axis (not a raw loop/dimension index) because the
    /// sequencer optimizer merges consecutive entries.
    pub last_axis: usize,
    /// How `rightmost_valid_count` is consumed.
    pub valid_count_dim: ValidCountDim,
    /// Per-cell valid (non-pad) element count.
    pub rightmost_valid_count: [u8; 8],
}

/// Selects how `FetchMaskConfig::rightmost_valid_count` is interpreted.
#[derive(Debug, Clone, Copy, Default)]
pub enum ValidCountDim {
    /// Use entry `0` only, applied to the rightmost cell of `last_axis`.
    #[default]
    Rightmost,
    /// Use entries `0` and `1` for the last two cells.
    RightmostAndSecondRightmost,
    /// Vary the array along the named sequencer loop; entry `i` is the
    /// valid count for the i-th iteration of that loop.
    Iterator(usize),
}

/// After the Fetch Adapter's masking stage.
#[derive(Debug)]
pub struct PositionFetchMask;

impl Position for PositionFetchMask {}

/// Tensor streamed after `fetch_mask`.
pub type FetchMaskTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchMask, D, Chip, Cluster, Slice, Time, Packet, B>;

/// After the Fetch Adapter's table-lookup stage.
#[derive(Debug)]
pub struct PositionFetchTableLookup;

impl Position for PositionFetchTableLookup {}

/// Tensor streamed after `fetch_table_lookup`.
pub type FetchTableLookupTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchTableLookup, D, Chip, Cluster, Slice, Time, Packet, B>;

/// After the Fetch Adapter's type-casting stage.
#[derive(Debug)]
pub struct PositionFetchCast;

impl Position for PositionFetchCast {}

/// Tensor streamed after `fetch_cast`.
pub type FetchCastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchCast, D, Chip, Cluster, Slice, Time, Packet, B>;

// ANCHOR: fetch_mask_impl
impl<'l, const T: Tu, P: CanApplyFetchMask, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Adapter's masking stage.
    ///
    /// Zeroes the padded slots described by the book chapter. `OutTime`
    /// and `OutPacket` carry the pad-kind change at the type level (e.g.
    /// `m![D # n]` → `m![D #{0} n]`). Callers spell them out explicitly
    /// because downstream methods do not constrain their input shape. The
    /// hardware masking config is derived by the compiler, so this method
    /// takes no runtime argument (see [`FetchMaskConfig`]).
    #[primitive(TuTensor::fetch_mask)]
    pub fn fetch_mask<OutTime: M, OutPacket: M>(
        self,
    ) -> FetchMaskTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_fetch_mask::<Time, Packet, OutTime, OutPacket>();
        FetchMaskTensor::new(self.ctx, self.inner.transpose(true))
    }
}
// ANCHOR_END: fetch_mask_impl

// ANCHOR: fetch_table_lookup_impl
impl<
    'l,
    const T: Tu,
    P: CanApplyFetchTableLookup,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Adapter's table-lookup stage.
    #[primitive(TuTensor::fetch_table_lookup)]
    #[allow(unreachable_code)]
    pub fn fetch_table_lookup<OutD: Scalar>(
        self,
    ) -> FetchTableLookupTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B> {
        verify_fetch_table_lookup::<D, OutD, Time, Packet>();
        FetchTableLookupTensor::new(self.ctx, todo!())
    }
}
// ANCHOR_END: fetch_table_lookup_impl

// ANCHOR: fetch_cast_impl
impl<'l, const T: Tu, P: CanApplyFetchCast, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Adapter's type-casting stage.
    ///
    /// Converts the stream's element type from `D` to `OutD`. The mapping
    /// shape is preserved.
    #[primitive(TuTensor::fetch_cast)]
    pub fn fetch_cast<OutD: Scalar>(self) -> FetchCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: FetchCast<OutD>,
    {
        FetchCastTensor::new(self.ctx, self.inner.map(|v| v.map(|v| v.cast())))
    }
}
// ANCHOR_END: fetch_cast_impl

#[allow(clippy::extra_unused_type_parameters)]
fn verify_fetch_mask<Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    todo!("fetch_mask is not yet implemented")
}

#[allow(clippy::extra_unused_type_parameters)]
fn verify_fetch_table_lookup<D: Scalar, OutD: Scalar, Time: M, Packet: M>() {
    todo!("fetch_table_lookup is not yet implemented")
}
