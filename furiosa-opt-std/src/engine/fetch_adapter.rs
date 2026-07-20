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
//! `fetch_cast` is fully implemented. `fetch_mask` and `fetch_table_lookup` are stubs whose
//! `verify_*()` helpers `todo!()` at runtime.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::{FetchCast, FetchZeroPointSub};
use crate::constraints;
use crate::context::*;
use crate::engine::{CanApplyFetchCast, CanApplyFetchMask, CanApplyFetchTableLookup, CanApplyFetchZeroPointSub};
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
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

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchMaskTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

/// After the Fetch Adapter's table-lookup stage.
#[derive(Debug)]
pub struct PositionFetchTableLookup;

impl Position for PositionFetchTableLookup {}

/// Tensor streamed after `fetch_table_lookup`.
pub type FetchTableLookupTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchTableLookup, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchTableLookupTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

/// After the Fetch Adapter's type-casting stage.
#[derive(Debug)]
pub struct PositionFetchCast;

impl Position for PositionFetchCast {}

/// Tensor streamed after `fetch_cast`.
pub type FetchCastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchCast, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchCastTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

// ANCHOR: fetch_mask_impl
impl<'l, const T: Tu, P: CanApplyFetchMask, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Adapter's masking stage.
    ///
    /// Zeroes the padded slots described by the book chapter. `OutTime`
    /// and `OutPacket` carry the pad-kind change at the type level (e.g.
    /// `m![D # n]` → `m![D #{0} n]`). Callers spell them out explicitly
    /// because downstream methods do not constrain their input shape. This
    /// method takes no runtime argument (see [`FetchMaskConfig`]).
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
impl<
    'l,
    const T: Tu,
    P: CanApplyFetchCast,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
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
        FetchCastTensor::new(self.ctx, self.inner.map(|v| v.cast()))
    }
}
// ANCHOR_END: fetch_cast_impl

/// After the Fetch Adapter's zero-point-subtraction stage.
#[derive(Debug)]
pub struct PositionFetchZeroPointSub;

impl Position for PositionFetchZeroPointSub {}

/// Tensor streamed after `fetch_zero_point_sub`. Its element type is a
/// zero-point-subtracted staging type ([`i5`]/[`i9`]); the type system routes
/// it only into `contract_outer`.
pub type FetchZeroPointSubTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchZeroPointSub, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchZeroPointSubTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

// ANCHOR: fetch_zero_point_sub_impl
impl<
    'l,
    const T: Tu,
    P: CanApplyFetchZeroPointSub,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Adapter's zero-point-subtraction stage.
    ///
    /// Subtracts `zero_point` and widens the stream from `D` to its contraction-engine staging
    /// type `OutD` (`i4 -> i5`, `i8 -> i9`), the only way to produce an i5/i9
    /// stream. The result may only feed `contract_outer`; it is not
    /// [`MaterializableScalar`], so committing or re-routing it is a compile
    /// error. The mapping shape is preserved.
    ///
    /// Panics if `zero_point` is outside the source type's range
    /// ([`FetchZeroPointSub::ZERO_POINT_RANGE`]); a zero point in range keeps
    /// every widened residual within `OutD`, so this one check (independent of
    /// the stream data) is enough.
    #[primitive(TuTensor::fetch_zero_point_sub)]
    pub fn fetch_zero_point_sub<OutD: Scalar>(
        self,
        zero_point: i32,
    ) -> FetchZeroPointSubTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: FetchZeroPointSub<OutD>,
    {
        let zero_point_range = <D as FetchZeroPointSub<OutD>>::ZERO_POINT_RANGE;
        assert!(
            zero_point_range.contains(&zero_point),
            "zero_point {zero_point} is outside the source type's quantized range {zero_point_range:?}",
        );
        FetchZeroPointSubTensor::new(self.ctx, self.inner.map(|v| v.zero_point_sub(zero_point)))
    }
}
// ANCHOR_END: fetch_zero_point_sub_impl

#[allow(clippy::extra_unused_type_parameters)]
fn verify_fetch_mask<Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    todo!("fetch_mask is not yet implemented")
}

#[allow(clippy::extra_unused_type_parameters)]
fn verify_fetch_table_lookup<D: Scalar, OutD: Scalar, Time: M, Packet: M>() {
    todo!("fetch_table_lookup is not yet implemented")
}
