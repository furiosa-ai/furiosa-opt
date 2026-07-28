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
//! `fetch_cast` and `fetch_table_lookup` are implemented. `fetch_mask` is a stub whose
//! `verify_fetch_mask()` helper `todo!()`s at runtime.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::{FetchCast, FetchZeroPointSub, TableLookup};
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
// Table lookup is a **main-context-only** hardware feature: only the main Fetch Unit's
// `mode_indirect_table` can point at a resident lookup table, so this impl is fixed to
// `{ Tu::Main }`. The sub-context Fetch Unit (used by the StoTrf weight-staging path) has no
// table-lookup register, so decoding a packed weight must happen in a main-context fetch that
// commits the decoded stream to DM, after which a plain convert-only StoTrf stages it into the TRF.
impl<
    'l,
    P: CanApplyFetchTableLookup,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, { Tu::Main }, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Adapter's table-lookup stage (main context only).
    ///
    /// Each input value indexes a decode table selected by the input type `D`
    /// (via the `TableLookup` trait), not by a runtime argument. The only decode is
    /// `f4e2m1 -> f8e4m3`, the RNGD paired-key 4b->8b table for NVFP4 / MXFP4
    /// weights. Chain a [`fetch_cast`](Self::fetch_cast) to widen `f8e4m3` to
    /// `bf16`/`f32`; the per-block scale is not applied here, it is a separate
    /// downstream VE multiply. See the book chapter
    /// `computing-tensors/fetch-adapter.md` (the "Table Lookup" section).
    ///
    /// Only available on the main context: the lookup table lives in a main Fetch Unit register the
    /// sub context lacks. To feed a decoded weight into a contraction, decode here and commit the
    /// `f8e4m3` stream to DM, then stage that DM tensor into the TRF with a plain convert-only StoTrf.
    #[primitive(TuTensor::fetch_table_lookup)]
    pub fn fetch_table_lookup<OutD: Scalar>(
        self,
    ) -> FetchTableLookupTensor<'l, { Tu::Main }, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: TableLookup<OutD>,
    {
        FetchTableLookupTensor::new(self.ctx, self.inner.map(|v| v.lookup()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Emulation;
    use crate::cast::TableLookup;
    use crate::scalar::{f4e2m1, f8e4m3};
    use crate::tensor::Tensor;

    axes![C = 16];

    /// Spec ground truth: the 16 e2m1 codes in nibble order, as f32. Independent of
    /// the fp8 table under test. Matches the hardware F4E2 -> F32 conversion table.
    const E2M1_F32_ORACLE: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// Pins the per-element decode: all 16 e2m1 codes through the `f4e2m1 -> f8e4m3`
    /// model match the spec. Bit-compares, not `f32 ==`, so the two zero codes are
    /// pinned distinct: `0x0 -> +0.0`, `0x8 -> -0.0`.
    #[test]
    fn table_lookup_decodes_e2m1_to_f8e4m3_matching_spec() {
        for code in 0..16u8 {
            let decoded: f8e4m3 = TableLookup::lookup(f4e2m1::from_bits(code));
            let expected = E2M1_F32_ORACLE[code as usize];
            assert_eq!(
                decoded.to_f32().to_bits(),
                expected.to_bits(),
                "e2m1 code {code:#x} decoded to {} (0x{:08x}), expected {expected} (0x{:08x})",
                decoded.to_f32(),
                decoded.to_f32().to_bits(),
                expected.to_bits(),
            );
        }
    }

    /// Same decode at tensor granularity (the primitive body `input.map(|v| v.lookup())`),
    /// bit-compared so the `-0.0` code is pinned here too.
    #[test]
    fn table_lookup_tensor_map_matches_spec() {
        let keys: Vec<f4e2m1> = (0..16u8).map(f4e2m1::from_bits).collect();
        let input = Tensor::<f4e2m1, m![C], Emulation>::from_vec(keys);
        let decoded: Tensor<f8e4m3, m![C], Emulation> = input.map(|v| v.lookup());
        let got: Vec<u32> = decoded.into_vec().into_iter().map(|v| v.to_f32().to_bits()).collect();
        let want: Vec<u32> = E2M1_F32_ORACLE.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want);
    }

    /// Pins the value-based `PartialEq`: the `Zero` law `a.is_zero() == (a == zero())`
    /// holds for `0x8` (-0.0), not just `0x0`. A raw-nibble derive would break it.
    #[test]
    fn e2m1_zero_law_holds_for_negative_zero() {
        use num_traits::Zero;
        let neg_zero = f4e2m1::from_bits(0x8);
        assert!(neg_zero.is_zero());
        assert_eq!(neg_zero, f4e2m1::zero());
        assert_eq!(neg_zero.is_zero(), neg_zero == f4e2m1::zero());
    }
}
