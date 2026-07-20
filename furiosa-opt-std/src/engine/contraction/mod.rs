//! Contraction Engine: Outer -> Packet Reducer -> Time Reducer -> Lane Folder.
//!
//! Submodules:
//! - [`outer`]: stashes the two un-broadcast operands as a [`LazyContraction`] (`.contract_outer`).
//! - [`packet`]: Packet Reducer (`.contract_packet`).
//! - [`time`]: Time Reducer (`.contract_time`).
//! - [`lane`]: Lane Folder (`.contract_lane`).
//!
//! Output: a [`ContractTensor`], at [`PositionContraction`].
//!
//! The engine always defers: a [`LazyContraction`] carrier threads through all four stages untouched and
//! is finalized once, at the Lane Folder, with one fused [`crate::backend::Backend::contraction`].
//! See [`LazyContraction`].

pub mod lane;
pub mod outer;
pub mod packet;
pub mod time;

use std::marker::PhantomData;

pub use lane::LaneMode;
pub use outer::ContractOuterTensor;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// The two un-broadcast operands (already widened to the contraction output type) plus the mappings
/// the Lane Folder needs to fuse them, stashed by the Outer stage in place of a materialized product.
/// This is the only inner value a contraction stage carries: the Packet/Time reducers pass it through
/// untouched, and [`lane::contract_lane`] (via [`crate::backend::Backend::contraction`]) finalizes it
/// with one fused multiply-reduce, allocating zero product cells — so a wide-output GEMM never
/// materializes the hundreds-of-MB `pre_reduce`-shaped outer product.
#[derive(Debug, Clone)]
pub(crate) struct LazyContraction<D: Scalar, B: Backend> {
    pub(crate) lhs: B::Storage<D>,
    pub(crate) rhs: B::Storage<D>,
    /// `lhs`'s own compact layout. [`crate::storage::buf::BufStorage`] reads its strides from this;
    /// [`crate::storage::math::MathStorage`] carries its own axes and ignores it.
    pub(crate) lhs_map: MappingValue,
    /// See `lhs_map`.
    pub(crate) rhs_map: MappingValue,
    /// The full (un-reduced) index space the fold reduces onto the Lane Folder's rebuilt `out`.
    pub(crate) pre_reduce: MappingValue,
}

/// Number of columns in the temporal accumulator buffer, single-sourced from the published verifier crate.
pub(crate) use furiosa_opt_lower::TEMPORAL_ACCUMULATOR_COLS;

/// Asserts a reduce packet's element count is a power of two and at most
/// [`TEMPORAL_ACCUMULATOR_COLS`] (`packet_reduce` / `time_reduce` output).
fn assert_packet_pow2_within_accumulator_cols<Packet: M>() {
    const {
        let size = Packet::SIZE;
        assert!(
            size != 0 && size & (size - 1) == 0 && size <= TEMPORAL_ACCUMULATOR_COLS,
            "Packet element count must be a power of two and at most the accumulator column count (32)"
        );
    };
}

/// After the contraction engine.
#[derive(Debug)]
pub struct PositionContraction;

impl Position for PositionContraction {}

/// Intermediate tensor after the Packet Reducer (reduce-add within `Packet`),
/// before the Time Reducer.
#[derive(Debug)]
pub struct ContractPacketTensor<
    'l,
    const T: Tu,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    /// The deferred carrier (see [`LazyContraction`]); this stage only re-types it to the post-Packet `OutPacket`.
    pub(crate) inner: LazyContraction<D, B>,
    /// Axis typestate the stage transitions carry; `inner` is mapping-free.
    pub(crate) _axes: PhantomData<(Chip, Cluster, Slice, Lane, Time, Packet)>,
}

/// Intermediate tensor after the Time Reducer (accumulator buffer reduce across `Time`),
/// before the Lane Folder.
#[derive(Debug)]
pub struct ContractTimeTensor<
    'l,
    const T: Tu,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    /// The deferred carrier (see [`LazyContraction`]); [`lane::contract_lane`] finalizes it.
    pub(crate) inner: LazyContraction<D, B>,
    /// Pre-reduce `Time` mapping captured by `contract_time`, used by `contract_lane` to
    /// derive the cross-stage `inner_size` accumulator-buffer bound.
    pub(crate) pre_reduce_time: Mapping,
    /// Axis typestate the stage transitions carry; `inner` is mapping-free.
    pub(crate) _axes: PhantomData<(Chip, Cluster, Slice, Lane, Time, Packet)>,
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        assert_packet_pow2_within_accumulator_cols::<Packet>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: LazyContraction<D, B>) -> Self {
        Self::check_constraints();
        Self {
            ctx,
            inner,
            _axes: PhantomData,
        }
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        assert_packet_pow2_within_accumulator_cols::<Packet>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: LazyContraction<D, B>, pre_reduce_time: Mapping) -> Self {
        Self::check_constraints();
        Self {
            ctx,
            inner,
            pre_reduce_time,
            _axes: PhantomData,
        }
    }
}

/// Tensor streamed after the contraction engine.
pub type ContractTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionContraction, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    ContractTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_one_flit::<D, Packet>();
    }

    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}
