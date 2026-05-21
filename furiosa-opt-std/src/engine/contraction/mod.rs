//! Contraction Engine: Outer -> Packet Reducer -> Time Reducer -> Lane Folder.
//!
//! Submodules:
//! - [`outer`]: prepares operands in the joint computation mapping
//!   (`.contract_outer`).
//! - [`packet`]: Packet Reducer (`.contract_packet`).
//! - [`time`]: Time Reducer (`.contract_time`).
//! - [`lane`]: Lane Folder (`.contract_lane`).
//!
//! Output: a [`ContractTensor`], at [`PositionContraction`].

pub mod lane;
pub mod outer;
pub mod packet;
pub mod time;

pub use lane::LaneMode;
pub use outer::ContractOuterTensor;

use furiosa_mapping::*;

use crate::context::*;
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// Number of columns in the temporal accumulator buffer.
pub(crate) const TEMPORAL_ACCUMULATOR_COLS: usize = 32;

/// Number of elements in the `contract_lane` output packet (`i32`/`f32`).
pub(crate) const CONTRACT_LANE_OUT_PACKET_ELEMENTS: usize = 8;

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
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
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
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
    /// Pre-reduce `Time` mapping captured by `contract_time`, used by `contract_lane` to
    /// derive the cross-stage `inner_size` accumulator-buffer bound.
    pub(crate) pre_reduce_time: FMapping,
}

/// Tensor streamed after the contraction engine.
pub type ContractTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionContraction, D, Chip, Cluster, Slice, Time, Packet, B>;
