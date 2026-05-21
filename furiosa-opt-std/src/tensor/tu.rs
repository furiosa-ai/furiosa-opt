//! Tensor Unit stream typestate primitive.
//!
//! Defines [`TuTensor<P>`], the in-motion tensor type whose [`Position`]
//! marker `P` tracks the Tensor Unit pipeline stage at compile time.
//!
//! Each engine ([`crate::engine`]) owns the typestate it *produces*: its
//! [`Position`] marker, its `XTensor` type alias, and the transition methods
//! that consume the source typestate. The exception is the pipeline entry
//! point ([`PositionBegin`] / [`BeginTensor`]), which is not produced by any
//! engine and lives here.
//!
//! `PositionVectorFinal` / `VectorFinalTensor` live in
//! [`crate::engine::vector::tensor`], produced by the Vector Engine.

use std::marker::PhantomData;

use furiosa_mapping::*;

use crate::context::*;
use crate::runtime::{Backend, CurrentBackend};
use crate::scalar::*;
use crate::tensor::Tensor;

/// Marker trait for pipeline position of Tensor Unit tensors.
///
/// Position does not contain Vector Engine position: VectorTensor has its own typestate.
pub trait Position: std::fmt::Debug + 'static {}

/// After beginning the pipeline.
#[derive(Debug)]
pub struct PositionBegin;

impl Position for PositionBegin {}

/// Tensor streamed through the Tensor Unit pipeline.
#[derive(Debug)]
pub struct TuTensor<
    'l,
    const T: Tu,
    P: Position,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Packet>>>>, B>,
    _position: PhantomData<P>,
}

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

    /// Creates a new Tensor Unit tensor.
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

/// Tensor streamed after the beginning.
pub type BeginTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionBegin, D, Chip, Cluster, Slice, Time, Packet, B>;
