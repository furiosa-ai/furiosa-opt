//! Outer stage: the entry to the ONE deferred contraction algorithm.
//!
//! `contract_outer` validates both operand sides ([`stream_adapter::verify_stream_adapter`],
//! [`trf_sequencer::verify_trf_sequencer`]) and stashes the two un-broadcast operands (widened to the
//! contraction output type) plus their layouts in a [`super::LazyContraction`], WITHOUT materializing the
//! `[.., Lane, ..]`-shaped outer product. The Packet/Time reducers thread it through unchanged, and
//! [`super::lane`] finalizes it with one fused [`crate::backend::Backend::contraction`].
//!
//! The widen happens *entering* Outer: `contract_outer` takes storage-typed operands `D` and stashes
//! them cast to the widened accumulator [`ContractionCast::Output`]. That widened type is the carrier's
//! *own* type parameter, so every downstream stage carries the accumulator width verbatim; the narrow
//! back to a storage dtype happens at the next [`crate::engine::cast::CastTensor`] the consumer applies.

pub(super) mod stream_adapter;
pub(super) mod trf_sequencer;

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::{Cast, ContractionCast, ContractionWeight};
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplyContractOuter;
use crate::engine::contraction::LazyContraction;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::memory::TrfTensor;
use crate::tensor::tu::TuTensor;

/// Output of the Outer stage: the deferred contraction carrier (the two un-broadcast operands, widened
/// to the accumulator type), threaded through the reducers and fused at the Lane Folder.
///
/// Two dtype params, related by the `Storage: ContractionCast<Output = D>` bound that pins them:
/// - `D` is the *widened accumulator* the operands are carried in (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`).
/// - `Storage` is the pre-widen operand dtype, one the engine can multiply ([`ContractionCast`]). It is
///   retained only so the Packet Reducer's physical-flit
///   check ([`super::packet::verify_contract_packet`]) can size the DPE *input* packet in storage bytes,
///   which the widened `D` no longer encodes (`Storage` is not recoverable from `D` — the widen is
///   many-to-one, so this second param is load-bearing, not redundant).
///
/// See the module docs for where the widen and the (downstream) narrow happen.
#[derive(Debug)]
pub struct ContractOuterTensor<
    'l,
    const T: Tu,
    D: Scalar,
    Storage: ContractionCast<Output = D>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    /// The deferred dense-GEMM carrier: the two compact operands (already widened to the contraction
    /// output type) plus the mappings [`super::lane::contract_lane`] fuses them with. `D` is already
    /// the widened accumulator (`Storage: ContractionCast<Output = D>`), so the carrier is keyed on `D`.
    pub(crate) inner: LazyContraction<D, B>,
    /// The pre-widen storage dtype, retained for the Packet Reducer's flit check (see struct doc).
    pub(crate) _storage: PhantomData<Storage>,
    /// Axis typestate the stage transitions carry; `inner` is mapping-free ([`LazyContraction`]).
    pub(crate) _axes: PhantomData<(Chip, Cluster, Slice, Lane, Time, Packet)>,
}

impl<
    'l,
    const T: Tu,
    D: Scalar,
    Storage: ContractionCast<Output = D>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractOuterTensor<'l, T, D, Storage, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        // The DPE reads storage-width input flits, then widens; size the packet in `Storage` bytes.
        constraints::assert_packet_one_or_two_flit::<Storage, Packet>();
    }

    #[doc(hidden)]
    pub(crate) fn new(ctx: &'l mut TuContext<{ T }>, inner: LazyContraction<D, B>) -> Self {
        Self::check_constraints();
        Self {
            ctx,
            inner,
            _storage: PhantomData,
            _axes: PhantomData,
        }
    }
}

// ANCHOR: contract_outer_def
impl<
    'l,
    const T: Tu,
    P: CanApplyContractOuter,
    D: Scalar + ContractionCast,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Outer stage: stashes the two un-broadcast operands (widened to the accumulator type)
    /// and the layouts [`super::ContractTimeTensor::contract_lane`] needs to fuse them into a [`LazyContraction`]. No
    /// materializing alternative, no per-backend branch -- every backend fuses the same way.
    ///
    /// The impl block binds [`ContractionCast`], so this method does not exist on a `TuTensor` whose
    /// element type the Multiplier has no MAC for.
    #[primitive(TuTensor::contract_outer)]
    pub fn contract_outer<OutTime: M, OutPacket: M, Lane: M, TrfElement: M, TrfD>(
        self,
        trf_tensor: &TrfTensor<TrfD, Chip, Cluster, Slice, Lane, TrfElement, B>,
    ) -> ContractOuterTensor<'l, T, <D as ContractionCast>::Output, D, Chip, Cluster, Slice, Lane, OutTime, OutPacket, B>
    where
        D: Cast<<D as ContractionCast>::Output>,
        // The weight (TRF) type must form a valid contraction-engine operand pair with the
        // stream type `D`: same type, or a mixed integer precision within a
        // family (i4/i5 x i4/i5, i8/i9 x i8/i9). Both operands widen to the
        // stream's accumulator for the multiply.
        TrfD: Scalar + ContractionWeight<D> + Cast<<D as ContractionCast>::Output>,
    {
        type Out<D> = <D as ContractionCast>::Output;

        // Skipping the broadcast transpose does not skip its validity contract -- a malformed
        // contraction would otherwise slip past these asserts and panic far downstream instead.
        stream_adapter::verify_stream_adapter::<D, Lane, Time, Packet, OutTime, OutPacket>();
        trf_sequencer::verify_trf_sequencer::<TrfD, Lane, TrfElement, OutTime, OutPacket>();

        // The operands keep their own compact layouts: lhs is `[Chip, Cluster, Slice, Time, Packet]`
        // (`self.inner`), rhs is `[Chip, Cluster, Slice, Lane, TrfElement]` (`trf_tensor`). A
        // bare-buffer backend reads its strides from these; `MathStorage` ignores them (its axes live
        // in the storage).
        let lhs_map = <m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }]>::to_value();
        let rhs_map = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { TrfElement }]>::to_value();
        let pre_reduce = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { OutTime }, { OutPacket }]>::to_value();

        // Widen each operand to the `Out<D>` accumulator up front, at the operand's own (compact) size,
        // not pre_reduce's -- the fold at `contract_lane` then runs entirely in `Out<D>`. Parity-identical
        // to the old per-cell widen (same `Cast::cast` per element), it just never allocates the
        // pre_reduce-shaped broadcast this stage used to build.
        //
        let contraction = LazyContraction {
            lhs: self.inner.map(|v| -> Out<D> { v.cast() }).inner,
            rhs: trf_tensor.inner.map(|v| -> Out<D> { v.cast() }).inner,
            lhs_map,
            rhs_map,
            pre_reduce,
        };
        ContractOuterTensor::new(self.ctx, contraction)
    }
}
// ANCHOR_END: contract_outer_def
