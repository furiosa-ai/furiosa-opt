//! TRF Sequencer: TRF-side validation for the [`super`] `contract_outer` deferred fold.
//!
//! [`verify_trf_sequencer`] is the TRF-side counterpart of
//! [`super::stream_adapter::verify_stream_adapter`]: it checks that `TrfElement` (broadcast across
//! `Lane`) expands into the joint `OutTime`/`OutPacket` mapping. Still a TODO stub -- the stream side
//! validates, the TRF side does not yet -- kept so that gap stays tracked rather than silently dropped.

use furiosa_mapping::*;

use crate::scalar::Scalar;

/// Validates the TRF-side mapping transformation for the [`super`] `contract_outer`: that `TrfElement`
/// (broadcast across `Lane`) expands into the joint `OutTime`/`OutPacket` mapping, the counterpart of
/// [`super::stream_adapter::verify_stream_adapter`].
#[allow(clippy::extra_unused_type_parameters)] // TODO: parameters are consumed once the check below lands.
pub(super) fn verify_trf_sequencer<D: Scalar, Lane: M, TrfElement: M, OutTime: M, OutPacket: M>() {
    // TODO: validate the TRF-side expansion (TrfElement -> OutTime/OutPacket),
    // mirroring the stream-side checks in `verify_stream_adapter`.
}
