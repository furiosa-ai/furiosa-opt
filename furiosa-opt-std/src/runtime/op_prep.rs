//! Per-op preparation step shared between Simulation and Typecheck.
//!
//! Each helper here is the "prep" that runs before an op's real work — it derives the mapping
//! data the op needs (broadcast residue, reshape quotient, scatter payload, …) and panics on
//! ill-formed mappings along the way. Called from exactly two places per op: the Simulation
//! impl on `MathRawTensor` consumes the returned mapping to drive its iteration loop, and the
//! Typecheck impl on `PhantomRawTensor` discards the value (`let _ = …`) but still triggers
//! the panic side, so type errors surface without the value-level loop ever running. Npu /
//! Emulation never call them — their `BufRawTensor` op bodies are `todo!()`.
//!
//! Most helpers are noun-style and return a derived `FMapping`; `assert_zip` is the one pure
//! check (returns `()`). The shared file lets Simulation and Typecheck stay in lockstep on
//! mapping algebra without duplicating the prep step.
//!
//! [`Backend`]: crate::runtime::backend::Backend
//! [`RawTensor`]: crate::tensor::raw::RawTensor

use abi_stable::std_types::RSlice;
use furiosa_mapping::*;

/// Broadcast residue for `RawTensor::write_transpose`. Returns the broadcast portion of the
/// dst-by-src division; `MathRawTensor`'s write loop iterates over it. Panics on ill-formed
/// mappings.
pub(crate) fn transpose_broadcast<Src: M, Dst: M>(allow_broadcast: bool) -> FMapping {
    let src_mapping = Src::to_value();
    let dst_mapping = Dst::to_value();
    let broadcast = dst_mapping
        .divide(&src_mapping)
        .exact_checked()
        .unwrap_or_else(|e| {
            panic!(
                "[write_transpose] failed to divide mapping expressions for transpose: {e:?}\n\
                 src_mapping: {src_mapping:?}\n\
                 dst_mapping: {dst_mapping:?}"
            )
        })
        .relaxed_residues()
        .dividend_residue;
    if !allow_broadcast {
        assert!(broadcast.is_padding());
    }
    broadcast
}

/// Axes-equality check for `RawTensor::zip_with`.
pub(crate) fn assert_zip(lhs_axes: &[Term], rhs_axes: &[Term]) {
    assert_eq!(
        lhs_axes, rhs_axes,
        "Tensors must have the same axes for element-wise binary operations"
    );
}

/// Broadcast residue for `RawTensor::reduce_then_broadcast`. Panics on ill-formed mappings.
pub(crate) fn reduce_broadcast(src_axes: &[Term], dst_axes: &[Term]) -> FMapping {
    let src_fmapping = FMapping::from_axes(RSlice::from(src_axes));
    let dst_fmapping = FMapping::from_axes(RSlice::from(dst_axes));
    dst_fmapping
        .divide(src_fmapping)
        .exact_checked()
        .expect("Failed to calculate broadcast factor")
        .relaxed_residues()
        .dividend_residue
}

/// Scatter payload mapping and destination axis term. Used by every backend whose
/// `RawTensor::write_scatter` body iterates the host. Panics on ill-formed scatters.
pub(crate) fn scatter_params(src_fmapping: &FMapping, dst_fmapping: &FMapping, key: &FMapping) -> (FMapping, Term) {
    let payload = src_fmapping
        .clone()
        .divide(key.clone())
        .exact_checked()
        .unwrap_or_else(|e| panic!("Scatter key `{key:?}` not found in source `{src_fmapping:?}`: {e:?}"))
        .relaxed_residues()
        .dividend_residue;
    let dst_residue = dst_fmapping
        .clone()
        .divide(payload.clone())
        .exact_checked()
        .unwrap_or_else(|e| panic!("Destination `{dst_fmapping:?}` missing payload axes `{payload:?}`: {e:?}"))
        .relaxed_residues()
        .dividend_residue;
    let (dst_term, _) = dst_residue
        .into_inner()
        .into_iter()
        .find_map(|factor| match factor {
            Factor::Term { inner, resize } => Some((inner, resize)),
            Factor::Padding { .. } => None,
        })
        .expect("Destination has no scatter target axis after removing payload");
    (payload, dst_term)
}

/// Derived parameters for `RawTensor::write_gather`.
///
/// Inverse of [`scatter_params`]:
/// - scatter: `src ÷ key = payload`, `dst ÷ payload = dst_term` (the indexed axis on dst).
/// - gather:  `dst ÷ idx_axes = payload`, `src ÷ payload = src_term` (the indexed axis on src/table).
pub(crate) struct GatherParams {
    /// Axes shared by `src` (the table) and `dst` (the gather output), iterated identically
    /// on both sides. Derived as `dst ÷ idx_axes`.
    pub payload: FMapping,
    /// Axes in `dst` that replace the indexed axis on the source side. The `write_gather` loop
    /// iterates these positions and writes each output slot sequentially. Derived as
    /// `dst ÷ payload`.
    pub idx_residue: FMapping,
    /// Single-term locator on the `src` side identifying the indexed (lookup) axis. The runtime
    /// indices tensor's values index into this axis. Derived as the surviving term in
    /// `src ÷ payload`.
    pub src_term: Term,
}

/// Compute [`GatherParams`] for a gather op. Used by every backend whose
/// `RawTensor::write_gather` body iterates the host. Panics on ill-formed gathers.
///
/// `idx_fmapping` is the full mapping of the indices tensor; the "key-axes-replacement"
/// inside `dst_fmapping` is derived as `dst ÷ payload`.
pub(crate) fn gather_params(src_fmapping: &FMapping, dst_fmapping: &FMapping, idx_fmapping: &FMapping) -> GatherParams {
    let payload = dst_fmapping
        .clone()
        .divide(idx_fmapping.clone())
        .exact_checked()
        .unwrap_or_else(|e| {
            panic!("Gather indices `{idx_fmapping:?}` not contained in destination `{dst_fmapping:?}`: {e:?}")
        })
        .relaxed_residues()
        .dividend_residue;
    let idx_residue = dst_fmapping
        .clone()
        .divide(payload.clone())
        .exact_checked()
        .unwrap_or_else(|e| panic!("Destination `{dst_fmapping:?}` missing payload axes `{payload:?}`: {e:?}"))
        .relaxed_residues()
        .dividend_residue;
    let src_residue = src_fmapping
        .clone()
        .divide(payload.clone())
        .exact_checked()
        .unwrap_or_else(|e| panic!("Source `{src_fmapping:?}` missing payload axes `{payload:?}`: {e:?}"))
        .relaxed_residues()
        .dividend_residue;
    let (src_term, _) = src_residue
        .into_inner()
        .into_iter()
        .find_map(|factor| match factor {
            Factor::Term { inner, resize } => Some((inner, resize)),
            Factor::Padding { .. } => None,
        })
        .expect("Source has no gather target axis after removing payload");
    GatherParams {
        payload,
        idx_residue,
        src_term,
    }
}
