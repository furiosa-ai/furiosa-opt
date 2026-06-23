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
//! Most helpers are noun-style and return a derived `Mapping`; `assert_zip` is the one pure
//! check (returns `()`). The shared file lets Simulation and Typecheck stay in lockstep on
//! mapping algebra without duplicating the prep step.
//!
//! [`Backend`]: crate::runtime::backend::Backend
//! [`RawTensor`]: crate::tensor::raw::RawTensor

use std::collections::HashSet;

use furiosa_mapping::*;

/// The broadcast portion of `dst` for `RawTensor::write_transpose`, the part `src` does not cover;
/// `MathRawTensor`'s write loop iterates over it. Panics on ill-formed mappings.
pub(crate) fn transpose_broadcast<Src: M, Dst: M>(allow_broadcast: bool) -> Mapping {
    let src_mapping = Src::to_value();
    let dst_mapping = Dst::to_value();
    // Carve `src` out of `dst` via the matcher; the leftover is the broadcast portion.
    let broadcast = dst_mapping.carve(&src_mapping);
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
pub(crate) fn reduce_broadcast(src_axes: &[Term], dst_axes: &[Term]) -> Mapping {
    Mapping::from_terms(dst_axes.iter().cloned()).carve(&Mapping::from_terms(src_axes.iter().cloned()))
}

/// The broadcast portion of `dst` for `reduce_then_broadcast`: the axes built from symbols absent
/// in `src` (a symbol-level split, since broadcast axes are always new).
pub(crate) fn broadcast_axes(src: &Mapping, dst: &Mapping) -> Mapping {
    let src_idents: HashSet<Ident> = src.idents().into_iter().collect();
    let dst_axes = dst.axes();
    let mut ids = Vec::new();
    // The symbol-level filter is exact only when each dst axis is WHOLLY new or WHOLLY shared.
    // A dst axis straddling both (a shared symbol partially split into a broadcast factor) would
    // split silently; pin that this does not happen.
    debug_assert!(
        dst_axes.iter().all(|term| {
            collect_term_idents(term, &mut ids);
            let in_src = ids.iter().filter(|i| src_idents.contains(i)).count();
            in_src == 0 || in_src == ids.len()
        }),
        "broadcast_axes assumes each dst axis is wholly new or wholly shared, not a straddling split"
    );
    let axes: Vec<Term> = dst_axes
        .into_iter()
        .filter(|term| {
            collect_term_idents(term, &mut ids);
            ids.iter().all(|ident| !src_idents.contains(ident))
        })
        .collect();
    Mapping::from_terms(axes)
}

/// Collects `term`'s idents into `out` (cleared first), so the caller reuses one buffer.
fn collect_term_idents(term: &Term, out: &mut Vec<Ident>) {
    out.clear();
    match &term.inner {
        Atom::Symbol { symbol, .. } => out.push(*symbol),
        Atom::Composite(inner) => out.extend(inner.idents()),
    }
}

/// Scatter payload mapping and destination axis term. Used by every backend whose
/// `RawTensor::write_scatter` body iterates the host. Panics on ill-formed scatters.
pub(crate) fn scatter_params(src: &Mapping, dst: &Mapping, key: &Mapping) -> (Mapping, Term) {
    let payload = src.carve(key);
    let dst_term = dst
        .carve(&payload)
        .axes()
        .into_iter()
        .next()
        .expect("scatter dst residue has no live target axis");
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
    pub payload: Mapping,
    /// Axes in `dst` that replace the indexed axis on the source side. The `write_gather` loop
    /// iterates these positions and writes each output slot sequentially. Derived as
    /// `dst ÷ payload`.
    pub idx_residue: Mapping,
    /// Single-term locator on the `src` side identifying the indexed (lookup) axis. The runtime
    /// indices tensor's values index into this axis. Derived as the surviving term in
    /// `src ÷ payload`.
    pub src_term: Term,
}

/// Compute [`GatherParams`] for a gather op. Used by every backend whose
/// `RawTensor::write_gather` body iterates the host. Panics on ill-formed gathers.
///
/// `idx` is the full mapping of the indices tensor; the "key-axes-replacement" inside `dst` is
/// derived as `dst ÷ payload`.
pub(crate) fn gather_params(src: &Mapping, dst: &Mapping, idx: &Mapping) -> GatherParams {
    let payload = dst.carve(idx);
    let idx_residue = dst.carve(&payload);
    let src_term = src
        .carve(&payload)
        .axes()
        .into_iter()
        .next()
        .expect("gather src residue has no live target axis");
    GatherParams {
        payload,
        idx_residue,
        src_term,
    }
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::broadcast_axes;

    axes![A = 4, B = 2, C = 8];

    /// `broadcast_axes` is the symbol-level narrowing of `dst.carve(src)`: it keeps the SAME live axes
    /// but drops the positional padding carve adds. Where its invariant holds (each dst axis wholly new
    /// or wholly shared), its axes must equal carve's live axes.
    #[test]
    fn broadcast_axes_matches_carve() {
        let cases: [(Mapping, Mapping); 4] = [
            (<m![A, C]>::to_value(), <m![A, B, C]>::to_value()), // B is the new (broadcast) axis
            (<m![A]>::to_value(), <m![A, B]>::to_value()),
            (<m![B]>::to_value(), <m![A, B, C]>::to_value()), // two new axes (A, C)
            (<m![A, B, C]>::to_value(), <m![A, B, C]>::to_value()), // no broadcast
        ];
        let sorted_axes = |m: &Mapping| {
            let mut a = m.axes();
            a.sort();
            a
        };
        for (src, dst) in cases {
            assert_eq!(
                sorted_axes(&broadcast_axes(&src, &dst)),
                sorted_axes(&dst.carve(&src)),
                "broadcast_axes(src={src:?}, dst={dst:?})"
            );
        }
    }
}
