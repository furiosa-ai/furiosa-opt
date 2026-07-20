//! Per-op preparation step shared across the backends.
//!
//! Each helper derives the mapping data an op needs (broadcast residue, reshape quotient, scatter
//! payload, ...) and panics on ill-formed mappings. Emulation (`BufStorage`) consumes the
//! returned mapping to drive its loops; Typecheck (`PhantomStorage`) discards the value
//! (`let _ = ...`) but still triggers the panic side, so type errors surface without any
//! value-level loop running.
//!
//! All helpers are noun-style and return a derived `Mapping` (or its parts). The shared file keeps
//! the backends in lockstep on the mapping algebra without duplicating the prep step.
//!
//! [`Backend`]: crate::backend::Backend
//! [`Tensor`]: crate::tensor::Tensor

use std::collections::HashSet;

use furiosa_mapping::*;

/// The broadcast portion of `dst` for `Tensor::transpose`, the part `src` does not cover;
/// `BufStorage`'s write loop iterates over it. Panics on ill-formed mappings.
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

/// The broadcast portion of `dst` for `reduce`: the axes whose symbol is absent in `src` (broadcast
/// axes are always new). Each [`AxisTerm`] is a single resolved symbol, so a dst axis is wholly new or
/// wholly shared, there is no straddling split (a composite spanning both) to guard against.
pub(crate) fn broadcast_axes(src: &Mapping, dst: &Mapping) -> Mapping {
    let src_idents: HashSet<Ident> = src.idents().into_iter().collect();
    let axes: Vec<Term> = dst
        .axes()
        .into_iter()
        .filter(|term| !src_idents.contains(&term.symbol))
        .map(AxisTerm::to_term)
        .collect();
    Mapping::from_terms(axes)
}

/// Scatter payload mapping and destination axis term. Used by every backend whose
/// `Tensor::scatter` body iterates the host. Panics on ill-formed scatters.
pub(crate) fn scatter_params(src: &Mapping, dst: &Mapping, key: &Mapping) -> (Mapping, AxisTerm) {
    let payload = src.carve(key);
    let dst_term = dst
        .carve(&payload)
        .axes()
        .into_iter()
        .next()
        .expect("scatter dst residue has no live target axis");
    (payload, dst_term)
}

/// Derived parameters for `Tensor::gather`.
///
/// Inverse of [`scatter_params`]:
/// - scatter: `src ÷ key = payload`, `dst ÷ payload = dst_term` (the indexed axis on dst).
/// - gather:  `dst ÷ idx_axes = payload`, `src ÷ payload = src_term` (the indexed axis on src/table).
pub(crate) struct GatherParams {
    /// Axes shared by `src` (the table) and `dst` (the gather output), iterated identically
    /// on both sides. Derived as `dst ÷ idx_axes`.
    pub payload: Mapping,
    /// Single-term locator on the `src` side identifying the indexed (lookup) axis. The runtime
    /// indices tensor's values index into this axis. Derived as the surviving term in
    /// `src ÷ payload`.
    pub src_term: AxisTerm,
}

/// Compute [`GatherParams`] for a gather op. Used by every backend whose
/// `Tensor::gather` body iterates the host. Panics on ill-formed gathers.
///
/// `idx` is the full mapping of the indices tensor; the "key-axes-replacement" inside `dst` is
/// derived as `dst ÷ payload`. `BufStorage::gather` (the sole caller) derives that residue from its
/// own `Idx` type parameter instead, so the result here is discarded (`let _ = ...`); `carve` still
/// runs for its ill-formed-mapping panic, the same pattern the module doc describes for Typecheck.
pub(crate) fn gather_params(src: &Mapping, dst: &Mapping, idx: &Mapping) -> GatherParams {
    let payload = dst.carve(idx);
    let _ = dst.carve(&payload);
    let src_term = src
        .carve(&payload)
        .axes()
        .into_iter()
        .next()
        .expect("gather src residue has no live target axis");
    GatherParams { payload, src_term }
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
