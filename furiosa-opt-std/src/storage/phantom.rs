use std::marker::PhantomData;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::backend::op_prep::{broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::scalar::*;

/// Typecheck tensor: metadata only (axes), no host buffer. All reads return `Opt::Uninit` and all
/// writes are silently dropped.
///
/// Each op runs the relevant `op_prep` helper for its panic side (so mapping errors still surface
/// under Typecheck) and then leaves the (empty) phantom output untouched, no per-element iteration.
#[derive(Clone, Debug)]
pub struct PhantomStorage<D: Scalar> {
    axes: Vec<AxisTerm>,
    _phantom: PhantomData<D>,
}

impl<D: Scalar> PartialEq for PhantomStorage<D> {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes
    }
}
impl<D: Scalar> Eq for PhantomStorage<D> {}

impl<D: Scalar> PhantomStorage<D> {
    /// An empty phantom tensor of the given `Mapping`'s layout, Typecheck carries axes only, no
    /// values.
    fn new<Mapping: M>() -> Self {
        Self::new_from_axes(Mapping::to_value().axes())
    }

    /// An empty phantom tensor carrying `axes` directly. Lets elementwise ops preserve the source
    /// axes without naming a mapping type.
    fn new_from_axes(axes: Vec<AxisTerm>) -> Self {
        Self {
            axes,
            _phantom: PhantomData,
        }
    }

    /// Typecheck carries no values; map/zip produce an empty phantom of the SAME axes, no
    /// per-element iteration. Back [`crate::backend::Backend::map`] / [`crate::backend::Backend::zip_with`].
    pub(crate) fn map<D2: Scalar>(&self, _f: impl FnMut(D) -> D2) -> PhantomStorage<D2> {
        PhantomStorage::new_from_axes(self.axes.clone())
    }

    pub(crate) fn zip_with<D2: Scalar, D3: Scalar>(
        &self,
        _other: &PhantomStorage<D2>,
        _f: impl Fn(D, D2) -> D3,
    ) -> PhantomStorage<D3> {
        PhantomStorage::new_from_axes(self.axes.clone())
    }

    /// Ternary peer of [`Self::zip_with`]; phantom carries no values.
    pub(crate) fn zip3_with<D2: Scalar, D3: Scalar, D4: Scalar>(
        &self,
        _b: &PhantomStorage<D2>,
        _c: &PhantomStorage<D3>,
        _f: impl Fn(D, D2, D3) -> D4,
    ) -> PhantomStorage<D4> {
        PhantomStorage::new_from_axes(self.axes.clone())
    }

    /// Typecheck no-op: runs only the structural check, leaving the empty phantom `self` untouched.
    /// Backs [`crate::backend::Backend::transpose`].
    pub(crate) fn transpose<Src: M, Mapping: M>(
        &mut self,
        _src: &PhantomStorage<D>,
        _src_offset: &Index,
        _dst_offset: &Index,
        allow_broadcast: bool,
    ) {
        let _ = transpose_broadcast::<Src, Mapping>(allow_broadcast);
    }

    /// Backs [`crate::backend::Backend::reduce`].
    pub(crate) fn reduce<Src: M, Dst: M, R: Fn(D, D) -> D>(
        &self,
        _reduce_fn: R,
        _identity: D,
        allow_broadcast: bool,
    ) -> Self {
        // Structural parity check with the value backends (`BufStorage::reduce`): reject an
        // unintended broadcast, then run the same containment carves so a non-contained / over-padded
        // reduce is rejected at type-check time exactly as it is at runtime. The carves panic on
        // a mismatch; their offsets are unused here since Typecheck moves no values.
        let src = Src::to_value();
        let dst = Dst::to_value();
        let broadcast = broadcast_axes(&src, &dst);
        assert!(
            allow_broadcast || broadcast.axes().is_empty(),
            "reduce: Dst adds axes absent from the source; pass allow_broadcast=true for reduce-then-broadcast"
        );
        let inter = dst.carve(&broadcast);
        let _reduce_residue = src.carve(&inter);
        Self::new::<Dst>()
    }

    /// Backs [`crate::backend::Backend::contraction`]. Typecheck carries no values, so it neither
    /// widens nor folds; `out` is a runtime value here (the deferred engine-level fold only has it as
    /// one), so it is taken directly rather than rebuilt from a static mapping type.
    pub(crate) fn contraction(&self, _rhs: &Self, out: &MappingValue) -> Self {
        Self::new_from_axes(out.axes())
    }

    /// Backs [`crate::backend::Backend::scatter`].
    // `Idx` is unused here (Typecheck decodes no indices) but must match the `Backend::scatter`
    // signature that `BufStorage` needs (`Idx::SIZE`).
    #[expect(
        clippy::extra_unused_type_parameters,
        reason = "signature parity with Backend::scatter"
    )]
    pub(crate) fn scatter<Src: M, Key: M, Dst: M, Idx: M>(
        &self,
        _dst: &mut PhantomStorage<D>,
        _index: &PhantomStorage<i32>,
        _scaled: bool,
    ) {
        let _ = scatter_params(&Src::to_value(), &Dst::to_value(), &Key::to_value());
    }

    /// Backs [`crate::backend::Backend::gather`].
    pub(crate) fn gather<Src: M, Dst: M, Idx: M>(
        &self,
        _dst: &mut PhantomStorage<D>,
        _index: &PhantomStorage<i32>,
        _scaled: bool,
    ) {
        let _ = gather_params(&Src::to_value(), &Dst::to_value(), &Idx::to_value());
    }

    /// Backs [`crate::backend::Backend::reshape`].
    pub(crate) fn reshape<Mapping: M, Mapping2: M>(&self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        Self::new::<Mapping2>()
    }

    /// Typecheck has no values, the input buffer is dropped regardless of its length. (Length is
    /// still validated by [`crate::tensor::Tensor::from_vec`] in the wrapper, unlike the `Opt` path.)
    pub(crate) fn from_vec(mapping: &MappingValue, _data: impl IntoIterator<Item = D>) -> Self {
        Self::new_from_axes(mapping.axes())
    }

    /// Typecheck carries no values.
    pub(crate) fn into_vec(self, _mapping: &MappingValue) -> Vec<D> {
        Vec::new()
    }

    /// An uninitialized (empty) phantom tensor of this layout.
    pub(crate) fn uninit(mapping: &MappingValue) -> Self {
        Self::new_from_axes(mapping.axes())
    }
}
