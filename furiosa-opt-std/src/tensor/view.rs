use std::marker::PhantomData;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;
use furiosa_opt_lower::config_tile;

use super::Tensor;
use crate::backend::Backend;
use crate::runtime::CurrentBackend;
use crate::scalar::*;

/// Mutable view into a tensor. Borrows the concrete storage of some base tensor; the element
/// mapping `Mapping` is the view's *current* logical layout (changed by [`Self::tile`]), tracked
/// at the view's `_marker` (the borrowed `B::Storage<D>` is mapping-agnostic after the storage-layer
/// mapping erasure, so retiling is a plain rewrap of the same borrow, no cast).
pub struct TensorViewMut<'l, D: Scalar, Mapping: M, B: Backend = CurrentBackend> {
    inner: &'l mut B::Storage<D>,
    offset: Index,
    // The base tensor's live-axis mapping, set at construction and left untouched by `tile` (which
    // only narrows the `Mapping` type param). A tile records its axis in `offset` as a live `Symbol`
    // but represents it in `Mapping` as padding; `base_map` keeps the axis live so a partial-view
    // relayout can resolve the offset's physical wire base against the real layout (see
    // [`crate::backend::Backend::transpose`]).
    base_map: MappingValue,
    _marker: PhantomData<(Mapping, B)>,
}

impl<'l, D: Scalar, Mapping: M, B: Backend> std::fmt::Debug for TensorViewMut<'l, D, Mapping, B>
where
    B::Storage<D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorViewMut")
            .field("inner", &self.inner)
            .field("offset", &self.offset)
            .finish()
    }
}

/// Immutable view into a tensor.
pub struct TensorView<'l, D: Scalar, Mapping: M, B: Backend = CurrentBackend> {
    inner: &'l B::Storage<D>,
    offset: Index,
    /// The base tensor's live-axis mapping (see [`TensorViewMut::base_map`]).
    base_map: MappingValue,
    _marker: PhantomData<(Mapping, B)>,
}

impl<'l, D: Scalar, Mapping: M, B: Backend> std::fmt::Debug for TensorView<'l, D, Mapping, B>
where
    B::Storage<D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("inner", &self.inner)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'l, D: Scalar, Mapping: M, B: Backend> Clone for TensorView<'l, D, Mapping, B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            offset: self.offset.clone(),
            base_map: self.base_map.clone(),
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, Mapping: M, B: Backend> From<TensorViewMut<'l, D, Mapping, B>> for TensorView<'l, D, Mapping, B> {
    fn from(view: TensorViewMut<'l, D, Mapping, B>) -> Self {
        Self {
            inner: view.inner,
            offset: view.offset,
            base_map: view.base_map,
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorViewMut<'l, D, E, B> {
    /// Creates a new tensor view mut.
    pub(crate) fn new(inner: &'l mut B::Storage<D>) -> Self {
        Self {
            inner,
            offset: Index::new(),
            base_map: E::to_value(),
            _marker: PhantomData,
        }
    }

    /// Splits the tensor view by tiling. As a write destination, the cells outside
    /// the tile must be down padding ([`PaddingKind::Bottom`]) so the commit
    /// sequencer never writes them.
    pub fn tile<I: M, E2: M, const LEN: usize>(self, start: usize) -> TensorViewMut<'l, D, E2, B> {
        config_tile(
            &I::to_value(),
            &E::to_value(),
            &E2::to_value(),
            LEN,
            PaddingKind::Bottom,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut offset = self.offset;
        offset.add_mapping::<I>(start);
        // Retiling is a plain rewrap (see `base_map`'s doc): same-type inner borrow, new mapping `E2`
        // recorded only at `_marker`, `base_map` left as the base tensor's live-axis mapping.
        TensorViewMut {
            inner: self.inner,
            offset,
            base_map: self.base_map,
            _marker: PhantomData,
        }
    }

    /// Reshapes the view to a different mapping `E2` over the same borrow, consuming `self`. A reshape
    /// is a MOVE: a rewrap of the same mapping-agnostic storage borrow (no data copied, no second
    /// handle), only the type-level mapping at `_marker` changes. Unlike [`Self::tile`] it neither
    /// shifts `offset` nor validates a division; `offset` and `base_map` ride through unchanged.
    /// Sound on every backend, whose storage is mapping-agnostic (bare buffer / codegen relabel).
    ///
    /// # Safety
    ///
    /// `E::SIZE == E2::SIZE` is asserted below; the genuine precondition is that `E` and `E2` lay the
    /// elements out in the SAME physical (wire) order, so the relabel moves no data. Axis regrouping
    /// (merge/split) preserves wire order and is valid; a permutation is not (use [`Self::transpose`]).
    /// `base_map` is left at the base tensor's live-axis mapping, so on a tiled (non-empty `offset`)
    /// view the relabel must preserve which axes are live for a later partial-view relayout to
    /// resolve.
    ///
    /// Kept a *runtime* `assert_eq!`, not a `const` block: every
    /// `{Dm,Hbm}TensorView(Mut)::reshape` delegates its `Element`-inclusive combined mapping here (see
    /// [`Tensor::reshape`]'s matching note -- a `const` block trips at monomorphization regardless of
    /// whether the call is ever reached at runtime).
    pub unsafe fn reshape<E2: M>(self) -> TensorViewMut<'l, D, E2, B> {
        assert_eq!(E::SIZE, E2::SIZE);
        TensorViewMut {
            inner: self.inner,
            offset: self.offset,
            base_map: self.base_map,
            _marker: PhantomData,
        }
    }

    /// Transposes from a tensor. Delegates to [`Backend::transpose`], passing both storages and their
    /// live-axis maps (`base_map`) so a partial-view relayout can resolve each offset's wire base.
    pub fn transpose<'lsrc, Src: M>(&mut self, src: TensorView<'lsrc, D, Src, B>, allow_broadcast: bool) {
        B::transpose::<D, Src, E>(
            self.inner,
            src.inner,
            &src.offset,
            &self.offset,
            &src.base_map,
            &self.base_map,
            allow_broadcast,
        );
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Splits the tensor view by tiling. As a read source, the cells outside the
    /// tile stay accessible ([`PaddingKind::Top`]).
    pub fn tile<I: M, E2: M, const LEN: usize>(&self, start: usize) -> TensorView<'l, D, E2, B> {
        config_tile(&I::to_value(), &E::to_value(), &E2::to_value(), LEN, PaddingKind::Top)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut offset = self.offset.clone();
        offset.add_mapping::<I>(start);
        // Retiling is a plain rewrap (see `base_map`'s doc): same-type inner borrow, new mapping `E2`
        // recorded only at `_marker`, `base_map` left as the base tensor's live-axis mapping.
        TensorView {
            inner: self.inner,
            offset,
            base_map: self.base_map.clone(),
            _marker: PhantomData,
        }
    }

    /// Reshapes the view to a different mapping `E2` over the same borrow, consuming `self`. A reshape
    /// is a MOVE: a rewrap of the same mapping-agnostic storage borrow (no data copied, no second
    /// handle), only the type-level mapping at `_marker` changes. Unlike [`Self::tile`] it neither
    /// shifts `offset` nor validates a division; `offset` and `base_map` ride through unchanged.
    /// Sound on every backend, whose storage is mapping-agnostic (bare buffer / codegen relabel).
    ///
    /// # Safety
    ///
    /// `E::SIZE == E2::SIZE` is asserted below; the genuine precondition is that `E` and `E2` lay the
    /// elements out in the SAME physical (wire) order, so the relabel moves no data. Axis regrouping
    /// (merge/split) preserves wire order and is valid; a permutation is not (use `transpose`).
    /// `base_map` is left at the base tensor's live-axis mapping, so on a tiled (non-empty `offset`)
    /// view the relabel must preserve which axes are live for a later partial-view relayout to
    /// resolve.
    ///
    /// Kept a *runtime* `assert_eq!`, not a `const` block: every
    /// `{Dm,Hbm}TensorView(Mut)::reshape` delegates its `Element`-inclusive combined mapping here (see
    /// [`Tensor::reshape`]'s matching note -- a `const` block trips at monomorphization regardless of
    /// whether the call is ever reached at runtime).
    pub unsafe fn reshape<E2: M>(self) -> TensorView<'l, D, E2, B> {
        assert_eq!(E::SIZE, E2::SIZE);
        TensorView {
            inner: self.inner,
            offset: self.offset,
            base_map: self.base_map,
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Reads the tensor view into a new tensor. Delegates to [`Backend::transpose`],
    /// whose body runs `transpose_broadcast` on every storage (so Typecheck still validates).
    pub fn read(self) -> Tensor<D, E, B> {
        let mut result = Tensor::uninit();
        result.view_mut().transpose(self, false);
        result
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Creates a new tensor view.
    pub(crate) fn new(inner: &'l B::Storage<D>) -> Self {
        Self {
            inner,
            offset: Index::new(),
            base_map: E::to_value(),
            _marker: PhantomData,
        }
    }
}
