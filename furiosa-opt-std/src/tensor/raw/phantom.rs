use std::marker::PhantomData;

use furiosa_mapping::{DivisionExt, Index, M, MappingExt, Term};
use ndarray::IxDyn;

use crate::engine::vector::operand::OperandTag;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::op_prep::{assert_zip, gather_params, reduce_broadcast, scatter_params, transpose_broadcast};
use crate::scalar::{Opt, Scalar};
use crate::tensor::BufferConvertError;
use crate::tensor::raw::{RawTensor, RawTensorOpt, gen_axes};

/// Phantom raw tensor (Typecheck): metadata only (axes), no host buffer. All reads return
/// `Opt::Uninit` and all writes are silently dropped.
///
/// Each [`RawTensor`] method runs the relevant `op_prep` helper for its panic side
/// (so mapping errors still surface under Typecheck) and then returns a fresh phantom tensor —
/// no per-element iteration. The Backend trait already routes value-producing ops through its
/// own typecheck overrides; the no-op writes only matter when user code reaches `RawTensor`
/// directly through `Tensor` inherent methods (e.g. `Tensor::reshape`, `view.write_transpose`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct PhantomRawTensor<D: Scalar> {
    axes: Vec<Term>,
    _phantom: PhantomData<D>,
}

impl<D: Scalar> RawTensor<D> for PhantomRawTensor<D> {
    fn axes(&self) -> &[Term] {
        &self.axes
    }

    fn uninit_from_axes(axes: Vec<Term>) -> Self {
        Self {
            axes,
            _phantom: PhantomData,
        }
    }

    fn read_index(&self, _index: Index) -> Opt<D> {
        Opt::Uninit
    }

    fn write_index(&mut self, _index: Index, _value: Opt<D>) {}

    fn from_buf<Mapping: M>(_data: impl IntoIterator<Item = D>) -> Self {
        Self::uninit_from_axes(gen_axes::<Mapping>())
    }

    /// Typecheck has no values — the input buffer is dropped on the floor regardless of its
    /// length, so skip the trait's default length check (which would fail for any caller writing
    /// `Tensor::<_, M, Typecheck>::try_from_buf(vec![])` to construct a phantom tensor without
    /// caring about size). Mapping errors still surface elsewhere via the Typecheck Backend
    /// overrides; this method's job is only buffer ingestion.
    fn try_from_buf<Mapping: M>(data: impl IntoIterator<Item = D>) -> Result<Self, BufferConvertError> {
        Ok(Self::from_buf::<Mapping>(data))
    }

    fn to_buf<Mapping: M>(&self) -> Vec<D> {
        Vec::new()
    }

    fn from_fn<F>(axes: Vec<Term>, _f: F) -> Self
    where
        F: FnMut(&Vec<Term>, &IxDyn) -> Opt<D>,
    {
        Self::uninit_from_axes(axes)
    }

    fn map<D2: Scalar, Output: RawTensor<D2>, F>(&self, _f: F) -> Output
    where
        F: FnMut(&Opt<D>) -> Opt<D2>,
    {
        Output::uninit_from_axes(self.axes.clone())
    }

    fn reduce<Src: M, Dst: M, Reduce>(&self, _reduce_fn: Reduce, _identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        // Run the structural check for parity with `MathRawTensor`, then return the Dst-shaped
        // axes (no values to compute).
        let _ = Src::to_value()
            .divide(&Dst::to_value())
            .exact_checked()
            .unwrap_or_else(|e| {
                panic!(
                    "[reduce] Dst is not a factor of Src: {e:?}\n\
                     Src: {:?}\n\
                     Dst: {:?}",
                    Src::to_value(),
                    Dst::to_value()
                )
            });
        Self::uninit_from_axes(gen_axes::<Dst>())
    }

    fn reduce_then_broadcast<Src: M, Dst: M, Reduce>(&self, _reduce_fn: Reduce, _identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        let dst_axes = gen_axes::<Dst>();
        // Filter own axes against Dst (mirrors `reduce`'s axis projection) so the broadcast
        // residue is computed against the same shape Math would see.
        let reduced_axes: Vec<Term> = self
            .axes
            .iter()
            .filter(|axis| dst_axes.contains(axis))
            .cloned()
            .collect();
        let _ = reduce_broadcast(&reduced_axes, &dst_axes);
        Self::uninit_from_axes(dst_axes)
    }

    fn reshape<Mapping: M, Mapping2: M>(self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        if Mapping::to_value() == Mapping2::to_value() {
            self
        } else {
            Self::uninit_from_axes(gen_axes::<Mapping2>())
        }
    }

    fn write_transpose<Src: M, Dst: M>(
        &mut self,
        _src: &Self,
        _src_offset: &Index,
        _dst_offset: &Index,
        allow_broadcast: bool,
    ) {
        let _ = transpose_broadcast::<Src, Dst>(allow_broadcast);
    }

    fn zip_with<D2, D3, Other, Output, F>(&self, rhs: &Other, _f: F) -> Output
    where
        D2: Scalar,
        D3: Scalar,
        Other: RawTensor<D2>,
        Output: RawTensor<D3>,
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>,
    {
        assert_zip(self.axes(), rhs.axes());
        Output::uninit_from_axes(self.axes.clone())
    }

    fn write_scatter<Src, Key, Dst, Idx, IdxRaw>(&self, _dst: &mut Self, _index: &IdxRaw, _scaled: bool)
    where
        Src: M,
        Key: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>,
    {
        let src_fmapping = Src::to_value().factorize();
        let dst_fmapping = Dst::to_value().factorize();
        let key = Key::to_value().factorize();
        let _ = scatter_params(&src_fmapping, &dst_fmapping, &key);
    }

    fn write_gather<Src, Dst, Idx, IdxRaw>(&self, _dst: &mut Self, _index: &IdxRaw, _scaled: bool)
    where
        Src: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>,
    {
        let src_fmapping = Src::to_value().factorize();
        let dst_fmapping = Dst::to_value().factorize();
        let idx_fmapping = Idx::to_value().factorize();
        let _ = gather_params(&src_fmapping, &dst_fmapping, &idx_fmapping);
    }

    /// Phantom skips the per-position iteration entirely — Typecheck has no values to update,
    /// and the closure body's `read_index` would only ever return `Opt::Uninit`. Returns a
    /// fresh phantom tensor.
    fn apply_branch_operands<Mapping, Operand, TagRaw, F>(
        &self,
        _tag: &TagRaw,
        _operands: &[Operand],
        _update: F,
    ) -> Self
    where
        D: VeScalar,
        Mapping: M,
        TagRaw: RawTensor<u8>,
        Operand: OperandTag<D, Mapping>,
        F: FnMut(&Index, &Operand, &mut Self),
    {
        self.clone()
    }
}

impl<D: Scalar> RawTensorOpt<D> for PhantomRawTensor<D> {
    fn from_opt_buf<Mapping: M>(_data: impl IntoIterator<Item = Opt<D>>) -> Self {
        Self::uninit_from_axes(gen_axes::<Mapping>())
    }

    /// Like [`RawTensor::try_from_buf`], skip the length check: Typecheck callers may
    /// intentionally pass a buffer of any size to construct a phantom tensor.
    fn try_from_opt_buf<Mapping: M>(data: impl IntoIterator<Item = Opt<D>>) -> Result<Self, BufferConvertError> {
        Ok(Self::from_opt_buf::<Mapping>(data))
    }

    fn to_opt_buf<Mapping: M>(&self) -> Vec<Opt<D>> {
        Vec::new()
    }
}
