use furiosa_mapping::{Index, M, Term};

use crate::engine::vector::operand::OperandTag;
use crate::engine::vector::scalar::VeScalar;
use crate::scalar::{Opt, Scalar};
use crate::tensor::raw::{RawTensor, finalize_coords, gen_axes, shape_from_axes};

/// Buf raw tensor: native `Vec<D>` host staging buffer, shared by `Npu` and `Emulation`.
///
/// Holds a flat `Vec<D>` that serves as the host staging area for DMA to/from device memory.
/// Under Npu the device kernel produces the values, so `BufRawTensor`'s value-touching
/// [`RawTensor`] methods are not reached at runtime; their bodies are `todo!()` placeholders
/// labelled "buffer semantics not implemented yet" — the Emulation backend (host-side
/// buffer interpreter) will eventually supply real implementations.
///
/// TODO(buffer-axes-consistency): `axes` here is derived via `gen_axes::<Mapping>()` which
/// returns canonical-order axes regardless of `Mapping`'s syntactic order. `data` is stored in
/// caller-supplied buffer order (= `Mapping`-order). The two fields therefore do not describe
/// the same layout when `Mapping`'s order differs from canonical. The inconsistency is dormant
/// today (`read_index` / `write_index` are `todo!()`, so no axis-vs-data lookup actually
/// happens), but a real Emulation interpreter will need this resolved — likely by replacing
/// `axes` with `mapping: furiosa_mapping::Mapping` so the storage carries its own buffer
/// shape. `MathRawTensor` is unaffected because its storage is in canonical math layout
/// regardless of the boundary `Mapping`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct BufRawTensor<D: Scalar> {
    pub(crate) axes: Vec<Term>,
    pub(crate) data: Vec<D>,
}

impl<D: Scalar> RawTensor<D> for BufRawTensor<D> {
    fn axes(&self) -> &[Term] {
        &self.axes
    }

    fn uninit_from_axes(axes: Vec<Term>) -> Self {
        let len = shape_from_axes(&axes).iter().product::<usize>();
        // SAFETY: zero-init may be incorrect for some `D`, but every concrete scalar in this crate
        // is plain-old-data; the buffer is only ever read after a DMA write fills it, never while
        // it still contains the placeholder bytes.
        let mut data: Vec<D> = Vec::with_capacity(len);
        // Use `resize_with` so we don't require `D: Default`.
        data.resize_with(len, || unsafe { std::mem::zeroed() });
        Self { axes, data }
    }

    fn read_index(&self, index: Index) -> Opt<D> {
        match self.linear_index(index) {
            Some(linear) => Opt::Init(self.data[linear]),
            None => Opt::Uninit,
        }
    }

    fn write_index(&mut self, index: Index, value: Opt<D>) {
        let Opt::Init(v) = value else { return };
        let Some(linear) = self.linear_index(index) else {
            return;
        };
        self.data[linear] = v;
    }

    fn from_buf<Mapping: M>(data: impl IntoIterator<Item = D>) -> Self {
        let axes = gen_axes::<Mapping>();
        let data: Vec<D> = data.into_iter().collect();
        let expected = shape_from_axes(&axes).iter().product::<usize>();
        assert_eq!(expected, data.len(), "shape mismatch");
        Self { axes, data }
    }

    fn to_buf<Mapping: M>(&self) -> Vec<D> {
        self.data.clone()
    }

    fn map<D2: Scalar, Output: RawTensor<D2>, F>(&self, _f: F) -> Output
    where
        F: FnMut(&Opt<D>) -> Opt<D2>,
    {
        todo!("BufRawTensor::map: buffer semantics not implemented yet")
    }

    fn reduce<Src: M, Dst: M, Reduce>(&self, _reduce_fn: Reduce, _identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        todo!("BufRawTensor::reduce: buffer semantics not implemented yet")
    }

    fn reduce_then_broadcast<Src: M, Dst: M, Reduce>(&self, _reduce_fn: Reduce, _identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        todo!("BufRawTensor::reduce_then_broadcast: buffer semantics not implemented yet")
    }

    fn reshape<Mapping: M, Mapping2: M>(self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        Self {
            axes: gen_axes::<Mapping2>(),
            data: self.data,
        }
    }

    fn write_transpose<Src: M, Dst: M>(
        &mut self,
        _src: &Self,
        _src_offset: &Index,
        _dst_offset: &Index,
        _allow_broadcast: bool,
    ) {
        todo!("BufRawTensor::write_transpose: buffer semantics not implemented yet")
    }

    fn zip_with<D2, D3, Other, Output, F>(&self, _rhs: &Other, _f: F) -> Output
    where
        D2: Scalar,
        D3: Scalar,
        Other: RawTensor<D2>,
        Output: RawTensor<D3>,
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>,
    {
        todo!("BufRawTensor::zip_with: buffer semantics not implemented yet")
    }

    fn write_scatter<Src, Key, Dst, Idx, IdxRaw>(&self, _dst: &mut Self, _index: &IdxRaw, _scaled: bool)
    where
        Src: M,
        Key: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>,
    {
        todo!("BufRawTensor::write_scatter: buffer semantics not implemented yet")
    }

    fn write_gather<Src, Dst, Idx, IdxRaw>(&self, _dst: &mut Self, _index: &IdxRaw, _scaled: bool)
    where
        Src: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>,
    {
        todo!("BufRawTensor::write_gather: buffer semantics not implemented yet")
    }

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
        todo!("BufRawTensor::apply_branch_operands: buffer semantics not implemented yet")
    }
}

impl<D: Scalar> BufRawTensor<D> {
    /// Resolve `index` to a row-major offset into `self.data`, or `None` if the
    /// index falls outside the tensor's axes.
    fn linear_index(&self, index: Index) -> Option<usize> {
        let coords = finalize_coords(&self.axes, index)?;
        let shape = shape_from_axes(&self.axes);
        Some(coords.iter().zip(shape.iter()).fold(0usize, |acc, (c, &s)| acc * s + c))
    }
}
