use ndarray::{ArrayD, IxDyn};

use furiosa_mapping::{Index, IndexExt, M, Mapping, MappingExt, Term};

use crate::engine::vector::operand::OperandTag;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::op_prep::{assert_zip, broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::scalar::{Opt, Scalar};
use crate::tensor::raw::{RawTensor, RawTensorOpt, finalize_coords, gen_axes, shape_from_axes};

/// Math raw tensor (Simulation): `ArrayD<Opt<D>>` host buffer with full ALU semantics.
///
/// `data` only stores the math (non-padding) values: its shape is the product of the axes'
/// moduli. Buffers in physical layout order (length = `Mapping::SIZE`, includes padding) are
/// loaded position-by-position via [`Self::write_index`], which silently no-ops on padding
/// positions. Constructors that want bulk-load from such a buffer should iterate
/// `Index::new().gen_indexes(Mapping::to_value())` and call `write_index`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct MathRawTensor<D: Scalar> {
    axes: Vec<Term>,
    data: ArrayD<Opt<D>>,
}

impl<D: Scalar> RawTensor<D> for MathRawTensor<D> {
    fn axes(&self) -> &[Term] {
        &self.axes
    }

    fn uninit_from_axes(axes: Vec<Term>) -> Self {
        let shape = shape_from_axes(&axes);
        let data = ArrayD::from_elem(IxDyn(&shape), Opt::Uninit);
        Self { axes, data }
    }

    fn read_index(&self, index: Index) -> Opt<D> {
        let Some(coords) = finalize_coords(&self.axes, index) else {
            return Opt::Uninit;
        };
        *self.data.get(coords.as_slice()).expect("Index out of bounds.")
    }

    fn write_index(&mut self, index: Index, value: Opt<D>) {
        let Some(coords) = finalize_coords(&self.axes, index) else {
            return;
        };
        *self.data.get_mut(coords.as_slice()).expect("Index out of bounds.") = value;
    }

    fn from_buf<Mapping: M>(data: impl IntoIterator<Item = D>) -> Self {
        let mut tensor = Self::uninit_from_axes(gen_axes::<Mapping>());
        for (index, value) in Mapping::to_value().indexes().into_iter().zip(data) {
            tensor.write_index(index, Opt::Init(value));
        }
        tensor
    }

    fn to_buf<Mapping: M>(&self) -> Vec<D> {
        Mapping::to_value()
            .indexes()
            .into_iter()
            .map(|index| match self.read_index(index) {
                Opt::Init(value) => value,
                Opt::Uninit => panic!(
                    "MathRawTensor::to_buf called on a tensor containing Opt::Uninit slots; \
                     use the logical Opt-buffer view instead."
                ),
            })
            .collect()
    }

    fn to_buf_or_default<Mapping: M>(&self) -> Vec<D> {
        self.to_buf_or_default_opt::<Mapping>()
    }

    fn map<D2: Scalar, Output: RawTensor<D2>, F>(&self, mut f: F) -> Output
    where
        F: FnMut(&Opt<D>) -> Opt<D2>,
    {
        let axes = self.axes.to_vec();
        let mut output = Output::uninit_from_axes(axes.clone());
        for index in Index::new().gen_indexes(Mapping::from_terms(axes.iter().cloned())) {
            let value = self.read_index(index.clone());
            output.write_index(index, f(&value));
        }
        output
    }

    fn reduce<Src: M, Dst: M, Reduce>(&self, reduce_fn: Reduce, identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        // Carve Dst out of Src for the reduced axes, derived structurally so a partial-axis reduction
        // survives gen_axes consolidation.
        let reduce_residue = Src::to_value().carve(&Dst::to_value());
        let mut output = Self::uninit_from_axes(gen_axes::<Dst>());
        for dst_index in Index::new().gen_indexes(Dst::to_value()) {
            let mut acc = identity;
            for src_index in dst_index.clone().gen_indexes(reduce_residue.clone()) {
                acc = reduce_fn(acc, self.read_index(src_index));
            }
            output.write_index(dst_index, acc);
        }
        output
    }

    fn reduce_then_broadcast<Src: M, Dst: M, Reduce>(&self, reduce_fn: Reduce, identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        let src = Src::to_value();
        let dst = Dst::to_value();
        // Broadcast axes: the Dst axes built from symbols absent in Src (a symbol-level split).
        let broadcast = broadcast_axes(&src, &dst);
        // Kept axes: what's left of Dst after carving the broadcast out.
        let inter = dst.carve(&broadcast);
        // Reduced axes: what's left of Src after carving the kept axes out.
        let reduce_residue = src.carve(&inter);

        let mut output = Self::uninit_from_axes(gen_axes::<Dst>());
        for inter_index in Index::new().gen_indexes(inter) {
            let mut acc = identity;
            for src_index in inter_index.clone().gen_indexes(reduce_residue.clone()) {
                acc = reduce_fn(acc, self.read_index(src_index));
            }
            for dst_index in inter_index.gen_indexes(broadcast.clone()) {
                output.write_index(dst_index, acc);
            }
        }
        output
    }

    fn reshape<Mapping: M, Mapping2: M>(self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        Self::from_opt_buf::<Mapping2>(self.to_opt_buf::<Mapping>())
    }

    fn write_transpose<Src: M, Dst: M>(
        &mut self,
        src: &Self,
        src_offset: &Index,
        dst_offset: &Index,
        allow_broadcast: bool,
    ) {
        let broadcast = transpose_broadcast::<Src, Dst>(allow_broadcast);
        for index in Index::new().gen_indexes(Src::to_value()) {
            let mut src_index = index.clone();
            src_index.add(src_offset.clone());
            let value = src.read_index(src_index);

            let mut dst_index_base = index;
            dst_index_base.add(dst_offset.clone());
            for broadcast_index in dst_index_base.gen_indexes(broadcast.clone()) {
                self.write_index(broadcast_index, value);
            }
        }
    }

    fn zip_with<D2, D3, Other, Output, F>(&self, rhs: &Other, f: F) -> Output
    where
        D2: Scalar,
        D3: Scalar,
        Other: RawTensor<D2>,
        Output: RawTensor<D3>,
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>,
    {
        assert_zip(self.axes(), rhs.axes());
        let axes = self.axes().to_vec();
        let mut output = Output::uninit_from_axes(axes.clone());
        for index in Index::new().gen_indexes(Mapping::from_terms(axes.iter().cloned())) {
            let l = self.read_index(index.clone());
            let r = rhs.read_index(index.clone());
            output.write_index(index, f(l, r));
        }
        output
    }

    fn write_scatter<Src, Key, Dst, Idx, IdxRaw>(&self, dst: &mut Self, index: &IdxRaw, scaled: bool)
    where
        Src: M,
        Key: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>,
    {
        let key = Key::to_value();
        let (payload, dst_term) = scatter_params(&Src::to_value(), &Dst::to_value(), &key);

        let index_stride = if scaled {
            payload.clone().remove_padding().size() * std::mem::size_of::<D>()
        } else {
            1
        };

        let indices: Vec<usize> = (0..Idx::SIZE)
            .map(|i| {
                let mut idx = Index::new();
                idx.add_mapping::<Idx>(i);
                let opt = index.read_index(idx);
                let Opt::Init(v) = opt else {
                    panic!("Scatter index must be initialized")
                };
                usize::try_from(v).expect("Scatter index must be non-negative") / index_stride
            })
            .collect();

        for payload_index in Index::new().gen_indexes(payload) {
            for (key_pos, key_index) in Index::new().gen_indexes(key.clone()).into_iter().enumerate() {
                let mut src_index = payload_index.clone();
                src_index.add(key_index);
                let value = self.read_index(src_index);

                let mut dst_index = payload_index.clone();
                dst_index.add_term(dst_term.clone(), indices[key_pos]);
                dst.write_index(dst_index, value);
            }
        }
    }

    fn write_gather<Src, Dst, Idx, IdxRaw>(&self, dst: &mut Self, index: &IdxRaw, scaled: bool)
    where
        Src: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>,
    {
        let params = gather_params(&Src::to_value(), &Dst::to_value(), &Idx::to_value());

        let index_stride = if scaled {
            params.payload.clone().remove_padding().size() * std::mem::size_of::<D>()
        } else {
            1
        };

        let indices: Vec<usize> = (0..Idx::SIZE)
            .map(|i| {
                let mut idx = Index::new();
                idx.add_mapping::<Idx>(i);
                let opt = index.read_index(idx);
                let Opt::Init(v) = opt else {
                    panic!("Gather index must be initialized")
                };
                usize::try_from(v).expect("Gather index must be non-negative") / index_stride
            })
            .collect();

        for payload_index in Index::new().gen_indexes(params.payload) {
            for (idx_pos, dst_iter_index) in Index::new()
                .gen_indexes(params.idx_residue.clone())
                .into_iter()
                .enumerate()
            {
                let mut src_index = payload_index.clone();
                src_index.add_term(params.src_term.clone(), indices[idx_pos]);
                let value = self.read_index(src_index);

                let mut dst_index = payload_index.clone();
                dst_index.add(dst_iter_index);
                dst.write_index(dst_index, value);
            }
        }
    }

    fn apply_branch_operands<Mapping, Operand, TagRaw, F>(
        &self,
        tag: &TagRaw,
        operands: &[Operand],
        mut update: F,
    ) -> Self
    where
        D: VeScalar,
        Mapping: M,
        TagRaw: RawTensor<u8>,
        Operand: OperandTag<D, Mapping>,
        F: FnMut(&Index, &Operand, &mut Self),
    {
        let mut output = self.clone();
        for index in Index::new().gen_indexes(Mapping::to_value()) {
            let eid = tag.read_index(index.clone());
            let Opt::Init(_) = eid else {
                continue;
            };
            for operand in operands {
                if !operand.tag_filter().matches(eid) {
                    continue;
                }
                update(&index, operand, &mut output);
            }
        }
        output
    }
}

impl<D: Scalar> RawTensorOpt<D> for MathRawTensor<D> {
    fn from_opt_buf<Mapping: M>(data: impl IntoIterator<Item = Opt<D>>) -> Self {
        // Physical-layout iteration: a padding position's finalized index is `RErr`, so
        // `write_index` silently no-ops there. The corresponding `data` value (typically
        // `Opt::Uninit` for padding slots) is dropped, matching `MathRawTensor`'s
        // padding-stripped `ArrayD` representation.
        let mut tensor = Self::uninit_from_axes(gen_axes::<Mapping>());
        for (index, value) in Mapping::to_value().indexes().into_iter().zip(data) {
            tensor.write_index(index, value);
        }
        tensor
    }

    fn to_opt_buf<Mapping: M>(&self) -> Vec<Opt<D>> {
        Mapping::to_value()
            .indexes()
            .into_iter()
            .map(|index| self.read_index(index))
            .collect()
    }
}
