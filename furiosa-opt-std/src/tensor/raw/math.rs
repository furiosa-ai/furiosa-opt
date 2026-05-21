use abi_stable::std_types::RSlice;
use ndarray::{ArrayD, IxDyn};

use furiosa_mapping::{DivisionExt, FMapping, FMappingExt, Index, IndexExt, M, MappingExt, Term};

use crate::engine::vector::operand::OperandTag;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::op_prep::{assert_zip, gather_params, scatter_params, transpose_broadcast};
use crate::scalar::{Opt, Scalar};
use crate::tensor::raw::{RawTensor, RawTensorOpt, finalize_coords, gen_axes, shape_from_axes};

/// Math raw tensor (Simulation): `ArrayD<Opt<D>>` host buffer with full ALU semantics.
///
/// `data` only stores the math (non-padding) values: its shape is the product of the axes'
/// moduli. Buffers in physical layout order (length = `Mapping::SIZE`, includes padding) are
/// loaded position-by-position via [`Self::write_index`], which silently no-ops on padding
/// positions. Constructors that want bulk-load from such a buffer should iterate
/// `Index::new().gen_indexes(Mapping::to_value().factorize())` and call `write_index`.
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
        let fmapping = Mapping::to_value().factorize();
        let mut tensor = Self::uninit_from_axes(gen_axes::<Mapping>());
        for (i, value) in data.into_iter().enumerate() {
            tensor.write_index(fmapping.eval(i), Opt::Init(value));
        }
        tensor
    }

    fn to_buf<Mapping: M>(&self) -> Vec<D> {
        let fmapping = Mapping::to_value().factorize();
        (0..Mapping::SIZE)
            .map(|i| match self.read_index(fmapping.eval(i)) {
                Opt::Init(value) => value,
                Opt::Uninit => panic!(
                    "MathRawTensor::to_buf called on a tensor containing Opt::Uninit slots; \
                     use the logical Opt-buffer view instead."
                ),
            })
            .collect()
    }

    fn map<D2: Scalar, Output: RawTensor<D2>, F>(&self, mut f: F) -> Output
    where
        F: FnMut(&Opt<D>) -> Opt<D2>,
    {
        let axes = self.axes.to_vec();
        let mut output = Output::uninit_from_axes(axes.clone());
        for index in Index::new().gen_indexes(FMapping::from_axes(RSlice::from(axes.as_slice()))) {
            let value = self.read_index(index.clone());
            output.write_index(index, f(&value));
        }
        output
    }

    fn reduce<Src: M, Dst: M, Reduce>(&self, reduce_fn: Reduce, identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>,
    {
        // Reduce residue = Src - Dst (per-factor algebra), derived structurally from the source
        // mapping so partial-axis reductions survive `gen_axes` consolidation.
        let reduce_residue = Src::to_value()
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
            })
            .relaxed_residues()
            .dividend_residue;
        let mut output = Self::uninit_from_axes(gen_axes::<Dst>());
        for dst_index in Index::new().gen_indexes(Dst::to_value().factorize()) {
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
        // `Src.divide(Dst)` decomposes the algebra; everything we need falls out as `remainder`s:
        //   * remainder(Dividend) → factors to reduce  (Src − matched),
        //   * remainder(Divisor)  → factors to broadcast (Dst − matched),
        //   * Inter = Src.factorize() − reduce_residue, again via `remainder`.
        // Going through `divide` rather than a `Term`-equality filter is what lets a single
        // symbol's sub-factors line up correctly (e.g., `Src=K(1,128)` and `Dst=K(16,8)` where
        // the K(1,16) sub-factor must be reduced and the K(16,8) sub-factor kept).
        //
        // No intermediate tensor: for each Inter position we accumulate from `self` over the
        // reduce residue, then write the accumulator to every Dst position that shares this
        // Inter (= broadcast residue × Inter).
        let src_mapping = Src::to_value();
        let division = src_mapping.divide(&Dst::to_value());
        let reduce_residue = division
            .remainder(furiosa_mapping::DivisionSide::Dividend)
            .expect("[reduce_then_broadcast] reduce residue must be well-formed");
        let broadcast_residue = division
            .remainder(furiosa_mapping::DivisionSide::Divisor)
            .expect("[reduce_then_broadcast] broadcast residue must be well-formed");
        let inter_fmapping = src_mapping
            .factorize()
            .divide(reduce_residue.clone())
            .remainder(furiosa_mapping::DivisionSide::Dividend)
            .expect("[reduce_then_broadcast] inter = Src − reduce_residue must be well-formed");

        let mut dst = Self::uninit_from_axes(gen_axes::<Dst>());
        for inter_index in Index::new().gen_indexes(inter_fmapping) {
            let mut acc = identity;
            for src_index in inter_index.clone().gen_indexes(reduce_residue.clone()) {
                acc = reduce_fn(acc, self.read_index(src_index));
            }
            for dst_index in inter_index.gen_indexes(broadcast_residue.clone()) {
                dst.write_index(dst_index, acc);
            }
        }
        dst
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
        for index in Index::new().gen_indexes(Src::to_value().factorize()) {
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
        for index in Index::new().gen_indexes(FMapping::from_axes(RSlice::from(axes.as_slice()))) {
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
        let src_fmapping = Src::to_value().factorize();
        let dst_fmapping = Dst::to_value().factorize();
        let key = Key::to_value().factorize();

        let index_stride = if scaled {
            let payload = src_fmapping
                .clone()
                .divide(key.clone())
                .exact_checked()
                .expect("Src must contain scatter key")
                .relaxed_residues()
                .dividend_residue;
            payload.remove_padding().size() * std::mem::size_of::<D>()
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

        let (payload, dst_term) = scatter_params(&src_fmapping, &dst_fmapping, &key);
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
        let src_fmapping = Src::to_value().factorize();
        let dst_fmapping = Dst::to_value().factorize();
        let idx_fmapping = Idx::to_value().factorize();

        let params = gather_params(&src_fmapping, &dst_fmapping, &idx_fmapping);

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
        for index in Index::new().gen_indexes(Mapping::to_value().factorize()) {
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
        // Physical-layout iteration: padding positions land at an index whose `finalize()`
        // returns `RErr`, so `write_index` silently no-ops there. The corresponding `data[i]`
        // value (typically `Opt::Uninit` for padding slots) is dropped, matching
        // `MathRawTensor`'s padding-stripped `ArrayD` representation.
        let fmapping = Mapping::to_value().factorize();
        let mut tensor = Self::uninit_from_axes(gen_axes::<Mapping>());
        for (i, value) in data.into_iter().enumerate() {
            tensor.write_index(fmapping.eval(i), value);
        }
        tensor
    }

    fn to_opt_buf<Mapping: M>(&self) -> Vec<Opt<D>> {
        let fmapping = Mapping::to_value().factorize();
        (0..Mapping::SIZE).map(|i| self.read_index(fmapping.eval(i))).collect()
    }
}
