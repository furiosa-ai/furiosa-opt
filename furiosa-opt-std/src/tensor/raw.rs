//! `RawTensor` trait and the three storage implementations selected by `Backend::RawTensor<D>`.
//!
//! The three semantic kinds of raw tensor are:
//!
//! - [`MathRawTensor`] (Simulation): `ArrayD<Opt<D>>` host buffer with full ALU interpreter
//!   semantics.
//! - [`PhantomRawTensor`] (Typecheck): axes only, no buffer; reads return `Opt::Uninit`, writes
//!   are dropped.
//! - [`BufRawTensor`] (Emulation, Npu): native `Vec<D>` host staging buffer for DMA transfers.
//!   Carries no math; ALU semantics are not defined for this storage in the current crate, so
//!   the [`RawTensor`] methods that would touch values are `todo!()` until the Emulation backend
//!   lands real implementations.
//!
//! Each backend selects exactly one of these via `Backend::RawTensor<D>`; storage choice is what
//! determines whether a backend interprets tensor math on the host or routes it to a device.

use std::fmt::Debug;

use abi_stable::std_types::{RResult, RSlice};
use furiosa_mapping::{FMapping, FMappingExt, Index, IndexExt, M, Term};
use ndarray::IxDyn;

use crate::engine::vector::operand::OperandTag;
use crate::engine::vector::scalar::VeScalar;
use crate::scalar::{Opt, Scalar};
use crate::tensor::BufferConvertError;

mod buf;
mod math;
mod phantom;

pub use buf::BufRawTensor;
pub use math::MathRawTensor;
pub use phantom::PhantomRawTensor;

/// Common surface implemented by every backend storage type.
///
/// Almost every method is signature-only and each `RawTensor` impl states its own body
/// explicitly (mirroring the `Backend` trait convention). This is the only way to keep Typecheck
/// honest: under `PhantomRawTensor`, methods run only the `op_prep` helpers (so
/// mapping errors still surface) but skip per-element iteration entirely, instead of running
/// no-op loops. The single exception is `try_from_buf`, which is pure plumbing (length check
/// then `from_buf`) and has the same body across all storages — so it carries a default impl.
///
/// Per-impl summary:
/// - `MathRawTensor` (Simulation): real loops, `read_index` / `write_index` touch the
///   `ArrayD<Opt<D>>` host buffer.
/// - `PhantomRawTensor` (Typecheck): assertion-only — calls the relevant `op_prep`
///   helper for its panic side, then returns a phantom tensor without iterating. `read_index`
///   returns `Opt::Uninit`, `write_index` is a no-op (still required so user code that hits
///   them directly compiles).
/// - `BufRawTensor` (Emulation / Npu): `todo!()` placeholders until buffer-level semantics
///   land.
pub trait RawTensor<D: Scalar>: 'static + Clone + Debug {
    /// Returns the axes of this raw tensor.
    fn axes(&self) -> &[Term];

    /// Constructs an uninitialized raw tensor with the given axes.
    fn uninit_from_axes(axes: Vec<Term>) -> Self;

    /// Reads the value at `index`. Backends without host-side values may panic / `todo!()`.
    fn read_index(&self, index: Index) -> Opt<D>;

    /// Writes `value` at `index`. Backends without host-side values may panic / `todo!()`.
    fn write_index(&mut self, index: Index, value: Opt<D>);

    /// Constructs a raw tensor from a flat sequence of `D` values in physical-layout order.
    /// Axes are derived from `Mapping` internally. Each backend stores the buffer in its native
    /// shape; `MathRawTensor` lifts each value through `Opt::Init`, `BufRawTensor` collects into
    /// its native `Vec<D>`, `PhantomRawTensor` discards the input (axes only).
    fn from_buf<Mapping: M>(data: impl IntoIterator<Item = D>) -> Self;

    /// Validates the input length matches `Mapping::SIZE` then constructs via [`Self::from_buf`].
    /// Used at the `Tensor::try_from_buf` API boundary. Default impl is pure plumbing
    /// (collect → length check → `from_buf`); same across every storage, so no per-storage body
    /// needed.
    fn try_from_buf<Mapping: M>(data: impl IntoIterator<Item = D>) -> Result<Self, BufferConvertError>
    where
        Self: Sized,
    {
        let data: Vec<D> = data.into_iter().collect();
        if data.len() != Mapping::SIZE {
            return Err(BufferConvertError::length_mismatch(Mapping::SIZE, data.len()));
        }
        Ok(Self::from_buf::<Mapping>(data))
    }

    /// Serializes the raw tensor to a flat `Vec<D>` in `Mapping`-order (matching
    /// [`Self::from_buf`]'s input layout — same Mapping → roundtrips). `MathRawTensor` panics on
    /// `Opt::Uninit` slots; `BufRawTensor` clones its native `Vec<D>` (Mapping is unused — its
    /// storage is the wire-format buffer); `PhantomRawTensor` returns `Vec::new()`.
    fn to_buf<Mapping: M>(&self) -> Vec<D>;

    /// Constructs a raw tensor by applying `f` to each (axes, multi-dim coordinate) pair.
    ///
    /// The default body walks every generated index, materializes its multi-dim coordinate, and
    /// routes the producer's result through [`Self::write_index`]. `BufRawTensor` and
    /// `MathRawTensor` both inherit this body; `PhantomRawTensor` overrides it with a
    /// short-circuit because there is no buffer to fill.
    fn from_fn<F>(axes: Vec<Term>, mut f: F) -> Self
    where
        F: FnMut(&Vec<Term>, &IxDyn) -> Opt<D>,
    {
        let mut tensor = Self::uninit_from_axes(axes.clone());
        for index in Index::new().gen_indexes(FMapping::from_axes(RSlice::from(axes.as_slice()))) {
            let coords = finalize_coords(&axes, index.clone()).expect("generated index must be valid");
            tensor.write_index(index, f(&axes, &IxDyn(&coords)));
        }
        tensor
    }

    /// Element-wise map into another raw-tensor storage chosen by `Output`.
    ///
    /// `Tensor::map` relays here with `Output = B::RawTensor<D2>`, so backends stay in the same
    /// storage family while the logic itself remains purely storage-level.
    fn map<D2: Scalar, Output: RawTensor<D2>, F>(&self, f: F) -> Output
    where
        F: FnMut(&Opt<D>) -> Opt<D2>;

    /// Reduces the factors of `Src` that are absent in `Dst`. `Dst` must be an exact factor of
    /// `Src` (i.e., `Src::divide(Dst)` must `exact_check`); the missing factors form the
    /// reduction residue. Passing the source mapping explicitly preserves the per-factor
    /// distinction that `self.axes`'s consolidated form would otherwise lose (e.g., when a
    /// single symbol appears as multiple sub-factors and only some are reduced).
    fn reduce<Src: M, Dst: M, Reduce>(&self, reduce_fn: Reduce, identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>;

    /// Reduces, then broadcasts the result up to `Dst`'s full shape.
    fn reduce_then_broadcast<Src: M, Dst: M, Reduce>(&self, reduce_fn: Reduce, identity: Opt<D>) -> Self
    where
        Reduce: Fn(Opt<D>, Opt<D>) -> Opt<D>;

    /// Reshape from `Mapping`-shaped raw to `Mapping2`-shaped raw, reinterpreting the underlying
    /// physical buffer (C-major order over `Mapping` → C-major order over `Mapping2`).
    /// Identity (`Mapping == Mapping2`) is a fast-path returning `self` unchanged.
    fn reshape<Mapping: M, Mapping2: M>(self) -> Self;

    /// Writes a transposed/broadcast view of `src` into `self`. `src_offset` and `dst_offset`
    /// allow operating on partial views.
    fn write_transpose<Src: M, Dst: M>(
        &mut self,
        src: &Self,
        src_offset: &Index,
        dst_offset: &Index,
        allow_broadcast: bool,
    );

    /// Element-wise zip of `self` and `rhs` by `f`. Output storage is `Output: RawTensor<D3>`,
    /// chosen by the caller — `Tensor::zip_with` passes `B::RawTensor<D3>` so the result stays
    /// in the same backend storage family.
    fn zip_with<D2, D3, Other, Output, F>(&self, rhs: &Other, f: F) -> Output
    where
        D2: Scalar,
        D3: Scalar,
        Other: RawTensor<D2>,
        Output: RawTensor<D3>,
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>;

    /// Scatter values from `self` (with mapping `Src`) into `dst` (with mapping `Dst`) at
    /// positions read from `index` (with mapping `Idx`). `Key` is the scatter key axis; `scaled`
    /// selects between byte-stride (`true`) and element-stride (`false`) index decoding.
    fn write_scatter<Src, Key, Dst, Idx, IdxRaw>(&self, dst: &mut Self, index: &IdxRaw, scaled: bool)
    where
        Src: M,
        Key: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>;

    /// Gather values from `self` (table, with mapping `Src`) into `dst` (output, with mapping
    /// `Dst`) at positions read from `index` (with mapping `Idx`). Inverse of
    /// [`Self::write_scatter`].
    ///
    /// The gather-axis (the indexed axis in the table) is derived implicitly: the payload
    /// (= `dst ÷ idx`) determines which `dst` axes are non-indexed; the remaining single
    /// term in `src ÷ payload` is the indexed axis on the table side. `scaled` selects
    /// between byte-stride (`true`) and element-stride (`false`) index decoding (matching
    /// `write_scatter`'s semantics).
    fn write_gather<Src, Dst, Idx, IdxRaw>(&self, dst: &mut Self, index: &IdxRaw, scaled: bool)
    where
        Src: M,
        Dst: M,
        Idx: M,
        IdxRaw: RawTensor<i32>;

    /// VE branch-conditional update. Iterates every real position of `Mapping`, reads `eid`
    /// at each position, and for each operand whose `tag_filter()` matches the eid runs
    /// `update`. Returns a tensor with the per-position results applied to a clone of `self`.
    ///
    /// Lives on the trait specifically so `PhantomRawTensor` can short-circuit the iteration
    /// entirely (Typecheck has no values to update); pushing it out to a free fn would force
    /// Phantom through the full position loop unnecessarily.
    fn apply_branch_operands<Mapping, Operand, TagRaw, F>(&self, tag: &TagRaw, operands: &[Operand], update: F) -> Self
    where
        D: VeScalar,
        Mapping: M,
        TagRaw: RawTensor<u8>,
        Operand: OperandTag<D, Mapping>,
        F: FnMut(&Index, &Operand, &mut Self);
}

/// Storage-level extension for raw tensors that support logical [`Opt<D>`] values.
///
/// Implemented by `MathRawTensor` (real `Opt<D>` host buffer) and `PhantomRawTensor`
/// (no buffer; reads return `Opt::Uninit`). `BufRawTensor` deliberately does **not**
/// implement this trait — its `Vec<D>` staging buffer has no `Opt::Uninit` representation,
/// and the wider `Tensor` / `HostTensor` Opt-buffer surface should not be reachable on
/// Npu / Emulation.
///
/// Each backend whose `Backend::RawTensor<D>` implements `RawTensorOpt<D>` automatically
/// surfaces the `Tensor::{from_opt_buf, try_from_opt_buf, to_buf_opt}` and matching
/// `HostTensor` constructors via a generic `where`-bound on the storage type.
pub trait RawTensorOpt<D: Scalar>: RawTensor<D> {
    /// Construct from a logical `Opt<D>` buffer in physical layout. Padding positions in
    /// the input are dropped where the storage doesn't represent them.
    fn from_opt_buf<Mapping: M>(data: impl IntoIterator<Item = Opt<D>>) -> Self;

    /// Validates the input length matches `Mapping::SIZE` then constructs via
    /// [`Self::from_opt_buf`]. Default impl is pure plumbing (mirrors [`RawTensor::try_from_buf`]);
    /// `PhantomRawTensor` overrides to skip the length check (Typecheck doesn't represent buffer
    /// values, so any length is acceptable).
    fn try_from_opt_buf<Mapping: M>(data: impl IntoIterator<Item = Opt<D>>) -> Result<Self, BufferConvertError>
    where
        Self: Sized,
    {
        let data: Vec<Opt<D>> = data.into_iter().collect();
        if data.len() != Mapping::SIZE {
            return Err(BufferConvertError::length_mismatch(Mapping::SIZE, data.len()));
        }
        Ok(Self::from_opt_buf::<Mapping>(data))
    }

    /// Serialize to a logical `Opt<D>` buffer in physical layout. `MathRawTensor` reads
    /// each slot; `PhantomRawTensor` returns an empty `Vec` since there's no buffer to
    /// surface.
    fn to_opt_buf<Mapping: M>(&self) -> Vec<Opt<D>>;
}

/// Generates axes from a mapping.
pub(crate) fn gen_axes<Mapping: M>() -> Vec<Term> {
    let mut index = Index::new();
    index.add_mapping::<Mapping>(0);
    index
        .finalize()
        .expect("Invalid mapping")
        .into_iter()
        .map(|(term, _)| term)
        .collect()
}

/// Resolves an `Index` to a multi-dim coordinate vector against `axes`. Returns `None` when the
/// index lands in a padding slot.
pub(crate) fn finalize_coords(axes: &[Term], index: Index) -> Option<Vec<usize>> {
    let RResult::ROk(index) = index.finalize() else {
        return None;
    };
    assert!(
        axes.iter().zip(index.iter()).all(|(a, (b, _))| a == b),
        "Index terms ({:?}) do not match tensor axes ({:?}).",
        index,
        axes
    );
    Some(index.into_iter().map(|(_, value)| value).collect())
}

/// Returns the per-axis modulus list (i.e. the dense buffer shape implied by `axes`).
pub(crate) fn shape_from_axes(axes: &[Term]) -> Vec<usize> {
    axes.iter().map(|term| term.modulo).collect()
}
