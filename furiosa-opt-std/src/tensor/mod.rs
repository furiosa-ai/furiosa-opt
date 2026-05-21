use std::fmt;
use std::marker::PhantomData;

use furiosa_mapping::*;
use ndarray::IxDyn;
use num_traits::Zero;
use rand::Rng;
use rand::distr::StandardUniform;

use self::view::*;
use crate::scalar::*;
use crate::tensor::raw::gen_axes;

use crate::engine::vector::operand::OperandTag;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::{Backend, CurrentBackend};

pub(crate) mod memory;
pub mod pseudo;
pub(crate) mod raw;
pub(crate) mod tu;
pub(crate) mod view;

pub use raw::{BufRawTensor, MathRawTensor, PhantomRawTensor, RawTensor, RawTensorOpt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferConvertErrorKind {
    LengthMismatch { expected: usize, actual: usize },
}

/// Error raised when a logical buffer cannot be lowered into the backend's storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferConvertError {
    kind: BufferConvertErrorKind,
}

impl BufferConvertError {
    pub(crate) fn length_mismatch(expected: usize, actual: usize) -> Self {
        Self {
            kind: BufferConvertErrorKind::LengthMismatch { expected, actual },
        }
    }
}

impl fmt::Display for BufferConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            BufferConvertErrorKind::LengthMismatch { expected, actual } => {
                write!(f, "buffer length mismatch: expected {expected} elements, got {actual}")
            }
        }
    }
}

impl std::error::Error for BufferConvertError {}

/// Tensor with scalar type `D`, axes determined by `Mapping`, and backend determined by `B`.
///
/// `B` defaults to [`CurrentBackend`], a cfg-aliased type that picks Simulation / Emulation / Npu
/// / Typecheck. User code writes `Tensor<D, M>` and gets the backend-appropriate storage
/// automatically. Explicit `Tensor<D, M, SomeBackend>` overrides for testing / cross-backend code.
pub struct Tensor<D: Scalar, Mapping: M, B: Backend = CurrentBackend> {
    inner: B::RawTensor<D>,
    _marker: PhantomData<(Mapping, B)>,
}

impl<D: Scalar, Mapping: M, B: Backend> std::fmt::Debug for Tensor<D, Mapping, B>
where
    B::RawTensor<D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor").field("inner", &self.inner).finish()
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Clone for Tensor<D, Mapping, B>
where
    B::RawTensor<D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Wraps a backend-native raw tensor in `Tensor`.
    pub(crate) fn from_inner(inner: B::RawTensor<D>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Consumes the tensor and returns its underlying backend-native raw tensor. Pair this with
    /// [`RawTensor::from_buf`] to compare against an expected raw tensor without a `Vec<D>`
    /// round-trip.
    pub fn into_raw(self) -> B::RawTensor<D> {
        self.inner
    }

    pub(crate) fn read_index(&self, index: Index) -> Opt<D> {
        self.inner.read_index(index)
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Creates a new tensor from an initialized buffer.
    pub fn from_buf(data: impl IntoIterator<Item = D>) -> Self {
        Self::try_from_buf(data).unwrap_or_else(|err| panic!("failed to convert buffer for backend storage: {err}"))
    }

    /// Creates a new tensor from an initialized buffer, returning an error if the input cannot be
    /// lowered.
    pub fn try_from_buf(data: impl IntoIterator<Item = D>) -> Result<Self, BufferConvertError> {
        B::RawTensor::try_from_buf::<Mapping>(data).map(Self::from_inner)
    }

    /// Returns the tensor data as a flat `Vec<D>`. On Simulation, panics if any slot is
    /// `Opt::Uninit`. Use `to_buf_opt` (Simulation/Typecheck only) to inspect logical Opt-bearing
    /// values.
    pub fn to_buf(&self) -> Vec<D> {
        self.inner.to_buf::<Mapping>()
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Creates a new tensor from a function. Delegates to [`RawTensor::from_fn`]; storage choice
    /// (`MathRawTensor` / `BufRawTensor` / `PhantomRawTensor`) determines whether iteration writes
    /// real values, propagates `todo!()`, or silently drops them.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(&Vec<Term>, &IxDyn) -> Opt<D>,
    {
        Self::from_inner(B::RawTensor::from_fn(gen_axes::<Mapping>(), f))
    }

    /// Maps the tensor with a function. Delegates to [`RawTensor::map`].
    pub fn map<D2: Scalar, F>(&self, f: F) -> Tensor<D2, Mapping, B>
    where
        F: FnMut(&Opt<D>) -> Opt<D2>,
    {
        Tensor::from_inner(self.inner.map::<D2, _, F>(f))
    }

    /// Zips two tensors with a function. Delegates to [`RawTensor::zip_with`].
    pub fn zip_with<D2: Scalar, D3: Scalar, F>(&self, other: &Tensor<D2, Mapping, B>, f: F) -> Tensor<D3, Mapping, B>
    where
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>,
    {
        Tensor::from_inner(self.inner.zip_with::<D2, D3, _, _, F>(&other.inner, f))
    }

    /// Performs reduction (sum) over axes not present in `Dst`.
    pub fn reduce_add<Dst: M>(&self) -> Tensor<D, Dst, B> {
        self.reduce::<Dst>(|a, b| a + b, Opt::zero())
    }

    /// Reduces the factors of `self`'s mapping that are absent in `Dst`. `Dst` must be an
    /// exact factor of the source mapping; the source mapping is passed through so that
    /// per-factor identity is preserved (e.g., a symbol appearing as multiple sub-factors,
    /// only some of which are reduced).
    pub fn reduce<Dst: M>(&self, reduce_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>, identity: Opt<D>) -> Tensor<D, Dst, B> {
        Tensor::from_inner(self.inner.reduce::<Mapping, Dst, _>(reduce_fn, identity))
    }

    /// Creates a zero tensor.
    pub fn zero() -> Self
    where
        D: Zero,
    {
        Self::from_fn(|_, _| Opt::Init(D::zero()))
    }

    /// Creates a random tensor.
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Self::from_fn(|_, _| Opt::Init(rng.random::<D>()))
    }

    /// Performs reduction followed by broadcasting to match destination axes.
    pub fn reduce_then_broadcast<Dst: M>(&self) -> Tensor<D, Dst, B> {
        self.reduce_then_broadcast_with::<Dst>(|a, b| a + b, Opt::zero())
    }

    /// Performs reduction with a custom function, followed by broadcasting to match destination axes.
    pub fn reduce_then_broadcast_with<Dst: M>(
        &self,
        reduce_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
        identity: Opt<D>,
    ) -> Tensor<D, Dst, B> {
        Tensor::from_inner(self.inner.reduce_then_broadcast::<Mapping, Dst, _>(reduce_fn, identity))
    }

    /// Performs contraction between two tensors.
    pub fn contraction<Union: M, Lhs: M, Rhs: M>(lhs: &Tensor<D, Lhs, B>, rhs: &Tensor<D, Rhs, B>) -> Self {
        lhs.transpose::<Union>(true)
            .zip_with(&rhs.transpose(true), |a, b| a * b)
            .reduce_add()
    }
}

/// Host-side tensor methods. NPU has no host-side value semantics; Npu's `Backend` impl supplies
/// `todo!()` stubs for the underlying iteration primitives so `B = Npu` resolves here without a
/// separate parallel inherent block.
impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Creates an uninitialized tensor.
    pub fn uninit() -> Self {
        Self::from_inner(B::RawTensor::uninit_from_axes(gen_axes::<Mapping>()))
    }

    /// Creates a mutable view of the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Mapping, B> {
        TensorViewMut::new(&mut self.inner)
    }

    /// Creates an immutable view of the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Mapping, B> {
        TensorView::new(&self.inner)
    }

    /// Transmutes the tensor to a different mapping.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying data layout is compatible with the new mapping.
    pub unsafe fn transmute<Mapping2: M>(self) -> Tensor<D, Mapping2, B> {
        Tensor {
            inner: self.inner,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B> {
    /// Reshapes the tensor to a different mapping. Delegates to [`RawTensor::reshape`], which
    /// reinterprets the underlying physical buffer (C-major over `Mapping` → C-major over
    /// `Mapping2`); each backend supplies its own implementation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the tensor mapping sizes are equal,
    /// i.e. `Mapping::SIZE == Mapping2::SIZE`.
    pub unsafe fn reshape<Mapping2: M>(self) -> Tensor<D, Mapping2, B> {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        Tensor::from_inner(self.inner.reshape::<Mapping, Mapping2>())
    }

    /// Performs transpose. The mapping division is validated inside
    /// [`RawTensor::write_transpose`] (via `transpose_broadcast`), so this generic body works
    /// for every backend: Simulation writes values, Typecheck only runs the assertion, Npu /
    /// Emulation hit `todo!()`.
    pub fn transpose<Dst: M>(&self, allow_broadcast: bool) -> Tensor<D, Dst, B> {
        let mut dst = Tensor::uninit();
        dst.view_mut().write_transpose(self.view(), allow_broadcast);
        dst
    }

    /// Scatter elements from self into dst at positions given by index tensor. Delegates to
    /// [`RawTensor::write_scatter`].
    pub fn write_scatter<Key: M, Dst: M, Idx: M>(
        &self,
        dst: &mut Tensor<D, Dst, B>,
        index: &Tensor<i32, Idx, B>,
        scaled: bool,
    ) {
        self.inner
            .write_scatter::<Mapping, Key, Dst, Idx, _>(&mut dst.inner, &index.inner, scaled);
    }

    /// Gather elements from self (table) into dst at positions given by index tensor. Delegates
    /// to [`RawTensor::write_gather`].
    pub fn write_gather<Dst: M, Idx: M>(&self, dst: &mut Tensor<D, Dst, B>, index: &Tensor<i32, Idx, B>, scaled: bool) {
        self.inner
            .write_gather::<Mapping, Dst, Idx, _>(&mut dst.inner, &index.inner, scaled);
    }

    /// VE branch-conditional update. Delegates to [`RawTensor::apply_branch_operands`].
    pub(crate) fn apply_branch_operands<Operand, F>(
        &self,
        tag: &Tensor<u8, Mapping, B>,
        operands: &[Operand],
        update: F,
    ) -> Self
    where
        D: VeScalar,
        Operand: OperandTag<D, Mapping>,
        F: FnMut(&Index, &Operand, &mut B::RawTensor<D>),
    {
        Self::from_inner(
            self.inner
                .apply_branch_operands::<Mapping, Operand, _, F>(&tag.inner, operands, update),
        )
    }
}

/// `Opt`-buffer constructors and serializers — gated on `B::RawTensor<D>: RawTensorOpt<D>` so
/// only backends whose storage represents `Opt<D>` (Simulation: `MathRawTensor`, Typecheck:
/// `PhantomRawTensor`) surface them. Npu / Emulation's `BufRawTensor` deliberately does not
/// implement `RawTensorOpt`, so this surface is unreachable for those backends.
impl<D: Scalar, Mapping: M, B: Backend> Tensor<D, Mapping, B>
where
    B::RawTensor<D>: RawTensorOpt<D>,
{
    /// Creates a new tensor from a logical `Opt<D>` buffer.
    pub fn from_opt_buf(data: impl IntoIterator<Item = Opt<D>>) -> Self {
        Self::try_from_opt_buf(data)
            .unwrap_or_else(|err| panic!("failed to convert logical buffer for backend storage: {err}"))
    }

    /// Creates a new tensor from a logical `Opt<D>` buffer, returning an error if the input
    /// cannot be lowered. Length-check policy is delegated to the storage's
    /// [`RawTensorOpt::try_from_opt_buf`] impl.
    pub fn try_from_opt_buf(data: impl IntoIterator<Item = Opt<D>>) -> Result<Self, BufferConvertError> {
        B::RawTensor::try_from_opt_buf::<Mapping>(data).map(Self::from_inner)
    }

    /// Returns the tensor data as a flat logical `Opt<D>` buffer in physical layout order.
    pub fn to_buf_opt(&self) -> Vec<Opt<D>> {
        self.inner.to_opt_buf::<Mapping>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Npu, Simulation, Typecheck};

    /// Regression for `MathRawTensor::reduce` where the source and destination mappings share a
    /// symbol whose factors are unevenly split: `Src = m![K, M]` consolidates K into a single
    /// `K(stride 1, mod 8)` term in `self.axes`, while `Dst = m![K / 2, M]` carries only K's
    /// `(stride 2, mod 4)` sub-factor — the `(stride 1, mod 2)` piece must be reduced. The
    /// old filter-by-`Term`-equality logic dropped K entirely (since the two K Terms differ).
    /// The strict per-factor reduction must split K into "keep" (stride 2, mod 4) and
    /// "reduce" (stride 1, mod 2) and sum exactly the in-between K positions.
    #[test]
    fn simulation_reduce_keeps_partial_axis_when_only_a_sub_factor_remains() {
        axes![K = 8, M = 4];

        // m![K, M] layout (K outer, M inner): position (k, m) ↦ value `k + 100 * m`.
        let source_buf: Vec<i32> = (0..8).flat_map(|k| (0..4).map(move |m| k + 100 * m)).collect();
        let source = Tensor::<i32, m![K, M], Simulation>::from_buf(source_buf);

        // m![K / 2, M] keeps K positions {0, 2, 4, 6} (stride 2, mod 4), reducing the
        // (stride 1, mod 2) sub-factor: dst[j, m] = source[2j, m] + source[2j+1, m].
        let reduced: Tensor<i32, m![K / 2, M], Simulation> = source.reduce_add();

        // dst[j, m] = (2j + 100m) + (2j+1 + 100m) = 4j + 1 + 200m.
        // K/2 outer (stride M=4), M inner: iteration is (j=0,m=0..4), (j=1,m=0..4), ...
        let expected: Vec<i32> = (0..4).flat_map(|j| (0..4).map(move |m| 4 * j + 1 + 200 * m)).collect();
        assert_eq!(reduced.to_buf(), expected);
    }

    /// Regression for `MathRawTensor::reduce_then_broadcast` covering the same partial-sub-factor
    /// hazard as the `reduce` test above, *plus* a real broadcast axis that exists only in `Dst`.
    /// `Src = m![A]` consolidates A as `A(stride 1, mod 4)`; `Dst = m![A / 2, B]` keeps A's
    /// `(stride 2, mod 2)` sub-factor (reducing `A % 2`) and broadcasts over a fresh `B` axis.
    /// The divide-based intersection must split A correctly so each `B` column receives the
    /// pair-sum (not the full sum across all of A).
    #[test]
    fn simulation_reduce_then_broadcast_keeps_partial_axis_and_broadcasts_extra_dst_axis() {
        axes![A = 4, B = 2];

        // m![A]: source = [0, 1, 2, 3].
        let source = Tensor::<i32, m![A], Simulation>::from_buf(vec![0, 1, 2, 3]);

        // Reduce A%2 (sum pairs), broadcast over B:
        //   dst[j, b] = source[2j] + source[2j+1] for every b.
        let result: Tensor<i32, m![A / 2, B], Simulation> = source.reduce_then_broadcast();

        // m![A/2, B] iteration: (j=0,b=0), (j=0,b=1), (j=1,b=0), (j=1,b=1) → [1, 1, 5, 5].
        assert_eq!(result.to_buf(), vec![1, 1, 5, 5]);
    }

    #[test]
    fn simulation_from_buf_round_trips_through_opt_storage() {
        axes![A = 2];

        let tensor = Tensor::<i32, m![A], Simulation>::from_buf(vec![1, 2]);

        assert_eq!(tensor.to_buf(), vec![1, 2]);
        assert_eq!(tensor.to_buf_opt(), vec![Opt::Init(1), Opt::Init(2)]);
    }

    #[test]
    fn typecheck_try_from_buf_ignores_length_mismatch() {
        axes![A = 2];

        let tensor = Tensor::<i32, m![A], Typecheck>::try_from_buf(vec![1]).unwrap();

        assert!(tensor.to_buf().is_empty());
        assert!(tensor.to_buf_opt().is_empty());
    }

    #[test]
    fn typecheck_to_buf_is_empty() {
        axes![A = 2];

        let tensor = Tensor::<i32, m![A], Typecheck>::empty();

        assert!(tensor.to_buf().is_empty());
        assert!(tensor.to_buf_opt().is_empty());
    }

    /// Math-backend round-trip: gather a small table by a list of indices and confirm
    /// we get table rows back in indexed order. Mirrors the inverse-of-scatter contract
    /// documented on `HbmTensor::dma_gather`.
    ///
    /// `scaled=true` matches the existing `write_scatter` semantics: indices are in
    /// byte-offset units (e.g., row 1 of `[W, V=2]` of i32 = byte offset `1*2*4 = 8`).
    /// The math impl divides by `payload_size * sizeof(D) = 8` to get the position index.
    /// `scaled=false` treats indices as raw position indices.
    #[test]
    fn simulation_write_gather_roundtrip_scaled() {
        // table: [W=3, V=2] = [[10,11], [20,21], [30,31]]. gather-key = W.
        // indices select rows 0, 2, 1, 0 → byte-offsets 0, 16, 8, 0 (V*sizeof(i32)=8 per row).
        // output: [K=4, V=2].
        axes![W = 3, V = 2, K = 4];

        let table = Tensor::<i32, m![W, V], Simulation>::from_buf(vec![10, 11, 20, 21, 30, 31]);
        let index = Tensor::<i32, m![K], Simulation>::from_buf(vec![0, 16, 8, 0]);
        let mut output = Tensor::<i32, m![K, V], Simulation>::uninit();
        table.write_gather::<_, _>(&mut output, &index, true);

        assert_eq!(output.to_buf(), vec![10, 11, 30, 31, 20, 21, 10, 11]);
    }

    /// `scaled=false`: indices are raw row-position values.
    #[test]
    fn simulation_write_gather_roundtrip_unscaled() {
        axes![W = 3, V = 2, K = 4];

        let table = Tensor::<i32, m![W, V], Simulation>::from_buf(vec![10, 11, 20, 21, 30, 31]);
        let index = Tensor::<i32, m![K], Simulation>::from_buf(vec![0, 2, 1, 0]);
        let mut output = Tensor::<i32, m![K, V], Simulation>::uninit();
        table.write_gather::<_, _>(&mut output, &index, false);

        assert_eq!(output.to_buf(), vec![10, 11, 30, 31, 20, 21, 10, 11]);
    }

    /// Typecheck-backend smoke: gather should propagate mapping checks (via
    /// `gather_params`) without iterating any buffer. The output tensor under Typecheck
    /// has no values; we only verify that the call doesn't panic for a well-formed shape.
    #[test]
    fn typecheck_write_gather_runs_assertion_only() {
        axes![W = 3, V = 2, K = 4];

        let table = Tensor::<i32, m![W, V], Typecheck>::empty();
        let index = Tensor::<i32, m![K], Typecheck>::empty();
        let mut output = Tensor::<i32, m![K, V], Typecheck>::empty();
        table.write_gather::<_, _>(&mut output, &index, true);
    }

    #[test]
    fn npu_to_buf_returns_plain_values() {
        axes![A = 2];

        let tensor = Tensor::<i32, m![A], Npu>::from_buf(vec![1, 2]);

        assert_eq!(tensor.to_buf(), vec![1, 2]);
    }
}
