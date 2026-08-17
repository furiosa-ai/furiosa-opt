//! Backend abstraction: the storage-op seam for tensor operations.
//!
//! A [`Backend`] picks the concrete tensor storage type and implements every tensor operation
//! against it. The concrete impls are [`Cpu`] (host-side buffer emulator) and [`Npu`] (real
//! device). The cfg-selected [`crate::runtime::CurrentBackend`] alias picks which one the crate
//! compiles against.
//! Device-function execution (`launch`, `DeviceFn`, `DeviceSend`) lives in the sibling
//! [`crate::runtime`] module.

pub(crate) mod op_prep;

mod cpu;
/// NPU backend.
pub mod npu;

use std::fmt::Debug;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::cast::{ContractionAccumulator, ContractionCast};
use crate::scalar::{MaterializableScalar, Scalar};
use crate::tensor::memory::{HbmTensor, HostTensor};

pub use cpu::Cpu;
pub use npu::Npu;

/// Backend for tensor operations.
///
/// A backend picks the concrete tensor storage type (`Self::Storage<D>`) and the DMA
/// protocol. The storage type is opaque (`Clone + Debug` only); every operation on it, both the
/// lifecycle constructors/serializers and the value-ops (map, reduce, zip_with, scatter, ...),
/// is a method on this trait, delegating to the concrete storage's inherent methods.
///
/// `Cpu`: interprets ops on `BufStorage`'s physical staging buffer (default).
/// `Npu`: owns native staging buffers and does real PCIe DMA; the host never interprets ops.
///
/// The host backend (Cpu) implements the DMA as a plain host-side `transpose`
/// passthrough; `Npu` provides real device transfers. DMA stays a required method (no default) so
/// each backend states its choice explicitly.
pub trait Backend: Sized + 'static {
    /// Concrete per-backend tensor type. Opaque:
    /// only `Clone + Debug + PartialEq` (so the memory-tier `#[derive(Debug)]` structs and the
    /// `Tensor<D, Mapping, B>` wrapper's `Debug`/`PartialEq` resolve without a per-backend `where`
    /// clause). `PartialEq` (not `Eq`) because `f32` tensors must compare. All operations on the
    /// storage are methods on this trait.
    type Storage<D: Scalar>: Clone + Debug + PartialEq;

    /// Build a storage from a flat `D` buffer in `mapping`-order, delegating to the concrete
    /// storage's inherent constructor. The runtime `mapping` carries the layout, construction is
    /// the type→runtime boundary, so the wrapper converts its type param to a value here.
    fn from_vec<D: Scalar>(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self::Storage<D>;

    /// Build a storage from its dense physical device byte image (`mapping`-order, packed on `D::BITS`):
    /// the inverse of [`Self::into_buf`]. The default decodes the packed bytes to a logical `Vec<D>` and
    /// packs via [`Self::from_vec`]; a backend whose storage already IS the packed image (Cpu /
    /// Npu's [`crate::storage::BufStorage`]) overrides this to store the bytes directly, no decode / re-pack. Pre-packed
    /// data (fp4 / f4e2m1 weights) enters through here. Backs [`crate::tensor::Tensor::from_buf`].
    fn from_buf<D: MaterializableScalar>(mapping: &MappingValue, buf: Vec<u8>) -> Self::Storage<D> {
        Self::from_vec(mapping, (0..mapping.size()).map(|i| D::load(&buf, i)))
    }

    /// A zeroed storage of this layout, delegating to the concrete storage's inherent
    /// constructor.
    fn zeroed<D: Scalar>(mapping: &MappingValue) -> Self::Storage<D>;

    /// A fresh HBM tensor with no source (see [`crate::tensor::memory::HbmTensor::new`]).
    ///
    /// Required, no default: whether an HBM tensor needs a device allocation is a backend's own call.
    /// `Npu` takes one and hands its ownership to the tensor; the host-side backends hold bytes.
    fn alloc_hbm<D: Scalar, Chip: M, Element: M>() -> HbmTensor<D, Chip, Element, Self>;

    /// Serialize the storage to a flat `D` buffer in `mapping`-order (the physical / wire layout),
    /// consuming the storage, delegating to the concrete storage's inherent serializer.
    fn into_vec<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<D>;

    /// Serialize the storage to its dense physical device byte image (`mapping`-order, packed on
    /// `D::BITS`): the exact bytes that sit in HBM / DM, half the logical length for a 4-bit `D`. The
    /// default decodes to a logical `Vec<D>` and re-packs via [`Scalar::to_buf`]; a backend whose
    /// storage already IS the packed image (Cpu / Npu's [`crate::storage::BufStorage`]) overrides this to move the
    /// internal bytes out directly. Backs [`crate::tensor::memory::HbmTensor::to_buf`].
    fn into_buf<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<u8> {
        D::to_buf(&Self::into_vec(storage, mapping))
    }

    /// Element-wise map to a new scalar, preserving layout. The closure is bare-`D`; each backend
    /// applies its own per-storage strategy; Cpu/Npu walk the `BufStorage` buffer over bare
    /// `D`. Mapping-free: elementwise ops preserve the source
    /// layout, so no mapping type parameter is needed.
    /// The closure must be `Fn + Sync` (not `FnMut`): an elementwise map has no cross-cell state, and
    /// [`Cpu`] folds the cells across the rayon pool.
    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl Fn(D) -> D2 + Sync) -> Self::Storage<D2>;

    /// Element-wise zip of two same-layout tensors to a new scalar. See [`Self::map`]; the result is
    /// defined only where both inputs are.
    fn zip_with<D: MaterializableScalar, D2: MaterializableScalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> Self::Storage<D3>;

    /// Element-wise zip of three same-layout tensors to a new scalar, the ternary peer of
    /// [`Self::map`] (1-ary) and [`Self::zip_with`] (2-ary). The result is defined only where all
    /// three inputs are. The VE engine uses it for branch-conditional combine: it passes the
    /// execution-id `tag` as the third operand and resolves the slot's `TagGuard`/select inside the closure,
    /// so the backend stays free of VE concepts.
    fn zip3_with<D: MaterializableScalar, D2: MaterializableScalar, D3: MaterializableScalar, D4: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        c: &Self::Storage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> Self::Storage<D4>;

    /// Writes a transposed/broadcast view of `src` into `dst`. `src_offset` / `dst_offset` allow
    /// operating on partial views; `src_map` / `dst_map` are each storage's base (live-axis) mapping,
    /// where a tiled axis the `Src` / `Dst` *type* carries as padding stays a live `Symbol`, needed
    /// to resolve a partial-view offset's physical wire base. Each backend applies its own per-storage
    /// strategy: Cpu/Npu resolve the offset bases against the maps and drive a
    /// sequencer over the `BufStorage` buffer.
    #[allow(clippy::too_many_arguments)]
    fn transpose<D: Scalar, Src: M, Dst: M>(
        dst: &mut Self::Storage<D>,
        src: &Self::Storage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &MappingValue,
        dst_map: &MappingValue,
        allow_broadcast: bool,
    );

    /// Reduces the factors of `src`'s mapping absent in `Dst`, returning the result. Axes that `Dst`
    /// adds beyond the reduced source are broadcast (the reduced value repeats across them).
    /// `allow_broadcast` gates that, exactly like [`Self::transpose`]'s flag: `false` is a
    /// plain reduce that asserts `Dst` is a pure factor and *rejects* an unintended extra axis;
    /// `true` permits the broadcast axes. The reduce fn works over bare scalars; the backend lifts
    /// it to its cell representation.
    ///
    /// `reduce_fn` should be associative: the parallel backends may regroup the fold across threads. An
    /// associative fn gives a deterministic, serial-identical result; a non-associative one (saturating
    /// integer add, float add/mul) can differ from a serial left-fold and vary run-to-run on
    /// [`Cpu`], which regroups the fold across the rayon pool.
    fn reduce<D: MaterializableScalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D>;

    /// Contracts `lhs` and `rhs` into an `out`-shaped result: a generalized matmul
    /// `out[..] = sum over the axes of pre_reduce absent from out of (lhs * rhs)`. `pre_reduce` is the
    /// full (un-reduced) index space; `lhs_map` / `rhs_map` / `out` are its sub-mappings (the contracted
    /// axes are those in `pre_reduce` but not `out`).
    ///
    /// A required primitive, not a default: every backend fuses this into a single accumulating pass
    /// over `out`-many cells (`O(out)` memory), never materializing the `pre_reduce`-shaped outer
    /// product (`O(pre_reduce)`, which for a wide-output GEMM is the hundreds-of-MB dense buffer this
    /// primitive exists to avoid). The mappings arrive as runtime values because the engine-level
    /// contraction pipeline ([`crate::engine::contraction`]) only has them as values at the point it
    /// defers the fold; `lhs`/`rhs` keep their own compact (un-broadcast) layout, read through
    /// `lhs_map`/`rhs_map`.
    ///
    /// `pre_reduce` is not simply "the union of `lhs_map`'s and `rhs_map`'s axes": the engine-level
    /// caller can reconcile the two operands' contracted portions through DIFFERENT symbols (e.g. one
    /// operand's `Time`/`Packet` against the other's `TrfElement`, per the hardware layout-expansion
    /// contract `contract_outer` validates), which a naive per-symbol union of the two maps cannot
    /// recover. So it is a required, independently-computed argument, not derived here.
    ///
    /// Like [`Self::reduce`], [`Cpu`]'s per-output-cell sum regroups across threads, so its
    /// `f32` result can differ from serial and vary run-to-run.
    ///
    /// Operands arrive at the engine's operand width and accumulate in [`ContractionCast::Output`].
    /// [`Self::contraction_prewidened`] is the same fold over operands already at that width.
    fn contraction<D: ContractionCast + MaterializableScalar>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &MappingValue,
        rhs_map: &MappingValue,
        pre_reduce: &MappingValue,
        out: &MappingValue,
    ) -> Self::Storage<D>;

    /// [`Self::contraction`] over operands already at [`ContractionAccumulator`] width, so its widen and
    /// narrow are the identity. This is the pipeline's own fold, at `contract_lane`.
    fn contraction_prewidened<D: ContractionAccumulator>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &MappingValue,
        rhs_map: &MappingValue,
        pre_reduce: &MappingValue,
        out: &MappingValue,
    ) -> Self::Storage<D>;

    /// Scatters `src` into `dst` at positions read from the same-backend `i32` index tensor.
    fn scatter<D: Scalar, Src: M, Key: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    );

    /// Gathers from `src` (table) into `dst` at positions read from the index tensor.
    fn gather<D: MaterializableScalar, Src: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    );

    /// Reinterprets `src`'s buffer under a new mapping `Dst`, returning the result. A relabeling, not
    /// a data move: `BufStorage` keeps the same physical buffer, pairing wire position `i` under
    /// `Src` with wire position `i` under `Dst`.
    ///
    /// Memory-safe and length-checked (asserts `Src::SIZE == Dst::SIZE`). Whether the relabel is
    /// *meaningful* is the caller's logical precondition, discharged by the `unsafe` public `reshape`
    /// wrappers: `Src` and `Dst` must enumerate the elements in the SAME physical (wire) order, i.e.
    /// `Src::indexes()` and `Dst::indexes()` line the live elements up identically. Regrouping axes
    /// (merge/split) preserves wire order and is valid; a permutation (e.g. `m![A, B]` → `m![B, A]`)
    /// does NOT and must use `transpose`. This method itself only guarantees memory safety.
    fn reshape<D: Scalar, Src: M, Dst: M>(src: &Self::Storage<D>) -> Self::Storage<D>;

    /// Relabels `storage` from `src_map` to `dst_map` (`Tensor::transmute`). Unlike `reshape`, the two
    /// maps may differ in size: the vector engine relabels a padded read down to its compact slot
    /// (`…tile(k).read().transmute()`), so `dst_map.size() <= src_map.size()` is intended.
    ///
    /// `Cpu` / `Npu` COMPACT the
    /// physical buffer: they keep only the live (non-padding) cells of the `src_map` layout, in
    /// `dst_map` order, producing a `dst_map.size()`-length buffer (a no-op when no padding is dropped).
    fn transmute<D: MaterializableScalar, Src: M, Dst: M>(
        storage: Self::Storage<D>,
        src_map: &MappingValue,
        dst_map: &MappingValue,
    ) -> Self::Storage<D>;

    /// Transfer host tensor to HBM. Host-only: invoked from test/setup code over PCIe DMA, never
    /// from device kernel MIR, so no `#[primitive(...)]` annotation is needed.
    fn to_hbm<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
    ) -> impl std::future::Future<Output = HbmTensor<D, Chip, Element2, Self>>;

    /// Transfer HBM tensor to host. Host-only; see the `to_hbm` note above.
    fn from_hbm<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> impl std::future::Future<Output = HostTensor<D, Element2, Self>>;

    /// Bind the process to a logical [`Device`](crate::context::Device) before any host I/O. Only the
    /// NPU backend acts on this; backends without a physical device default to a no-op.
    fn bind_device(_device: crate::context::Device) {}
}
