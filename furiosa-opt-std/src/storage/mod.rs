//! Concrete backend storage types and their host-side capabilities.
//!
//! The layer below [`crate::tensor`]: each [`crate::backend::Backend`] picks one of these as its
//! `Storage` (Emulation/Npu → [`BufStorage`], Typecheck → [`PhantomStorage`]). The `Tensor`
//! wrapper drives them through the `Backend` seam; the storage types own their buffers and
//! implement the lifecycle constructors/serializers and value-ops as inherent methods.

use abi_stable::std_types::RResult;
use furiosa_mapping::*;

mod buf;
mod par_iters;
mod phantom;

pub(crate) use buf::Buf;
pub use buf::BufStorage;
pub use phantom::PhantomStorage;

/// Minimum work per parallel job, in scalar operations (multiply-adds / fold steps), for the storages'
/// rayon-parallel ops (`contraction`, `reduce`, `transpose`): rayon never splits below this, so a small
/// op stays one (effectively serial) job rather than fragmenting into micro-jobs whose dispatch overhead
/// would dominate (acute on a many-core box, where rayon splits aggressively). Each call site converts
/// it to its own split unit: an op whose split element does one scalar op (a stream position) uses it as
/// `with_min_len` directly, while a per-output-cell op divides by the inner block length so a job still
/// does about `PAR_MIN_JOB` scalar ops.
pub(crate) const PAR_MIN_JOB: usize = 1 << 16;

/// The `with_min_len` for a per-output-cell parallel op: how many output cells make one job do about
/// [`PAR_MIN_JOB`] scalar ops, given each cell folds an `inner_block`-length run. Clamped to `>= 1` so a
/// huge inner block does not yield `with_min_len(0)` (rayon would then over-split). Shared by the
/// storages' per-cell `reduce` / `contraction` / `gather`.
pub(crate) fn min_cells_per_job(inner_block: usize) -> usize {
    (PAR_MIN_JOB / inner_block.max(1)).max(1)
}

// Concrete per-backend storage types.
//
// Two concrete storage types, [`BufStorage`] (Emulation / Npu: physical `Vec<D>` staging buffer)
// and [`PhantomStorage`] (Typecheck: axes only), each own their storage and implement the
// lifecycle constructors/serializers and value-ops as inherent methods. The single backend seam
// ([`crate::backend::Backend`]) delegates to these.
//
// The storage `Mapping` type parameter is fully erased: each concrete storage is `XStorage<D>` and
// the backend GAT is `Storage<D>`. Element-wise ops (`map` / `zip_with` / `zip3_with`) are
// storage-native and mapping-free (they preserve the source layout). Mapping-changing ops
// (`transpose` / `reduce` / `scatter` / `gather` / `reshape`) keep their source/target mapping
// *type* parameters, the storage no longer carries the mapping, so these take it explicitly to
// drive the relayout.
//
// ## Offset address space is backend-private
//
// Where a per-backend body addresses storage by a flat offset (e.g. `BufStorage`'s `self.get(i)`),
// that address space is NOT portable across backends: `BufStorage` offsets are physical
// (`Mapping::SIZE`, padding included); `PhantomStorage` has none. The only contract is: two tensors
// of the **same backend and mapping** address the same logical position by the same offset. That
// suffices for the element-wise ops (same mapping) and for the per-backend restructuring bodies
// (which relate two mappings using their own backend's convention).

/// Resolves an `Index` to a multi-dim coordinate vector against `axes`. Returns `None` when the
/// index lands in a padding slot. `finalize` gives each symbol's absolute coordinate; recover this
/// axis's digit by its stride/modulo, since an axis may be a sub-factor of its symbol (`A % 4`).
///
/// Shared by [`BufStorage`]'s index decode and the vector engine's `AxisToggle` host pattern.
pub(crate) fn axis_coords(axes: &[AxisTerm], index: Index) -> Option<Vec<usize>> {
    let RResult::ROk(coords) = index.finalize() else {
        return None;
    };
    Some(
        axes.iter()
            .map(|axis| (coords.get(&axis.symbol).copied().unwrap_or(0) / axis.stride) % axis.modulo)
            .collect(),
    )
}

/// The position of the live axis term whose symbol is `axis` within `axes`, or `None` if absent.
/// Lets [`BufStorage`] map a per-cell coordinate vector to a single axis's coordinate when
/// building an axis-index tensor.
pub(crate) fn axis_position(axes: &[AxisTerm], axis: &Ident) -> Option<usize> {
    axes.iter().position(|term| term.symbol == *axis)
}
