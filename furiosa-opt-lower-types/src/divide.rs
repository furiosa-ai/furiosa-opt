//! FMapping-free results of a divide: the engines read strides, sizes, and idents without ever
//! naming the factor algebra they came from.

use abi_stable::StableAbi;
use abi_stable::std_types::{ROption, RVec};
use furiosa_mapping_types::{Ident, Mapping};

/// One factor leaf of a mapping, FMapping-free: a named axis (`ident`) or an untagged run (a composite
/// the matcher didn't flatten, or a padding over-read), with its live cell count. The IR fetch
/// projection reads these innermost-first instead of walking the hidden factor list.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct FactorLeaf {
    /// The axis symbol, or `None` for an untagged run (composite / padding).
    pub ident: ROption<Ident>,
    /// Live cells the leaf spans.
    pub size: usize,
}

/// One matched axis of a divide, FMapping-free: the engines read its strides, live size, and idents
/// without ever naming the factor algebra it came from. `dividend_stride` / `divisor_stride` are the
/// cumulative-product strides on each side; `resize` is the live cell count; `idents` are the symbols
/// the matched term carries.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct DivideTerm {
    /// Stride of the matched axis on the dividend (the larger mapping).
    pub dividend_stride: usize,
    /// Stride of the matched axis on the divisor (the carved piece).
    pub divisor_stride: usize,
    /// Live cells the matched axis spans.
    pub resize: usize,
    /// The unique symbols the matched term references.
    pub idents: RVec<Ident>,
}

/// The relaxed view of a divide (partial overlap allowed): the matched axes plus what is left over on
/// each side as a plain `Mapping`, and the contiguous reachable tail the DMA-layout check reads. All
/// FMapping-free — residues cross as `Mapping`, never as factors.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct RelaxedDivision {
    /// The axes carved out of the dividend by the divisor (the relaxed match).
    pub matched: RVec<DivideTerm>,
    /// What the divisor did not consume of the dividend — the "leftover" / broadcast quotient.
    pub dividend_residue: Mapping,
    /// What the dividend did not consume of the divisor.
    pub divisor_residue: Mapping,
    /// End (in dividend cells) of the contiguous run reachable from position 0.
    pub contiguous_tail: usize,
}
