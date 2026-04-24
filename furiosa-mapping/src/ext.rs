//! Impl traits exposing Mapping/FMapping/Index/Division methods via the `furiosa-mapping-impl` shared library.
//!
//! This trait is to invoke the methods of the types as if they are defined by `impl for T`, i.e. the method invocation
//! notation `x.foo()`, not a function call `foo(x)`.  Note that the implementation of `impl for T` should be in the
//! same crate with `T` due to the orphan rule, but we would like to provide/hide the method definitions as a separate
//! library.

use abi_stable::std_types::{ROption, RResult, RSlice, RVec, Tuple2};
use furiosa_mapping_types::{
    Atom, Division, DivisionError, DivisionMode, DivisionSide, FMapping, Factor, Ident, Mapping, PaddingKind,
    RSortedMap, Relaxed, Span, Strict, Term, TermBounds, TermPosition,
};

use crate::{Index, IndexValueError};

#[expect(improper_ctypes, reason = "all types are #[repr(C)] + StableAbi")]
unsafe extern "C-unwind" {
    fn factor_padding(size: usize, kind: PaddingKind) -> Factor;
    fn factor_idents(slf: &Factor) -> RVec<Ident>;
    fn atom_size(slf: &Atom) -> usize;
    fn atom_idents(slf: &Atom) -> RVec<Ident>;
    fn atom_contains_ident(slf: &Atom, ident: Ident) -> bool;
    fn atom_find_symbol_size(slf: &Atom, ident: Ident) -> ROption<usize>;
    fn term_depth(slf: &Term) -> usize;
    fn term_idents(slf: &Term) -> RVec<Ident>;
    fn mapping_size(slf: &Mapping) -> usize;
    fn mapping_pair(slf: Mapping, other: Mapping) -> Mapping;
    fn mapping_pairs(ms: RVec<Mapping>) -> Mapping;
    fn mapping_divide_span(slf: &Mapping, divisor: &Mapping, span: usize) -> RResult<Division<Span>, DivisionError>;
    fn mapping_divide_relaxed(slf: &Mapping, divisor: &Mapping) -> Division<Relaxed>;
    fn mapping_divide_strict(slf: &Mapping, divisor: &Mapping) -> Division<Strict>;
    fn mapping_factorize(slf: &Mapping) -> FMapping;
    fn fmapping_from_axes(axes: RSlice<Term>) -> FMapping;
    fn fmapping_pop(slf: &mut FMapping) -> ROption<Factor>;
    fn fmapping_into_inner(slf: FMapping) -> RVec<Factor>;
    fn fmapping_is_padding(slf: &FMapping) -> bool;
    fn fmapping_has_terms(slf: &FMapping) -> bool;
    fn fmapping_terms_with_stride(slf: &FMapping) -> RVec<TermPosition>;
    fn fmapping_factors(slf: &FMapping) -> RSlice<'_, Factor>;
    fn fmapping_contains_ident(slf: &FMapping, ident: Ident) -> bool;
    fn fmapping_idents(slf: &FMapping) -> RVec<Ident>;
    fn fmapping_find_symbol_size(slf: &FMapping, ident: Ident) -> ROption<usize>;
    fn fmapping_eval(slf: &FMapping, position: usize) -> Index;
    fn fmapping_is_ident_isolated(slf: &FMapping, ident: Ident) -> bool;
    fn fmapping_into_factor(slf: FMapping) -> Factor;
    fn fmapping_mul(slf: FMapping, inner: FMapping) -> FMapping;
    fn fmapping_stride(slf: FMapping, stride: usize) -> FMapping;
    fn fmapping_modulo(slf: FMapping, modulo: usize) -> FMapping;
    fn fmapping_is_resize_of(slf: &FMapping, original: &FMapping) -> bool;
    fn fmapping_padding(slf: FMapping, padding: usize, kind: PaddingKind) -> FMapping;
    fn fmapping_to_mapping(slf: &FMapping) -> Mapping;
    fn fmapping_divide_span(slf: FMapping, divisor: FMapping, span: usize) -> RResult<Division<Span>, DivisionError>;
    fn fmapping_divide_relaxed(slf: FMapping, divisor: FMapping) -> Division<Relaxed>;
    fn fmapping_divide_strict(slf: FMapping, divisor: FMapping) -> Division<Strict>;
    fn fmapping_normalize(slf: FMapping) -> FMapping;
    fn fmapping_remove_padding(slf: FMapping) -> FMapping;
    fn fmapping_split_at(slf: &FMapping, target: usize) -> Tuple2<FMapping, FMapping>;
    fn fmapping_pad(slf: FMapping, target: usize) -> FMapping;
    fn division_exact_relaxed(slf: Division<Relaxed>) -> RResult<Division<Relaxed>, DivisionError>;
    fn division_exact_strict(slf: Division<Strict>) -> RResult<Division<Strict>, DivisionError>;
    fn division_exact_span(slf: Division<Span>) -> RResult<Division<Span>, DivisionError>;
    fn division_strict_remainder(slf: &Division<Strict>, side: DivisionSide) -> FMapping;
    fn division_strict_bounds(slf: &Division<Strict>) -> RVec<TermBounds>;
    fn index_new() -> Index;
    fn index_add(slf: &mut Index, other: Index);
    fn index_mark_invalid(slf: &mut Index);
    fn index_add_term(slf: &mut Index, term: Term, value: usize);
    fn index_add_mapping(slf: &mut Index, mapping: FMapping, value: usize);
    fn index_ident_value(slf: &Index, ident: Ident) -> RResult<usize, IndexValueError>;
    fn index_finalize(slf: Index) -> RResult<RSortedMap<Term, usize>, ()>;
    fn index_gen_indexes(slf: &Index, mapping: FMapping) -> RVec<Index>;
}

// ── MappingExt ────────────────────────────────────────────────────────────────

/// Methods for [`Mapping`].
pub trait MappingExt: Sized {
    /// Returns the size of the mapping expression.
    fn size(&self) -> usize;
    /// Pairs two mapping expressions and simplifies.
    fn pair(self, other: Mapping) -> Mapping;
    /// Pairs multiple mapping expressions and simplifies.
    fn pairs(ms: RVec<Mapping>) -> Mapping;
    /// Divides the mapping expression by a single-term divisor with a span.
    fn divide_span(&self, divisor: &Mapping, span: usize) -> RResult<Division<Span>, DivisionError>;
    /// Divides the mapping expression without exposing analysis.
    fn divide_relaxed(&self, divisor: &Mapping) -> Division<Relaxed>;
    /// Divides the mapping expression.
    fn divide_strict(&self, divisor: &Mapping) -> Division<Strict>;
    /// Factorizes the mapping expression into factor representation.
    fn factorize(&self) -> FMapping;
}

impl MappingExt for Mapping {
    fn size(&self) -> usize {
        unsafe { mapping_size(self) }
    }
    fn pair(self, other: Mapping) -> Mapping {
        unsafe { mapping_pair(self, other) }
    }
    fn pairs(ms: RVec<Mapping>) -> Mapping {
        unsafe { mapping_pairs(ms) }
    }
    fn divide_span(&self, divisor: &Mapping, span: usize) -> RResult<Division<Span>, DivisionError> {
        unsafe { mapping_divide_span(self, divisor, span) }
    }
    fn divide_relaxed(&self, divisor: &Mapping) -> Division<Relaxed> {
        unsafe { mapping_divide_relaxed(self, divisor) }
    }
    fn divide_strict(&self, divisor: &Mapping) -> Division<Strict> {
        unsafe { mapping_divide_strict(self, divisor) }
    }
    fn factorize(&self) -> FMapping {
        unsafe { mapping_factorize(self) }
    }
}

// ── FMappingExt ───────────────────────────────────────────────────────────────

/// Methods for [`FMapping`].
pub trait FMappingExt: Sized {
    /// Creates a new empty factor mapping.
    fn new() -> Self;
    /// Creates a factor mapping from a slice of terms (axes).
    fn from_axes(axes: RSlice<Term>) -> Self;
    /// Pops the outermost factor of the mapping expression.
    fn pop(&mut self) -> ROption<Factor>;
    /// Converts the factor mapping into a vector of factors.
    fn into_inner(self) -> RVec<Factor>;
    /// Checks if the factor mapping is only padding.
    fn is_padding(&self) -> bool;
    /// Returns true if any factor is a Term (not Padding).
    fn has_terms(&self) -> bool;
    /// Extracts each Term with its effective stride in this FMapping.
    fn terms_with_stride(&self) -> RVec<TermPosition>;
    /// Returns a reference to the factors (innermost first).
    fn factors(&self) -> RSlice<'_, Factor>;
    /// Returns true if any term in this FMapping contains the given ident.
    fn contains_ident(&self, ident: Ident) -> bool;
    /// Returns the unique idents referenced in this FMapping.
    fn idents(&self) -> RVec<Ident>;
    /// Returns the original declared size of the given ident's Symbol.
    fn find_symbol_size(&self, ident: Ident) -> ROption<usize>;
    /// Evaluates this FMapping at the given position.
    fn eval(&self, position: usize) -> Index;
    /// Returns true if the given ident is isolated.
    fn is_ident_isolated(&self, ident: Ident) -> bool;
    /// Converts the factor mapping into a single factor.
    fn into_factor(self) -> Factor;
    /// Multiplies two factor mappings (composes outer × inner).
    fn mul(self, inner: FMapping) -> FMapping;
    /// Applies stride to the factor mapping.
    fn stride(self, stride: usize) -> FMapping;
    /// Applies modulo to the factor mapping.
    fn modulo(self, modulo: usize) -> FMapping;
    /// Returns true if `self` is a resize of `original`.
    fn is_resize_of(&self, original: &FMapping) -> bool;
    /// Converts the factor mapping into a mapping expression.
    fn to_mapping(&self) -> Mapping;
    /// Divides the factor mapping by a single-term divisor with a span.
    fn divide_span(self, divisor: FMapping, span: usize) -> RResult<Division<Span>, DivisionError>;
    /// Divides the factor mapping with read-accessible (Top) matched-hole padding.
    fn divide_relaxed(self, divisor: FMapping) -> Division<Relaxed>;
    /// Divides the factor mapping with analysis-capable (Bottom) matched-hole padding.
    fn divide_strict(self, divisor: FMapping) -> Division<Strict>;
    /// Normalizes the factor mapping via round-trip through `Mapping`.
    fn normalize(self) -> FMapping;
    /// Removes all right padding.
    fn remove_padding(self) -> FMapping;
    /// Splits the mapping at `target` from the innermost side.
    fn split_at(&self, target: usize) -> Tuple2<FMapping, FMapping>;
    /// Removes existing padding and pads to `target` size.
    fn pad(self, target: usize) -> FMapping;
    /// Applies padding to the factor mapping.
    fn padding(self, padding: usize, kind: PaddingKind) -> FMapping;
}

impl FMappingExt for FMapping {
    fn new() -> Self {
        FMapping::new()
    }
    fn from_axes(axes: RSlice<Term>) -> Self {
        unsafe { fmapping_from_axes(axes) }
    }
    fn pop(&mut self) -> ROption<Factor> {
        unsafe { fmapping_pop(self) }
    }
    fn into_inner(self) -> RVec<Factor> {
        unsafe { fmapping_into_inner(self) }
    }
    fn is_padding(&self) -> bool {
        unsafe { fmapping_is_padding(self) }
    }
    fn has_terms(&self) -> bool {
        unsafe { fmapping_has_terms(self) }
    }
    fn terms_with_stride(&self) -> RVec<TermPosition> {
        unsafe { fmapping_terms_with_stride(self) }
    }
    fn factors(&self) -> RSlice<'_, Factor> {
        unsafe { fmapping_factors(self) }
    }
    fn contains_ident(&self, ident: Ident) -> bool {
        unsafe { fmapping_contains_ident(self, ident) }
    }
    fn idents(&self) -> RVec<Ident> {
        unsafe { fmapping_idents(self) }
    }
    fn find_symbol_size(&self, ident: Ident) -> ROption<usize> {
        unsafe { fmapping_find_symbol_size(self, ident) }
    }
    fn eval(&self, position: usize) -> Index {
        unsafe { fmapping_eval(self, position) }
    }
    fn is_ident_isolated(&self, ident: Ident) -> bool {
        unsafe { fmapping_is_ident_isolated(self, ident) }
    }
    fn into_factor(self) -> Factor {
        unsafe { fmapping_into_factor(self) }
    }
    fn mul(self, inner: FMapping) -> FMapping {
        unsafe { fmapping_mul(self, inner) }
    }
    fn stride(self, stride: usize) -> FMapping {
        unsafe { fmapping_stride(self, stride) }
    }
    fn modulo(self, modulo: usize) -> FMapping {
        unsafe { fmapping_modulo(self, modulo) }
    }
    fn is_resize_of(&self, original: &FMapping) -> bool {
        unsafe { fmapping_is_resize_of(self, original) }
    }
    fn to_mapping(&self) -> Mapping {
        unsafe { fmapping_to_mapping(self) }
    }
    fn divide_span(self, divisor: FMapping, span: usize) -> RResult<Division<Span>, DivisionError> {
        unsafe { fmapping_divide_span(self, divisor, span) }
    }
    fn divide_relaxed(self, divisor: FMapping) -> Division<Relaxed> {
        unsafe { fmapping_divide_relaxed(self, divisor) }
    }
    fn divide_strict(self, divisor: FMapping) -> Division<Strict> {
        unsafe { fmapping_divide_strict(self, divisor) }
    }
    fn normalize(self) -> FMapping {
        unsafe { fmapping_normalize(self) }
    }
    fn remove_padding(self) -> FMapping {
        unsafe { fmapping_remove_padding(self) }
    }
    fn split_at(&self, target: usize) -> Tuple2<FMapping, FMapping> {
        unsafe { fmapping_split_at(self, target) }
    }
    fn pad(self, target: usize) -> FMapping {
        unsafe { fmapping_pad(self, target) }
    }
    fn padding(self, padding: usize, kind: PaddingKind) -> FMapping {
        unsafe { fmapping_padding(self, padding, kind) }
    }
}

// ── AtomExt ───────────────────────────────────────────────────────────────────

/// Methods for [`Atom`].
pub trait AtomExt {
    /// Returns the size of the atomic mapping expression.
    fn size(&self) -> usize;
    /// Returns the idents contained in this atom.
    fn idents(&self) -> RVec<Ident>;
    /// Returns true if this atom contains the given ident (recursively).
    fn contains_ident(&self, ident: Ident) -> bool;
    /// Returns the original declared size of the given ident's Symbol.
    fn find_symbol_size(&self, ident: Ident) -> ROption<usize>;
}

impl AtomExt for Atom {
    fn size(&self) -> usize {
        unsafe { atom_size(self) }
    }
    fn idents(&self) -> RVec<Ident> {
        unsafe { atom_idents(self) }
    }
    fn contains_ident(&self, ident: Ident) -> bool {
        unsafe { atom_contains_ident(self, ident) }
    }
    fn find_symbol_size(&self, ident: Ident) -> ROption<usize> {
        unsafe { atom_find_symbol_size(self, ident) }
    }
}

// ── TermExt ───────────────────────────────────────────────────────────────────

/// Methods for [`Term`].
pub trait TermExt {
    /// Returns the depth of the term.
    fn depth(&self) -> usize;
    /// Returns the idents contained in this term.
    fn idents(&self) -> RVec<Ident>;
}

impl TermExt for Term {
    fn depth(&self) -> usize {
        unsafe { term_depth(self) }
    }
    fn idents(&self) -> RVec<Ident> {
        unsafe { term_idents(self) }
    }
}

// ── FactorExt ─────────────────────────────────────────────────────────────────

/// Methods for [`Factor`].
pub trait FactorExt {
    /// Creates a padding factor.
    fn padding(size: usize, kind: PaddingKind) -> Factor;
    /// Returns the idents contained in this factor.
    fn idents(&self) -> RVec<Ident>;
}

impl FactorExt for Factor {
    fn padding(size: usize, kind: PaddingKind) -> Factor {
        unsafe { factor_padding(size, kind) }
    }
    fn idents(&self) -> RVec<Ident> {
        unsafe { factor_idents(self) }
    }
}

// ── IndexExt ──────────────────────────────────────────────────────────────────

/// Methods for [`Index`].
pub trait IndexExt: Sized {
    /// Creates a new empty index.
    fn new() -> Self;
    /// Adds another index to this index.
    fn add(&mut self, other: Index);
    /// Marks this index as invalid.
    fn mark_invalid(&mut self);
    /// Adds a term to this index.
    fn add_term(&mut self, term: Term, value: usize);
    /// Adds a mapping to this index.
    fn add_mapping<I: crate::M>(&mut self, value: usize);
    /// Returns the contribution of a specific ident to this Index.
    fn ident_value(&self, ident: Ident) -> RResult<usize, IndexValueError>;
    /// Finalizes the index.
    fn finalize(self) -> RResult<RSortedMap<Term, usize>, ()>;
    /// Generates all possible indexes based on the given mapping.
    fn gen_indexes(&self, mapping: FMapping) -> RVec<Index>;
}

impl IndexExt for Index {
    fn new() -> Self {
        unsafe { index_new() }
    }
    fn add(&mut self, other: Index) {
        unsafe { index_add(self, other) }
    }
    fn mark_invalid(&mut self) {
        unsafe { index_mark_invalid(self) }
    }
    fn add_term(&mut self, term: Term, value: usize) {
        unsafe { index_add_term(self, term, value) }
    }
    fn add_mapping<I: crate::M>(&mut self, value: usize) {
        unsafe { index_add_mapping(self, mapping_factorize(&I::to_value()), value) }
    }
    fn ident_value(&self, ident: Ident) -> RResult<usize, IndexValueError> {
        unsafe { index_ident_value(self, ident) }
    }
    fn finalize(self) -> RResult<RSortedMap<Term, usize>, ()> {
        unsafe { index_finalize(self) }
    }
    fn gen_indexes(&self, mapping: FMapping) -> RVec<Index> {
        unsafe { index_gen_indexes(self, mapping) }
    }
}

// ── DivisionExt ───────────────────────────────────────────────────────────────

/// Methods for [`Division<M>`].
pub trait DivisionExt<Md: DivisionMode> {
    /// Returns `Ok(self)` if all divisor terms were matched.
    fn exact(self) -> RResult<Division<Md>, DivisionError>;
}

impl DivisionExt<Relaxed> for Division<Relaxed> {
    fn exact(self) -> RResult<Division<Relaxed>, DivisionError> {
        unsafe { division_exact_relaxed(self) }
    }
}

impl DivisionExt<Strict> for Division<Strict> {
    fn exact(self) -> RResult<Division<Strict>, DivisionError> {
        unsafe { division_exact_strict(self) }
    }
}

impl DivisionExt<Span> for Division<Span> {
    fn exact(self) -> RResult<Division<Span>, DivisionError> {
        unsafe { division_exact_span(self) }
    }
}

/// Methods specific to [`Division<Strict>`].
pub trait StrictDivisionExt {
    /// Returns the original mapping with matched terms removed.
    fn remainder(&self, side: DivisionSide) -> FMapping;
    /// Returns per-term padding bounds reconstructed from the strict residue.
    fn bounds(&self) -> RVec<TermBounds>;
}

impl StrictDivisionExt for Division<Strict> {
    fn remainder(&self, side: DivisionSide) -> FMapping {
        unsafe { division_strict_remainder(self, side) }
    }

    /// Returns per-term padding bounds reconstructed from the strict residue.
    ///
    /// This reports the compact block bounds used by [`remainder`](Self::remainder).  For example, `m![A #
    /// 16].divide_strict(m![A]).bounds()` reports dividend bounds `4..16` and divisor bounds `4..4` when `A = 4`.
    ///
    /// ```rust
    /// use furiosa_mapping::*;
    /// use furiosa_opt_macro::m;
    ///
    /// axes![A = 4];
    /// let division = <m![A # 16]>::to_value()
    ///     .factorize()
    ///     .divide_strict(<m![A]>::to_value().factorize());
    /// let [row]: [TermBounds; 1] = division.bounds().into_vec().try_into().unwrap();
    ///
    /// assert_eq!(row.dividend, BlockBounds { min: 4, max: 16 });
    /// assert_eq!(row.divisor, BlockBounds { min: 4, max: 4 });
    /// ```
    fn bounds(&self) -> RVec<TermBounds> {
        unsafe { division_strict_bounds(self) }
    }
}
