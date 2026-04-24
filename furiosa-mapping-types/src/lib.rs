//! Mapping expressions.

#![feature(register_tool)]
#![register_tool(tcp)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]

mod sorted_map;
pub use sorted_map::RSortedMap;

use abi_stable::{
    StableAbi,
    std_types::{RBox, RResult, RVec},
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

use furiosa_mapping_macro::primitive;
use itertools::Itertools;

/// Axis identifiers.
#[primitive(mapping::Ident)]
#[repr(C)]
// SAFETY: &'static str is not formally ABI-stable, but its layout (*const u8, usize)
// is de facto stable across all Rust versions and extremely unlikely to change.
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[sabi(unsafe_opaque_fields)]
pub struct Ident(&'static str);

#[expect(missing_docs)]
impl Ident {
    /// Creates a new identifier.
    ///
    /// The identifier must start with an uppercase ASCII letter and contain
    /// only ASCII alphanumeric characters or underscores.
    pub const fn new(s: &'static str) -> Self {
        let b = s.as_bytes();
        assert!(!b.is_empty(), "Ident must not be empty");
        assert!(
            b[0].is_ascii_uppercase(),
            "Ident must start with an uppercase ASCII letter"
        );
        let mut i = 1;
        while i < b.len() {
            assert!(
                b[i].is_ascii_alphanumeric() || b[i] == b'_',
                "Ident must contain only ASCII alphanumeric or underscore characters"
            );
            i += 1;
        }
        Self(s)
    }

    /// Returns the string representation.
    pub fn as_str(&self) -> &'static str {
        self.0
    }

    pub const A: Self = Self("A");
    pub const B: Self = Self("B");
    pub const C: Self = Self("C");
    pub const D: Self = Self("D");
    pub const E: Self = Self("E");
    pub const F: Self = Self("F");
    pub const G: Self = Self("G");
    pub const H: Self = Self("H");
    pub const I: Self = Self("I");
    pub const J: Self = Self("J");
    pub const K: Self = Self("K");
    pub const L: Self = Self("L");
    pub const M: Self = Self("M");
    pub const N: Self = Self("N");
    pub const O: Self = Self("O");
    pub const P: Self = Self("P");
    pub const Q: Self = Self("Q");
    pub const R: Self = Self("R");
    pub const S: Self = Self("S");
    pub const T: Self = Self("T");
    pub const U: Self = Self("U");
    pub const V: Self = Self("V");
    pub const W: Self = Self("W");
    pub const X: Self = Self("X");
    pub const Y: Self = Self("Y");
    pub const Z: Self = Self("Z");
}

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Ident> for &'static str {
    fn from(value: Ident) -> Self {
        value.0
    }
}

impl serde::Serialize for Ident {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Ident {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s: String = serde::Deserialize::deserialize(deserializer)?;
        Ident::try_from(s.as_str()).map_err(|e| serde::de::Error::custom(format!("invalid Ident: {e}")))
    }
}

impl serde_lite::Deserialize for Ident {
    fn deserialize(val: &serde_lite::Intermediate) -> Result<Self, serde_lite::Error> {
        let s: String = serde_lite::Deserialize::deserialize(val)?;
        Ident::try_from(s.as_str()).map_err(|e| serde_lite::Error::custom(format!("invalid Ident: {e}")))
    }
}

impl<'a> TryFrom<&'a str> for Ident {
    type Error = &'a str;

    fn try_from(value: &'a str) -> std::result::Result<Self, Self::Error> {
        use lasso::ThreadedRodeo;
        use std::sync::LazyLock;
        static INTERNER: LazyLock<ThreadedRodeo> = LazyLock::new(ThreadedRodeo::new);

        let key = INTERNER.get_or_intern(value);
        let interned: &'static str = INTERNER.resolve(&key);
        std::panic::catch_unwind(|| Self::new(interned)).map_err(|_| value)
    }
}

/// Mapping expression enum.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Mapping {
    /// Identity mapping.
    Identity,
    /// Symbol mapping.
    Symbol {
        /// Symbol.
        symbol: Ident,
        /// Size.
        size: usize,
    },
    /// Stride mapping.
    Stride {
        /// Inner mapping.
        inner: RBox<Mapping>,
        /// Stride size.
        stride: usize,
    },
    /// Modulo mapping.
    Modulo {
        /// Inner mapping.
        inner: RBox<Mapping>,
        /// Stride size.
        modulo: usize,
    },
    /// Resize mapping.
    Resize {
        /// Inner mapping.
        inner: RBox<Mapping>,
        /// Truncate size.
        resize: usize,
    },
    /// Padding mapping.
    Padding {
        /// Inner mapping.
        inner: RBox<Mapping>,
        /// Size after padding.
        padding: usize,
        /// Accessibility of this padding region.
        kind: PaddingKind,
    },
    /// Pair mapping.
    Pair {
        /// Left mapping.
        left: RBox<Mapping>,
        /// Right mapping.
        right: RBox<Mapping>,
    },
}

impl Display for Mapping {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fn flatten_pair<'a>(acc: &mut Vec<&'a Mapping>, m: &'a Mapping) {
            match m {
                Mapping::Pair { left, right } => {
                    flatten_pair(acc, left);
                    flatten_pair(acc, right);
                }
                _ => acc.push(m),
            }
        }

        match self {
            Self::Identity => write!(f, "1"),
            Self::Symbol { symbol, size: _ } => {
                // We hide the size just for readability.
                write!(f, "{symbol}")
            }
            Self::Stride { inner, stride } => write!(f, "{inner} / {stride}"),
            Self::Modulo { inner, modulo } => write!(f, "{inner} % {modulo}"),
            Self::Resize { inner, resize } => write!(f, "{inner} = {resize}"),
            Self::Padding {
                inner,
                padding,
                kind: PaddingKind::Top,
            } => write!(f, "{inner} # {padding}"),
            Self::Padding {
                inner,
                padding,
                kind: PaddingKind::Bottom,
            } => write!(f, "{inner} #_ {padding}"),
            Self::Pair { left, right } => {
                // Collect all nested pairs and print them as flattened.
                let mut elements = vec![];
                flatten_pair(&mut elements, left);
                flatten_pair(&mut elements, right);
                write!(f, "({})", elements.iter().join(", "))
            }
        }
    }
}

/// Serde-compatible mirror of [`Mapping`] using `Box` instead of `RBox`, so that
/// the standard derive macros work. Used only for serialization/deserialization.
#[derive(serde::Serialize, serde::Deserialize, serde_lite::Deserialize)]
enum MappingSerde {
    Identity,
    Symbol {
        symbol: Ident,
        size: usize,
    },
    Stride {
        inner: Box<MappingSerde>,
        stride: usize,
    },
    Modulo {
        inner: Box<MappingSerde>,
        modulo: usize,
    },
    Resize {
        inner: Box<MappingSerde>,
        resize: usize,
    },
    Padding {
        inner: Box<MappingSerde>,
        padding: usize,
        kind: PaddingKind,
    },
    Pair {
        left: Box<MappingSerde>,
        right: Box<MappingSerde>,
    },
}

impl From<Mapping> for MappingSerde {
    fn from(m: Mapping) -> Self {
        match m {
            Mapping::Identity => Self::Identity,
            Mapping::Symbol { symbol, size } => Self::Symbol { symbol, size },
            Mapping::Stride { inner, stride } => Self::Stride {
                inner: Box::new(RBox::into_inner(inner).into()),
                stride,
            },
            Mapping::Modulo { inner, modulo } => Self::Modulo {
                inner: Box::new(RBox::into_inner(inner).into()),
                modulo,
            },
            Mapping::Resize { inner, resize } => Self::Resize {
                inner: Box::new(RBox::into_inner(inner).into()),
                resize,
            },
            Mapping::Padding { inner, padding, kind } => Self::Padding {
                inner: Box::new(RBox::into_inner(inner).into()),
                padding,
                kind,
            },
            Mapping::Pair { left, right } => Self::Pair {
                left: Box::new(RBox::into_inner(left).into()),
                right: Box::new(RBox::into_inner(right).into()),
            },
        }
    }
}

impl From<MappingSerde> for Mapping {
    fn from(m: MappingSerde) -> Self {
        match m {
            MappingSerde::Identity => Self::Identity,
            MappingSerde::Symbol { symbol, size } => Self::Symbol { symbol, size },
            MappingSerde::Stride { inner, stride } => Self::Stride {
                inner: RBox::new((*inner).into()),
                stride,
            },
            MappingSerde::Modulo { inner, modulo } => Self::Modulo {
                inner: RBox::new((*inner).into()),
                modulo,
            },
            MappingSerde::Resize { inner, resize } => Self::Resize {
                inner: RBox::new((*inner).into()),
                resize,
            },
            MappingSerde::Padding { inner, padding, kind } => Self::Padding {
                inner: RBox::new((*inner).into()),
                padding,
                kind,
            },
            MappingSerde::Pair { left, right } => Self::Pair {
                left: RBox::new((*left).into()),
                right: RBox::new((*right).into()),
            },
        }
    }
}

impl serde::Serialize for Mapping {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        MappingSerde::from(self.clone()).serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for Mapping {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        MappingSerde::deserialize(d).map(Into::into)
    }
}

impl serde_lite::Deserialize for Mapping {
    fn deserialize(val: &serde_lite::Intermediate) -> Result<Self, serde_lite::Error> {
        MappingSerde::deserialize(val).map(Into::into)
    }
}

/// Atomic mapping expression.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// Symbolic atomic mapping expression.
    Symbol {
        /// Symbol of the axis.
        symbol: Ident,
        /// Size of the axis.
        size: usize,
    },
    /// Composite mapping expression.
    Composite(RBox<FMapping>),
}

/// `inner / stride % modulo`.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term {
    /// Inner mapping expression.
    pub inner: Atom,
    /// Stride of the mapping.
    pub stride: usize,
    /// Modulo of the mapping.
    pub modulo: usize,
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            Atom::Symbol { symbol, size } => {
                if self.stride == 1 && self.modulo == *size {
                    write!(f, "{}", symbol)
                } else if self.stride == 1 {
                    write!(f, "{}%{}", symbol, self.modulo)
                } else if self.modulo == *size {
                    write!(f, "{}//{}", symbol, self.stride)
                } else {
                    write!(f, "({}//{})%{}", symbol, self.stride, self.modulo)
                }
            }
            Atom::Composite(inner) => {
                if self.stride == 1 && self.modulo == inner.size() {
                    write!(f, "({})", inner)
                } else if self.stride == 1 {
                    write!(f, "({})%{}", inner, self.modulo)
                } else if self.modulo == inner.size() {
                    write!(f, "({})//{}", inner, self.stride)
                } else {
                    write!(f, "(({})//{})%{}", inner, self.stride, self.modulo)
                }
            }
        }
    }
}

/// Kind of a padding factor.
#[repr(C)]
#[derive(
    StableAbi,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    serde_lite::Deserialize,
)]
pub enum PaddingKind {
    /// Accessible padding.
    Top,
    /// Inaccessible padding.
    Bottom,
}

/// Factor representation of a mapping expression.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Factor {
    /// Term.
    Term {
        /// Inner term.
        inner: Term,
        /// Resize of the mapping.
        resize: usize,
    },
    /// Padding.
    Padding {
        /// Size after padding.
        size: usize,
        /// Accessibility of this padding region.
        kind: PaddingKind,
    },
}

/// Factor mapping expression.
///
/// Factors are ordered from innermost (index 0) to outermost (last index).
/// `push`/`pop`/`last` operate on the outermost factor.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FMapping(pub RVec<Factor>);

impl Default for FMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl FMapping {
    /// Creates a new empty factor mapping.
    pub fn new() -> Self {
        Self(RVec::new())
    }

    /// Returns the size of the factor mapping.
    pub fn size(&self) -> usize {
        let mut x = 1;
        for term in self.0.iter().rev() {
            match term {
                Factor::Padding { size, .. } => {
                    return x * *size;
                }
                Factor::Term { resize, .. } => {
                    x *= *resize;
                }
            }
        }
        x
    }
}

impl Display for FMapping {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let terms: Vec<String> = self
            .0
            .iter()
            .map(|term| match term {
                Factor::Padding {
                    size,
                    kind: PaddingKind::Top,
                } => format!("pad({})", size),
                Factor::Padding {
                    size,
                    kind: PaddingKind::Bottom,
                } => format!("bottom_pad({})", size),
                Factor::Term {
                    inner: Term { inner, stride, modulo },
                    resize,
                } => {
                    let atom_str = match inner {
                        Atom::Symbol { symbol, size } => format!("{}:{}", symbol, size),
                        Atom::Composite(inner) => format!("({})", inner),
                    };
                    format!("term({} / {} % {} = {})", atom_str, stride, modulo, resize)
                }
            })
            .collect();
        write!(f, "FMapping[{}]", terms.join(" * "))
    }
}

/// Error during division of mapping expressions.
#[repr(C)]
#[derive(StableAbi, Debug)]
pub enum DivisionError {
    /// No divisor terms found.
    NoDivisorTerms,
    /// Divisor term cannot divide dividend.
    DivisorTermCannotDivide,
}

/// Selects which side of a [`DivisionTerm`] to operate on.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionSide {
    /// The dividend (left-hand side of the division).
    Dividend,
    /// The divisor (right-hand side of the division).
    Divisor,
}

/// Information about a single matched term in a division.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct DivisionTerm {
    /// Stride of the dividend term.
    pub dividend_stride: usize,
    /// Stride of the divisor term.
    pub divisor_stride: usize,
    /// Divisor term.
    pub term: Term,
    /// Divisor resize.
    pub resize: usize,
}

impl DivisionTerm {
    /// Returns the stride of this matched term on the selected side.
    pub fn stride(&self, side: DivisionSide) -> usize {
        match side {
            DivisionSide::Dividend => self.dividend_stride,
            DivisionSide::Divisor => self.divisor_stride,
        }
    }
}

/// Bounds for the padded block removed for a matched term.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockBounds {
    /// Minimum block size chosen by the current replay semantics.
    pub min: usize,
    /// Largest normalized padding boundary that still encloses the removed content.
    pub max: usize,
}

/// Per-term compact padding bounds.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct TermBounds {
    /// Matched term metadata for this bounds row.
    pub term: DivisionTerm,
    /// Bounds reconstructed on the dividend side.
    pub dividend: BlockBounds,
    /// Bounds reconstructed on the divisor side.
    pub divisor: BlockBounds,
}

/// Determines the padding kind used for matched-hole markers in division.
pub trait DivisionMode {
    /// The padding kind to use for matched-term holes.
    const PADDING_KIND: PaddingKind;
}

/// Marker for analysis-capable division results.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy)]
pub struct Strict;
impl DivisionMode for Strict {
    const PADDING_KIND: PaddingKind = PaddingKind::Bottom;
}

/// Marker for read-accessible division results.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy)]
pub struct Relaxed;
impl DivisionMode for Relaxed {
    const PADDING_KIND: PaddingKind = PaddingKind::Top;
}

/// Marker for span-division results without padding analysis.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy)]
pub struct Span;
impl DivisionMode for Span {
    const PADDING_KIND: PaddingKind = PaddingKind::Top;
}

/// Result of dividing two factor mappings.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct Division<M: DivisionMode> {
    /// Information about each matched divisor term.
    pub division_terms: RVec<DivisionTerm>,
    /// Original dividend before matching.
    pub dividend: FMapping,
    /// Dividend residue. Strict results preserve Bottom padding;
    /// relaxed results have all padding converted to Top.
    pub dividend_residue: FMapping,
    /// Original divisor before matching.
    pub divisor: FMapping,
    /// Divisor residue. Strict results preserve Bottom padding;
    /// relaxed results have all padding converted to Top.
    pub divisor_residue: FMapping,
    _mode: PhantomData<M>,
}

impl<M: DivisionMode> Division<M> {
    /// Creates a new division result.
    pub fn new(
        dividend: FMapping,
        dividend_residue: FMapping,
        mut division_terms: Vec<DivisionTerm>,
        divisor: FMapping,
        divisor_residue: FMapping,
    ) -> Self {
        division_terms.sort_by(|a, b| b.dividend_stride.cmp(&a.dividend_stride));

        Self {
            division_terms: division_terms.into(),
            dividend,
            dividend_residue,
            divisor,
            divisor_residue,
            _mode: PhantomData,
        }
    }

    /// Returns the original dividend.
    pub fn dividend(&self) -> &FMapping {
        &self.dividend
    }

    /// Returns the original divisor.
    pub fn divisor(&self) -> &FMapping {
        &self.divisor
    }

    /// Returns the residue for the selected side.
    pub fn residue(&self, side: DivisionSide) -> &FMapping {
        match side {
            DivisionSide::Dividend => &self.dividend_residue,
            DivisionSide::Divisor => &self.divisor_residue,
        }
    }

    /// Returns the original mapping for the selected side.
    pub fn mapping(&self, side: DivisionSide) -> &FMapping {
        match side {
            DivisionSide::Dividend => &self.dividend,
            DivisionSide::Divisor => &self.divisor,
        }
    }

    /// Returns matched division terms in dividend-order.
    pub fn division_terms(&self) -> &[DivisionTerm] {
        &self.division_terms
    }
}

/// A Term factor with its position (stride) in an FMapping.
///
/// Returned by [`FMapping::terms_with_stride`].
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct TermPosition {
    /// The term.
    pub term: Term,
    /// Size of this term (number of positions it contributes).
    pub resize: usize,
    /// Effective stride of this term in the FMapping
    /// (product of all inner factors' sizes).
    pub stride: usize,
}

/// Index mapping for tensor operations.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct Index(pub RResult<RSortedMap<Term, usize>, ()>);

/// Error returned when querying ident contributions from an [`Index`].
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexValueError {
    /// The index is invalid, typically because it points into padding.
    Invalid,
    /// The index still contains Composite terms and is not evaluable per ident.
    NonFlattened,
}

impl Default for Index {
    fn default() -> Self {
        Self(RResult::ROk(RSortedMap::new()))
    }
}

impl Display for Index {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            RResult::ROk(map) => {
                let terms = map.iter().map(|(k, v)| format!("{k} = {v}")).join(", ");
                write!(f, "Index[{}]", terms)
            }
            RResult::RErr(_) => {
                write!(f, "Invalid Index")
            }
        }
    }
}

/// Mapping expression that describes memory layout and computes size for a given shape.
#[primitive(mapping::M)]
// ANCHOR: trait_m
pub trait M: Debug + Clone {
    /// The computed size for the given shape.
    const SIZE: usize;

    /// Converts the mapping expression type into a value.
    fn to_value() -> Mapping;

    /// Converts a buffer index to a tensor index, returning `None` if out-of-bounds.
    fn map(i: usize) -> Option<Index>;
}
// ANCHOR_END: trait_m
