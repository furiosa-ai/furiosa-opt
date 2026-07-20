//! Mapping expressions.

#![feature(register_tool, adt_const_params)]
#![register_tool(furiosa_opt)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]

mod sorted_map;
pub use sorted_map::RSortedMap;

mod dsl;
pub use dsl::*;

// The `m!` / `axes!` expression macros expand into the `dsl` primitives above,
// so they live alongside the types they build. (`i!` stays in `furiosa-mapping`:
// it expands to FFI-backed `Index` mutation.)
pub use furiosa_mapping_macro::{axes, m};

use abi_stable::{
    StableAbi,
    std_types::{RArc, RBox, ROption, RResult, RVec},
};
use std::fmt::{self, Display, Formatter};

use furiosa_opt_macro::primitive;
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
    /// only ASCII alphanumeric characters, underscores.
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
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mapping {
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
    /// Broadcast mapping: `size` iterations of a stride-0 (don't-care) axis.
    /// `size == 1` is the identity element (see [`Mapping::identity`]).
    Broadcast {
        /// Number of broadcast iterations.
        size: usize,
    },
}

impl Mapping {
    /// The identity mapping (size 1), the unit of `pair`.
    pub const fn identity() -> Self {
        Self::Broadcast { size: 1 }
    }

    /// Pairs two mappings, dropping an identity operand rather than nesting it (`[1, X] == X`).
    /// The single source of truth shared by `MappingExt::pair` (above the FFI) and the impl
    /// crate's `mapping_pair` (the factorize path), so the two cannot diverge.
    pub fn pair(self, other: Self) -> Self {
        if self == Self::identity() {
            other
        } else if other == Self::identity() {
            self
        } else {
            Self::Pair {
                left: RBox::new(self),
                right: RBox::new(other),
            }
        }
    }

    /// The cell count this mapping reads. Tolerates a ragged `Stride` / `Modulo` (one whose
    /// divisor does not divide the inner size), which the matcher emits as a valid OUTPUT
    /// even though it is not a valid `from_mapping` INPUT.
    pub fn size(&self) -> usize {
        match self {
            Self::Symbol { size, .. } => *size,
            Self::Stride { inner, stride } => inner.size() / stride,
            Self::Modulo { modulo, .. } => *modulo,
            Self::Resize { resize, .. } => *resize,
            Self::Padding { padding, .. } => *padding,
            Self::Pair { left, right } => left.size() * right.size(),
            Self::Broadcast { size } => *size,
        }
    }

    /// Whether this is an acceptable leftover for `mode` — the per-mode coverage the engines run on a
    /// carved-down [`SequencerConfig`] remainder, and the matcher reuses to accept an over-capacity
    /// fold. A live `Symbol` is an unread/unwritten real cell, rejected by both modes. Read re-reads
    /// any pad or broadcast as a don't-care. Write must fill every cell, so it also rejects a surviving
    /// `Zero` pad (unwritten zeros) and a `Broadcast` of more than one (cells the write never reached).
    pub fn consumed(&self, mode: SequencerMode) -> bool {
        match self {
            Self::Symbol { .. } => false,
            Self::Broadcast { size } => mode.carving() == CarvingMode::Read || *size == 1,
            Self::Padding { inner, kind, .. } => {
                (mode.carving() == CarvingMode::Read || *kind != PaddingKind::Zero) && inner.consumed(mode)
            }
            Self::Stride { inner, .. } | Self::Modulo { inner, .. } | Self::Resize { inner, .. } => {
                inner.consumed(mode)
            }
            Self::Pair { left, right } => left.consumed(mode) && right.consumed(mode),
        }
    }

    /// Folds `ms` into one mapping with `pair`, identity-first.
    pub fn pairs(ms: impl IntoIterator<Item = Self>) -> Self {
        ms.into_iter().fold(Self::identity(), Self::pair)
    }

    /// Strides the mapping, shrinking it to `size / stride`. A raw constructor (`stride 1` is a no-op).
    /// Call `normalize` for canonical form.
    pub fn stride(self, stride: usize) -> Self {
        if stride == 1 {
            return self;
        }
        let size = self.size();
        assert!(
            size.is_multiple_of(stride),
            "stride {stride} does not divide size {size}"
        );
        Self::Stride {
            inner: RBox::new(self),
            stride,
        }
    }

    /// Keeps the first `modulo` cells. A raw constructor (`modulo == size` is a no-op). Call `normalize`
    /// for canonical form.
    pub fn modulo(self, modulo: usize) -> Self {
        let size = self.size();
        if modulo == size {
            return self;
        }
        assert!(
            size.is_multiple_of(modulo),
            "modulo {modulo} does not divide size {size}"
        );
        Self::Modulo {
            inner: RBox::new(self),
            modulo,
        }
    }

    /// Caps to the first `resize` cells (shrinks or stays). A raw constructor (`resize == size` is a
    /// no-op). Call `normalize` for canonical form.
    pub fn resize(self, resize: usize) -> Self {
        let size = self.size();
        if resize == size {
            return self;
        }
        assert!(resize <= size, "resize {resize} exceeds size {size}");
        Self::Resize {
            inner: RBox::new(self),
            resize,
        }
    }

    /// Pads up to total extent `padding` with `kind` (grows or stays; `padding == size` is a no-op). A
    /// raw constructor. Call `normalize` for canonical form.
    pub fn padding(self, padding: usize, kind: PaddingKind) -> Self {
        let size = self.size();
        if padding == size {
            return self;
        }
        assert!(padding >= size, "padding {padding} is below size {size}");
        Self::Padding {
            inner: RBox::new(self),
            padding,
            kind,
        }
    }

    /// Peels outermost `Padding` nodes until a live factor. Interior padding is preserved. `normalize`
    /// emits the outermost factor as the left of the topmost `Pair`, so trailing padding is exactly the
    /// `Padding` nodes on the outer (left) spine.
    pub fn remove_padding(self) -> Self {
        match self {
            Self::Padding { inner, .. } => RBox::into_inner(inner).remove_padding(),
            Self::Pair { left, right } => {
                let left = RBox::into_inner(left).remove_padding();
                if left == Self::identity() {
                    RBox::into_inner(right).remove_padding()
                } else {
                    left.pair(RBox::into_inner(right))
                }
            }
            other => other,
        }
    }

    /// Removes existing padding and pads to `target` size.
    pub fn replace_padding(self, target: usize) -> Self {
        let unpadded = self.remove_padding();
        let size = unpadded.size();
        assert!(size <= target, "unpadded size {size} exceeds target {target}");
        if size < target {
            unpadded.padding(target, PaddingKind::Top)
        } else {
            unpadded
        }
    }

    /// Whether the mapping carries no live cell (only padding or identity).
    pub fn is_padding(&self) -> bool {
        !self.has_live()
    }

    /// Whether any cell generates a value, a `Symbol` or a `Broadcast` of more than one (a size-1
    /// broadcast is the identity).
    fn has_live(&self) -> bool {
        match self {
            Self::Symbol { .. } => true,
            Self::Broadcast { size } => *size > 1,
            Self::Stride { inner, .. }
            | Self::Modulo { inner, .. }
            | Self::Resize { inner, .. }
            | Self::Padding { inner, .. } => inner.has_live(),
            Self::Pair { left, right } => left.has_live() || right.has_live(),
        }
    }

    /// The unique idents referenced, first-encounter order.
    pub fn idents(&self) -> RVec<Ident> {
        let mut out = Vec::new();
        self.collect_idents(&mut out);
        out.into()
    }

    fn collect_idents(&self, out: &mut Vec<Ident>) {
        match self {
            Self::Symbol { symbol, .. } => {
                if !out.contains(symbol) {
                    out.push(*symbol);
                }
            }
            Self::Broadcast { .. } => {}
            Self::Stride { inner, .. }
            | Self::Modulo { inner, .. }
            | Self::Resize { inner, .. }
            | Self::Padding { inner, .. } => inner.collect_idents(out),
            Self::Pair { left, right } => {
                left.collect_idents(out);
                right.collect_idents(out);
            }
        }
    }
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
            } => write!(f, "{inner} #{{!}} {padding}"),
            Self::Padding {
                inner,
                padding,
                kind: PaddingKind::Zero,
            } => write!(f, "{inner} #{{0}} {padding}"),
            Self::Pair { left, right } => {
                // Collect all nested pairs and print them as flattened.
                let mut elements = vec![];
                flatten_pair(&mut elements, left);
                flatten_pair(&mut elements, right);
                write!(f, "({})", elements.iter().join(", "))
            }
            // A broadcast prints as its size; size 1 is the identity element.
            Self::Broadcast { size } => write!(f, "{size}"),
        }
    }
}

/// Serde-compatible mirror of [`Mapping`] using `Box` instead of `RBox`, so that
/// the standard derive macros work. Used only for serialization/deserialization.
#[derive(serde::Serialize, serde::Deserialize, serde_lite::Deserialize)]
enum MappingSerde {
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
    Broadcast {
        size: usize,
    },
}

impl From<Mapping> for MappingSerde {
    fn from(m: Mapping) -> Self {
        match m {
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
            Mapping::Broadcast { size } => Self::Broadcast { size },
        }
    }
}

impl From<MappingSerde> for Mapping {
    fn from(m: MappingSerde) -> Self {
        match m {
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
            MappingSerde::Broadcast { size } => Self::Broadcast { size },
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

/// Kind of a padding factor. Variant order encodes strictness `Bottom < Zero < Top`,
/// relied on by the derived `Ord`.
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
    core::marker::ConstParamTy,
)]
pub enum PaddingKind {
    /// Inaccessible padding; reads are undefined behavior.
    Bottom,
    /// Accessible padding masked to a known constant zero.
    ///
    /// In the future this will generalize to `Value(u64)` so other identity
    /// constants (e.g. `INT_MIN` for `max` reductions) can be expressed.
    Zero,
    /// Accessible padding holding arbitrary (LLVM-style undef) values.
    Top,
}

/// Atomic operand of an [`Term`]: a named axis, or a composite sub-mapping carried as a
/// `Mapping` (FMapping-free, unlike [`Atom`]).
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// A named axis with its declared size.
    Symbol {
        /// Symbol of the axis.
        symbol: Ident,
        /// Size of the axis.
        size: usize,
    },
    /// A composite sub-mapping.
    Composite(RBox<Mapping>),
}

/// `inner / stride % modulo`, the key of an [`Index`].
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term {
    /// Inner atomic operand.
    pub inner: Atom,
    /// Stride.
    pub stride: usize,
    /// Modulo.
    pub modulo: usize,
}

impl Atom {
    /// The atom's live size.
    pub fn size(&self) -> usize {
        match self {
            Self::Symbol { size, .. } => *size,
            Self::Composite(inner) => inner.size(),
        }
    }
}

impl Term {
    /// This term as a `Mapping` node (`atom / stride % modulo`), the inverse of finalize's term build.
    pub fn to_mapping(&self) -> Mapping {
        let (mut m, size) = match &self.inner {
            Atom::Symbol { symbol, size } => (
                Mapping::Symbol {
                    symbol: *symbol,
                    size: *size,
                },
                *size,
            ),
            Atom::Composite(inner) => ((**inner).clone(), inner.size()),
        };
        if self.stride != 1 {
            m = Mapping::Stride {
                inner: RBox::new(m),
                stride: self.stride,
            };
        }
        if self.modulo != size / self.stride {
            m = Mapping::Modulo {
                inner: RBox::new(m),
                modulo: self.modulo,
            };
        }
        m
    }
}

impl Mapping {
    /// Builds a mapping from axis [`Term`]s, the inverse of finalize's terms.
    pub fn from_terms(terms: impl IntoIterator<Item = Term>) -> Self {
        Self::pairs(terms.into_iter().map(|t| t.to_mapping()))
    }
}

/// A [`Term`] guaranteed to be a resolved symbol axis (never a composite): `symbol / stride % modulo`.
/// The element type of `MappingExt::axes` and the `axes` a `MappingIter` is projected onto, produced
/// by the impl crate's `resolve`. Encoding "the composites are decoded away" as a type, rather than
/// re-checking it at every axis consumer, is the point: there is no composite case to guard.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AxisTerm {
    /// The axis symbol.
    pub symbol: Ident,
    /// The symbol's declared size.
    pub size: usize,
    /// Stride.
    pub stride: usize,
    /// Modulo.
    pub modulo: usize,
}

impl AxisTerm {
    /// As a full [`Term`] (an `Atom::Symbol`), for the places that rebuild a `Mapping` from an axis
    /// (via [`Mapping::from_terms`]).
    pub fn to_term(self) -> Term {
        Term {
            inner: Atom::Symbol {
                symbol: self.symbol,
                size: self.size,
            },
            stride: self.stride,
            modulo: self.modulo,
        }
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Elide a trivial stride/modulo, matching `Term`'s Display.
        let (head, full): (String, usize) = match &self.inner {
            Atom::Symbol { symbol, size } => (symbol.to_string(), *size),
            Atom::Composite(inner) => (format!("({inner})"), inner.size()),
        };
        match (self.stride == 1, self.modulo == full) {
            (true, true) => write!(f, "{head}"),
            (true, false) => write!(f, "{head} % {}", self.modulo),
            (false, true) => write!(f, "{head} // {}", self.stride),
            (false, false) => write!(f, "({head} // {}) % {}", self.stride, self.modulo),
        }
    }
}

/// A buffer cell's read: the per-axis contributions on a live cell, or the [`PaddingKind`] it lands
/// on otherwise (`Bottom` for an out-of-bounds / over-modulo read with no covering pad). A non-live
/// cell's kind is decided by the MAJOR (outermost) factor that hits a pad first.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct Index(pub RResult<RSortedMap<Term, usize>, PaddingKind>);

impl Default for Index {
    fn default() -> Self {
        Self(RResult::ROk(RSortedMap::new()))
    }
}

impl Index {
    /// Creates a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores a single term (composites kept WHOLE -- `finalize` decodes them). A term whose modulo
    /// is overshot is an out-of-bounds read (`Bottom`). Pure `Index` arithmetic, so it lives here
    /// (above the FFI) and is shared by the DSL `M::map`, the public `IndexExt`, and the decode.
    pub fn add_term(&mut self, term: Term, value: usize) {
        let RResult::ROk(map) = &mut self.0 else { return };
        let modulo = term.modulo;
        let entry = map.get_or_insert(term, 0);
        *entry += value;
        if *entry >= modulo {
            self.0 = RResult::RErr(PaddingKind::Bottom);
        }
    }

    /// Merges another index into this one. An existing error is kept, major priority. The outer
    /// factor's pad kind, set first, wins over a later one, like `add_term` no-opping once invalid.
    pub fn add(&mut self, other: Self) {
        if matches!(self.0, RResult::RErr(_)) {
            return;
        }
        match other.0 {
            RResult::ROk(terms) => {
                for (term, value) in terms {
                    self.add_term(term, value);
                }
            }
            RResult::RErr(kind) => self.0 = RResult::RErr(kind),
        }
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

/// Read or Write semantics for sequencing a memory layout against a stream
/// access pattern.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencerMode {
    /// Stream reads from memory (a memory cell may be read by many positions).
    Read,
    /// Stream writes to memory (each memory cell is written exactly once).
    Write,
    /// A read carve ([`MappingExt::carve`]) that also tolerates a `Bottom` pad in the stream (a
    /// `view_mut` hole, read as a `Top` don't-care). TRANSIENT (PROG-480): a stopgap until the sequencer
    /// can carve the broadcast from the memory leftover (see `read_carved_down_memory_keeps_broadcast_term`),
    /// at which point this mode can be retired.
    Carve,
}

impl SequencerMode {
    /// The two-valued carving direction the matcher branches on; `Carve` shares `Read`'s.
    pub const fn carving(self) -> CarvingMode {
        match self {
            SequencerMode::Read | SequencerMode::Carve => CarvingMode::Read,
            SequencerMode::Write => CarvingMode::Write,
        }
    }

    /// The pad kind an unbacked memory gap reads or writes as: delegates to [`CarvingMode::gap_kind`].
    pub const fn gap_kind(self) -> PaddingKind {
        self.carving().gap_kind()
    }
}

/// The two-valued carving direction the matcher branches on, projected from [`SequencerMode::carving`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarvingMode {
    /// Stream reads from memory: `memory ⊑ stream`, unmatched stream segments broadcast, gap = `Top`.
    Read,
    /// Stream writes to memory: `stream ⊑ memory`, every cell written once, gap = `Bottom`.
    Write,
}

impl CarvingMode {
    /// The default fill for an unbacked memory gap: `Top` (don't-care) for read, `Bottom` for write.
    pub const fn gap_kind(self) -> PaddingKind {
        match self {
            Self::Read => PaddingKind::Top,
            Self::Write => PaddingKind::Bottom,
        }
    }
}

/// Error returned by sequencing.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub enum SequencerError {
    /// A stream segment had no compatible memory match.
    StreamUnmatchedSegment,
    /// An input carried a `Bottom` pad: the post-match consumed marker, never
    /// valid on a fresh input.
    InputBottomPadding,
    /// A carved-down memory was left unconsumed (a live cell the streams never
    /// read / wrote). Carries every memory in input order so the caller can
    /// name the offending one.
    Unconsumed(RVec<Mapping>),
}

/// One entry of a [`SequencerConfig`]: the matched sub-layout, which input memory it
/// reads/writes (`memory_index` into the memory vector passed to the matcher),
/// and the buffer stride within that memory. `memory_stride == 0` marks a
/// broadcast (no memory access). Single-memory callers use `memory_index == 0`.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct SequencerEntry {
    /// The matched sub-layout (the same axis on the memory and stream sides).
    pub mapping: Mapping,
    /// Which input memory this entry reads/writes, indexing the memory vector.
    pub memory_index: usize,
    /// Buffer stride/base on the memory side; `0` makes it a broadcast.
    pub memory_stride: usize,
}

/// Successful result of sequencing: the matched entries that tile the stream,
/// keyed by their stream-side stride (the buffer position they tile,
/// innermost-first).
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct SequencerConfig(pub RSortedMap<usize, SequencerEntry>);

impl SequencerConfig {
    /// The number of stream positions the config tiles: the product of the entry sizes.
    pub fn stream_size(&self) -> usize {
        self.0.iter().map(|(_, e)| e.mapping.size()).product()
    }

    /// Walks the entries (innermost-first) yielding, for each stream position `0..stream_size`, the
    /// memory offset `Σ digitᵢ · memory_strideᵢ`. The stream position is the mixed-radix counter over
    /// the entry sizes, so the result is indexed by stream (output) buffer position. Pure arithmetic
    /// over the sequencer structure: no per-cell index decode / `finalize`.
    pub fn iter(&self) -> SequencerIter {
        self.iter_range(0, self.stream_size())
    }

    /// [`Self::iter`] restricted to the stream positions `[start, end)`. Seeks to `start` once
    /// (O(entries)) then walks the same O(1)-amortized odometer, so a driver can split the walk into
    /// independent ranges without a per-position address computation. Used at the rayon-parallel call
    /// site to give each worker its own sub-range.
    pub fn iter_range(&self, start: usize, end: usize) -> SequencerIter {
        let wheels: Vec<Wheel> = self
            .0
            .iter()
            .map(|(_, e)| Wheel {
                size: e.mapping.size(),
                stride: e.memory_stride,
            })
            .collect();
        // Seek the front odometer to `start`: its digits and the offset `Σ digitᵢ · strideᵢ` they sum to.
        let mut digit = vec![0usize; wheels.len()];
        let mut offset = 0usize;
        let mut p = start;
        for (i, e) in wheels.iter().enumerate() {
            digit[i] = p % e.size;
            offset += digit[i] * e.stride;
            p /= e.size;
        }
        SequencerIter {
            wheels,
            digit,
            offset,
            pos: start,
            end,
        }
    }
}

/// One wheel of a [`SequencerIter`]'s odometer, innermost-first: the digit's wheel size and the offset
/// step each tick moves. A named pair so the two `usize`s cannot be swapped at a use site. (The
/// hot-loop projection of a [`SequencerEntry`]: just its `size` + `stride`, no `Mapping` deref per tick.)
#[derive(Debug, Clone, Copy)]
struct Wheel {
    /// Number of values the digit takes (the entry's stream extent).
    size: usize,
    /// Offset moved per digit step (the entry's memory stride).
    stride: usize,
}

/// A walk over a [`SequencerConfig`]'s memory offsets for a contiguous range of stream positions
/// `[pos, end)` (see [`SequencerConfig::iter`]). The front advances by the O(1)-amortized
/// mixed-radix odometer; the back is addressed directly (O(entries)). `ExactSizeIterator` +
/// `DoubleEndedIterator`, and seekable at construction, so a rayon producer at the call site can split
/// it into ranges — keeping this crate rayon-free.
#[derive(Debug, Clone)]
pub struct SequencerIter {
    /// The odometer wheels, innermost-first.
    wheels: Vec<Wheel>,
    /// The front odometer reading (the digits at `pos`).
    digit: Vec<usize>,
    /// The front offset `Σ digitᵢ · memory_strideᵢ` (the offset at `pos`).
    offset: usize,
    /// Current front position; the next [`Iterator::next`] yields its offset.
    pos: usize,
    /// Exclusive back position; the next [`DoubleEndedIterator::next_back`] yields `end - 1`.
    end: usize,
}

impl Iterator for SequencerIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.pos >= self.end {
            return None;
        }
        let current = self.offset;
        // Tick the front odometer: the first digit with room advances (adding its stride); each fuller
        // digit before it wraps to 0 (subtracting what it had contributed).
        for (i, e) in self.wheels.iter().enumerate() {
            if self.digit[i] + 1 < e.size {
                self.digit[i] += 1;
                self.offset += e.stride;
                break;
            }
            self.offset -= self.digit[i] * e.stride;
            self.digit[i] = 0;
        }
        self.pos += 1;
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.pos;
        (len, Some(len))
    }
}

impl ExactSizeIterator for SequencerIter {}

impl DoubleEndedIterator for SequencerIter {
    fn next_back(&mut self) -> Option<usize> {
        if self.pos >= self.end {
            return None;
        }
        self.end -= 1;
        // The back reads at a decreasing position directly (decompose + re-sum, O(entries)), without
        // touching the front odometer.
        let mut p = self.end;
        let mut offset = 0;
        for e in &self.wheels {
            offset += (p % e.size) * e.stride;
            p /= e.size;
        }
        Some(offset)
    }
}

/// One segment of a [`MappingIter`]: the range its digit takes, and how stepping it moves the offset.
/// Hidden: a construction detail the impl crate fills in via [`MappingIter::new`]; consumers only walk
/// the finished iterator.
#[doc(hidden)]
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetFactor {
    /// How many values the digit takes, padding included (the wire extent).
    pub extent: usize,
    /// How many of those are live; a digit in `[live, extent)` is padding.
    pub live: usize,
    /// How stepping this digit moves the offset (see [`FactorStep`]).
    pub step: FactorStep,
}

/// How stepping a segment's digit moves the walk: exactly one of these per segment. Hidden: an
/// [`OffsetFactor`] construction detail, not part of the walking API.
#[doc(hidden)]
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorStep {
    /// The linear (common) case: add this fixed amount to the running offset.
    Linear(usize),
    /// A composite (non-linear) segment: advance the `composite_table` index by this weight; the
    /// offset itself comes from the table, not from a fixed step.
    Composite(usize),
    /// A broadcast segment: nothing (the cell repeats, e.g. a broadcast axis re-reads one source).
    Broadcast,
}

/// The immutable walk plan, shared by a walk and its split sub-ranges via `RArc` so a split is a
/// refcount bump, not a deep copy of `composite_table`.
#[repr(C)]
#[derive(StableAbi, Debug)]
struct MappingPlan {
    /// One factor per segment, innermost-first (its digit varies fastest).
    factors: RVec<OffsetFactor>,
    /// The offsets the composite segments' digits index (their `finalize`d contribution plus the
    /// base). One joint table because sibling composite slices `finalize` together (it cannot split
    /// per factor). Always at least one entry (just the base offset when there is no composite
    /// segment); `None` marks a combination that lands in padding.
    composite_table: RVec<ROption<usize>>,
}

/// A lazy iterator over where every cell of a mapping lands in a dense buffer laid out by some `axes`
/// (built by `MappingExt::iter`). It yields one `Option<usize>` per physical cell in canonical order:
/// `Some(off)` for a live cell at dense-buffer offset `off`, `None` for a padding cell. Computing the
/// offsets as a compact formula rather than a materialized per-cell array keeps it `O(1)` in memory
/// instead of `O(size)`.
///
/// A mapping is a product of segments, so a cell is one digit per [`OffsetFactor`] segment, a
/// mixed-radix number read like an odometer (the innermost segment varies fastest). The offset is
/// almost always *linear* in those digits: stepping one segment's digit moves the offset by a fixed
/// amount ([`FactorStep`]), so it keeps that per-segment step rather than every offset and recovers
/// each cell's offset by walking the digits and accumulating. For example, a buffer shaped
/// `[A = 4, B = 3]` has steps `B: +1` and `A: +3`, so cell `(a, b)` lands at `3a + b`.
///
/// The exception is a *composite* segment, whose offset is not linear in its digit. The combined
/// contribution of all composite segments is precomputed into the composite table, which their digits
/// index. So in full:
///
/// > cell offset = (sum of the linear segments' steps) + composite_table\[composite digits\]
///
/// The innermost factor is almost always a plain affine run (a `Linear` or `Broadcast` step with no
/// padding tail), so the iterator serves a whole *row* (the innermost factor's cells) from a cached
/// [`RowCursor`] (a `(start, stride)` and a column) by arithmetic, ticking the heavier outer odometer
/// only once per row. This cached-row path keeps the per-cell `Iterator` API while amortizing the
/// odometer; mappings whose innermost factor is composite (or a wire traversal with an inner padding
/// tail) carry no `RowCursor` and walk the plain per-cell odometer.
///
/// The same type also serves a bounded sub-range `[start, end)` for parallel splitting (see
/// [`Self::range`] and the rayon producer in the std crate). The plan is shared by `RArc`, so a split
/// is a refcount bump rather than a deep copy of `composite_table`; the sub-range re-seats the row
/// cache at its start (so a worker walks its slice on the same fast path as the serial whole) and is
/// `DoubleEndedIterator` + `ExactSizeIterator` (the back end reads directly, without disturbing the
/// forward reading). This keeps the mapping-types crate rayon-free: the std crate wraps the sub-ranges
/// in a rayon producer that splits by `range`.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct MappingIter {
    /// The walk plan (factors + composite table), shared with any split sub-ranges via `RArc`.
    plan: RArc<MappingPlan>,
    /// Visit every physical cell, padding included as `None` (`true`, the wire traversal the buffer
    /// seam needs); or only live cells, skipping the padding cells entirely (`false`, faster on padded
    /// mappings). Set at construction from `MappingExt::iter`'s `padding` argument.
    padding: bool,
    /// Odometer reading: the current digit of each factor.
    digit: RVec<usize>,
    /// Running sum of the linear factors' steps at the current digits. On the cached-row path this
    /// excludes the innermost factor (the row's column adds that), so it is the row's base linear
    /// contribution.
    linear_offset: usize,
    /// Current index into the plan's `composite_table` (the composite factors' combination).
    composite_index: usize,
    /// How many digits sit in their padding tail; the current cell is padding while this is non-zero.
    pad_depth: usize,
    /// Current cell position; the next [`Iterator::next`] yields its offset.
    pos: usize,
    /// Exclusive end; the walk stops at `pos == end`. A full walk has `end == size`; a sub-range bounds
    /// it to `[start, end)`. [`DoubleEndedIterator::next_back`] reads `end - 1` and retreats `end`.
    end: usize,
    /// The cached affine row when the innermost factor is served as one (see the struct docs), or
    /// `RNone` for the plain per-cell odometer. Set at construction; a sub-range ([`Self::range`]) keeps
    /// it and re-seats the row cache at its start. Holding the row state behind the option means the
    /// plain path cannot carry stale row fields.
    row: ROption<RowCursor>,
}

/// The cached affine row served on the fast path: the innermost factor's cells laid out as
/// `row_start + col * stride`. Present only when [`MappingIter`] takes the fast path; its fields are
/// meaningless otherwise, which is why they live behind [`MappingIter::row`]'s [`ROption`] rather than
/// beside a `fast` flag.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy)]
struct RowCursor {
    /// Cells per row: the innermost factor's traversed extent (`live`, or `extent` under `padding`).
    /// Always `> 0` (a zero-length innermost factor takes the plain path), so `range`'s `% len` is safe.
    len: usize,
    /// Per-column offset delta within a row: the innermost factor's step (`0` for a broadcast).
    stride: isize,
    /// Current column within the row, `0..len`.
    pos: usize,
    /// The current row's column-0 offset, or `None` if the whole row lands in padding.
    start: ROption<usize>,
}

impl MappingIter {
    /// Builds a fresh iterator from the plan the impl crate computes: `factors` (the odometer wheels)
    /// and `composite_table`; the `axes`/`base` it was projected onto are already baked in. `padding`
    /// picks the traversal: `true` visits every physical cell (padding as `None`, the wire order the
    /// buffer seam needs), `false` skips the padding cells entirely (faster on padded mappings, for the
    /// relayouts). Hidden because it is constructed by the impl crate over the FFI boundary, not by
    /// consumers (who go through `MappingExt::iter`).
    #[doc(hidden)]
    pub fn new(factors: RVec<OffsetFactor>, composite_table: RVec<ROption<usize>>, padding: bool) -> Self {
        let digit = RVec::from(vec![0usize; factors.len()]);
        // The cell count: the product of the factor bounds (`live`, or `extent` under `padding`).
        let end = factors
            .iter()
            .map(|f| if padding { f.extent } else { f.live })
            .product();
        // The innermost factor serves as a cached affine row unless it is composite (offset not linear
        // in its digit), empty, or a wire traversal with a padding tail (the row would not be uniform).
        // `start` is the row's offset at position 0: on the fresh odometer (all offsets 0) `peek()` is
        // just the first composite-table entry, so we read it here without needing the built iterator.
        let row = match factors.first() {
            Some(f) if !(padding && f.live != f.extent) => {
                let len = if padding { f.extent } else { f.live };
                match f.step {
                    FactorStep::Linear(delta) if len > 0 => Some((len, delta as isize)),
                    FactorStep::Broadcast if len > 0 => Some((len, 0)),
                    _ => None,
                }
            }
            _ => None,
        }
        .map(|(len, stride)| RowCursor {
            len,
            stride,
            pos: 0,
            start: ROption::from(Option::from(composite_table[0])),
        });
        Self {
            plan: RArc::new(MappingPlan {
                factors,
                composite_table,
            }),
            padding,
            digit,
            linear_offset: 0,
            composite_index: 0,
            pad_depth: 0,
            pos: 0,
            end,
            row: row.into(),
        }
    }
}

impl MappingIter {
    /// The offset of the cell at the current odometer reading: the linear sum plus the composite-table
    /// entry, or `None` when a digit sits in its padding tail (`pad_depth > 0`) or the composite
    /// combination is itself padding.
    fn peek(&self) -> Option<usize> {
        if self.pad_depth > 0 {
            None
        } else {
            Option::from(self.plan.composite_table[self.composite_index]).map(|c: usize| self.linear_offset + c)
        }
    }

    /// Ticks the odometer once, carrying from digit `from` outward, innermost (fastest) digit first: the
    /// first digit with room advances (`+1`) and the carry stops; each fuller digit before it wraps back
    /// to 0 (undoing the steps it ran up) and the carry moves on. Each ticked digit's step moves the
    /// linear offset or composite index (or neither, for a broadcast); `padding` picks each digit's
    /// bound (`extent` vs `live`) and tracks entering/leaving the padding tail. Returns whether any digit
    /// advanced (`false` once the walk rolls over). The cached-row path passes `from = 1` to advance a
    /// whole innermost row at once (the innermost digit is served from the row cache).
    fn advance_from(&mut self, from: usize) -> bool {
        let factors = &self.plan.factors;
        let mut advanced = false;
        for (j, f) in factors.iter().enumerate().skip(from) {
            let bound = if self.padding { f.extent } else { f.live };
            let count: isize = if self.digit[j] + 1 < bound {
                if self.padding && self.digit[j] + 1 == f.live {
                    self.pad_depth += 1; // entering the padding tail
                }
                self.digit[j] += 1;
                advanced = true;
                1
            } else {
                if self.padding && self.digit[j] >= f.live {
                    self.pad_depth -= 1; // leaving the padding tail
                }
                let undo = -(self.digit[j] as isize);
                self.digit[j] = 0;
                undo
            };
            match f.step {
                FactorStep::Linear(delta) => {
                    self.linear_offset = self.linear_offset.wrapping_add_signed(count * delta as isize)
                }
                FactorStep::Composite(place) => {
                    self.composite_index = self.composite_index.wrapping_add_signed(count * place as isize)
                }
                FactorStep::Broadcast => {}
            }
            if advanced {
                break;
            }
        }
        advanced
    }

    /// Seeks the reading to cell position `p` (fills `digit` and the running sums), so a sub-range can
    /// start at an arbitrary offset instead of stepping from 0.
    fn seek(&mut self, mut p: usize) {
        self.linear_offset = 0;
        self.composite_index = 0;
        self.pad_depth = 0;
        for i in 0..self.plan.factors.len() {
            let f = self.plan.factors[i];
            let bound = if self.padding { f.extent } else { f.live };
            let d = p % bound;
            p /= bound;
            self.digit[i] = d;
            match f.step {
                FactorStep::Linear(delta) => self.linear_offset += d * delta,
                FactorStep::Composite(place) => self.composite_index += d * place,
                FactorStep::Broadcast => {}
            }
            if self.padding && d >= f.live {
                self.pad_depth += 1;
            }
        }
    }

    /// Serves the next cell on the fast path's cached row (`row` must be `RSome`): one column of the
    /// row, advancing the outer odometer (and recomputing the row's base) only when a row is exhausted.
    /// The row's cells are `start + col * stride`; a `None` `start` is a fully-padding row.
    fn next_fast(&mut self) -> Option<Option<usize>> {
        if self.pos >= self.end {
            return None;
        }
        // Read the row's scalar state by value; the carry branch below re-borrows `row` because
        // `advance_from`/`peek` need `self` and so cannot coexist with a held cursor borrow.
        let &RowCursor {
            len,
            stride,
            pos,
            start,
        } = self.row.as_ref().expect("next_fast called on the plain (RNone) path");
        let cell = Option::from(start).map(|s: usize| s.wrapping_add_signed(pos as isize * stride));
        self.pos += 1;
        let row_pos = pos + 1;
        if row_pos < len {
            self.row.as_mut().unwrap().pos = row_pos;
        } else {
            // Row done: carry the outer digits (the innermost stays 0) and recompute the row base. A
            // single-factor plan has no outer digit, so the walk ends with the row.
            self.row.as_mut().unwrap().pos = 0;
            if self.plan.factors.len() == 1 || !self.advance_from(1) {
                self.pos = self.end;
            } else {
                let start = ROption::from(self.peek());
                self.row.as_mut().unwrap().start = start;
            }
        }
        Some(cell)
    }
}

impl Iterator for MappingIter {
    type Item = Option<usize>;

    fn next(&mut self) -> Option<Option<usize>> {
        if self.row.is_some() {
            return self.next_fast();
        }
        if self.pos >= self.end {
            return None;
        }
        let cell = self.peek();
        self.advance_from(0);
        self.pos += 1;
        Some(cell)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for MappingIter {}

impl DoubleEndedIterator for MappingIter {
    fn next_back(&mut self) -> Option<Option<usize>> {
        if self.pos >= self.end {
            return None;
        }
        self.end -= 1;
        // The back end reads at a decreasing position without disturbing the forward reading or its row
        // cache, so `next` and `next_back` can converge from both ends: a stateless decompose of `end`
        // into digits, re-summed (O(factors)).
        let mut p = self.end;
        let (mut linear_offset, mut composite_index, mut pad_depth) = (0usize, 0usize, 0usize);
        for f in &self.plan.factors {
            let bound = if self.padding { f.extent } else { f.live };
            let d = p % bound;
            p /= bound;
            match f.step {
                FactorStep::Linear(delta) => linear_offset += d * delta,
                FactorStep::Composite(place) => composite_index += d * place,
                FactorStep::Broadcast => {}
            }
            if self.padding && d >= f.live {
                pad_depth += 1;
            }
        }
        Some(if pad_depth > 0 {
            None
        } else {
            Option::from(self.plan.composite_table[composite_index]).map(|c: usize| linear_offset + c)
        })
    }
}

impl MappingIter {
    /// The number of cells remaining in the walk: `end - pos`.
    pub fn len(&self) -> usize {
        self.end - self.pos
    }

    /// Whether [`Self::len`] is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A bounded sub-walk over cell positions `[start, end)` — the unit a parallel driver splits into.
    /// Shares this walk's plan (an `RArc` bump, not a deep copy of `composite_table`) and seeks to
    /// `start`, yielding the same `Option<usize>` sequence as the full walk restricted to that range.
    /// Keeps the row cache: the sub-range re-seats it at `start`, so a parallel worker walks its slice
    /// on the same fast path as the serial whole.
    pub fn range(&self, start: usize, end: usize) -> MappingIter {
        let mut sub = self.clone();
        sub.pos = start;
        sub.end = end;
        if let ROption::RSome(cursor) = &sub.row {
            // Re-seat the row cache: seek the outer odometer to the row base (`start` rounded down to a
            // whole row), and the column within the row is the remainder.
            let col = start % cursor.len;
            sub.seek(start - col);
            let new_start = ROption::from(sub.peek());
            let cursor = sub.row.as_mut().unwrap();
            cursor.start = new_start;
            cursor.pos = col;
        } else {
            sub.seek(start);
        }
        sub
    }
}

#[cfg(test)]
mod mapping_iter_tests {
    use super::*;

    /// A named walk plan: factors (innermost-first), composite table, and the `padding` traversal flag.
    type Scenario = (&'static str, Vec<OffsetFactor>, Vec<Option<usize>>, bool);

    fn factor(extent: usize, live: usize, step: FactorStep) -> OffsetFactor {
        OffsetFactor { extent, live, step }
    }

    /// Independent oracle: each cell's offset from the digits of its position, computed without the
    /// iterator's incremental stepping. Returns `None` for a padding cell, mirroring `MappingIter`'s
    /// `Option<usize>` item but derived from scratch.
    fn oracle(factors: &[OffsetFactor], table: &[Option<usize>], padding: bool, p: usize) -> Option<usize> {
        let mut rest = p;
        let mut linear = 0usize;
        let mut composite = 0usize;
        let mut is_pad = false;
        for f in factors {
            let bound = if padding { f.extent } else { f.live };
            let d = rest % bound;
            rest /= bound;
            if d >= f.live {
                is_pad = true;
            }
            match f.step {
                FactorStep::Linear(delta) => linear += d * delta,
                FactorStep::Composite(place) => composite += d * place,
                FactorStep::Broadcast => {}
            }
        }
        if is_pad {
            None
        } else {
            table[composite].map(|c| linear + c)
        }
    }

    fn build(factors: Vec<OffsetFactor>, table: Vec<Option<usize>>, padding: bool) -> MappingIter {
        let rtable: RVec<ROption<usize>> = table.iter().copied().map(ROption::from).collect();
        MappingIter::new(RVec::from(factors), rtable, padding)
    }

    fn size(factors: &[OffsetFactor], padding: bool) -> usize {
        factors
            .iter()
            .map(|f| if padding { f.extent } else { f.live })
            .product()
    }

    /// Each scenario names the path it forces: `Linear`/`Broadcast` innermost takes the cached-row
    /// (fast) path, `Composite` innermost or a padded innermost takes the plain odometer.
    fn scenarios() -> Vec<Scenario> {
        vec![
            // Fast: Linear innermost, two-factor odometer, no padding.
            (
                "fast_linear",
                vec![factor(4, 4, FactorStep::Linear(1)), factor(3, 3, FactorStep::Linear(4))],
                vec![Some(0)],
                false,
            ),
            // Fast: Broadcast innermost (stride 0), so every cell in a row re-reads the row base.
            (
                "fast_broadcast",
                vec![
                    factor(3, 3, FactorStep::Broadcast),
                    factor(2, 2, FactorStep::Linear(10)),
                ],
                vec![Some(0)],
                false,
            ),
            // Fast under padding: innermost has no padding tail (live == extent), an outer factor does,
            // so some whole rows land in padding (`peek` -> None -> a None-`start` row).
            (
                "fast_padded_outer",
                vec![factor(3, 3, FactorStep::Linear(1)), factor(4, 2, FactorStep::Linear(8))],
                vec![Some(0)],
                true,
            ),
            // Plain: Composite innermost -> offset is not linear in its digit, so no row cache.
            (
                "plain_composite",
                vec![
                    factor(3, 3, FactorStep::Composite(1)),
                    factor(2, 2, FactorStep::Linear(100)),
                ],
                vec![Some(0), Some(5), None],
                false,
            ),
            // Plain: innermost itself has a padding tail under `padding`, excluded from the fast path.
            (
                "plain_padded_inner",
                vec![factor(4, 3, FactorStep::Linear(1)), factor(2, 2, FactorStep::Linear(8))],
                vec![Some(0)],
                true,
            ),
        ]
    }

    #[test]
    fn walk_matches_oracle_on_both_paths() {
        for (name, factors, table, padding) in scenarios() {
            // The scenario set must keep covering both paths, or this is a one-path test in disguise.
            let on_fast_path = build(factors.clone(), table.clone(), padding).row.is_some();
            assert_eq!(
                on_fast_path,
                name.starts_with("fast"),
                "scenario {name} took the wrong path"
            );
            let got: Vec<_> = build(factors.clone(), table.clone(), padding).collect();
            let want: Vec<_> = (0..size(&factors, padding))
                .map(|p| oracle(&factors, &table, padding, p))
                .collect();
            assert_eq!(got, want, "forward walk mismatch in scenario {name}");
        }
    }

    #[test]
    fn range_subwalk_matches_full_slice() {
        for (name, factors, table, padding) in scenarios() {
            let n = size(&factors, padding);
            let full: Vec<_> = build(factors.clone(), table.clone(), padding).collect();
            let it = build(factors.clone(), table.clone(), padding);
            // Cover row boundaries, mid-row starts, empty ranges, and the whole walk.
            for start in 0..=n {
                for end in start..=n {
                    let sub: Vec<_> = it.range(start, end).collect();
                    assert_eq!(
                        sub,
                        full[start..end],
                        "range({start}, {end}) mismatch in scenario {name}"
                    );
                }
            }
        }
    }

    #[test]
    fn front_and_back_converge() {
        for (name, factors, table, padding) in scenarios() {
            let full: Vec<_> = build(factors.clone(), table.clone(), padding).collect();
            // Interleave next()/next_back() from both ends; the back path recomputes cells without the
            // row cache, so this pins the two paths against each other.
            let mut it = build(factors.clone(), table.clone(), padding);
            let (mut front, mut back) = (Vec::new(), Vec::new());
            let mut take_front = true;
            loop {
                let item = if take_front { it.next() } else { it.next_back() };
                match item {
                    Some(cell) if take_front => front.push(cell),
                    Some(cell) => back.push(cell),
                    None => break,
                }
                take_front = !take_front;
            }
            back.reverse();
            front.extend(back);
            assert_eq!(front, full, "front/back convergence mismatch in scenario {name}");
        }
    }
}
