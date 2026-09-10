//! Macros for virtual ISA.

use lalrpop_util::lalrpop_mod;
use quote::quote;

mod diagnostic;
mod lexer;
lalrpop_mod!(parser, "/parser/parser.rs");

use lexer::{Lexer, LexerMode};
use parser::{IndexParser, MappingParser};

type ParseError = lalrpop_util::ParseError<lexer::Location, lexer::Token, lexer::LexicalError>;

pub(super) fn parse_mapping(input: proc_macro2::TokenStream) -> Result<Mapping, syn::Error> {
    MappingParser::new()
        .parse(Lexer::new(input, LexerMode::Mapping))
        .map_err(|error| diagnostic::from_parse_error(error, LexerMode::Mapping))
}

pub(super) fn parse_index(input: proc_macro2::TokenStream) -> Result<Vec<IndexAssignment>, syn::Error> {
    IndexParser::new()
        .parse(Lexer::new(input, LexerMode::Index))
        .map_err(|error| diagnostic::from_parse_error(error, LexerMode::Index))
}

/// Map a parser-level `PaddingKind` to the `Ident` of its
/// `furiosa_mapping_types::PaddingKind` variant.
fn padding_kind_ident(kind: PaddingKind) -> proc_macro2::Ident {
    let name = match kind {
        PaddingKind::Top => "Top",
        PaddingKind::Zero => "Zero",
        PaddingKind::Bottom => "Bottom",
    };
    proc_macro2::Ident::new(name, proc_macro2::Span::call_site())
}

/// A numeric argument in `/`, `%`, or `=`: either a literal, or an escaped `{ const_expr }`
/// that resolves to a `usize`. Padding `#` accepts an [`Extent`] instead, which may also name
/// an axis and therefore stay in type space.
#[derive(Debug, Clone)]
pub enum Num {
    /// A bare integer literal.
    Lit(usize),
    /// An escaped constant expression, expanded as a braced const-generic argument.
    Const(proc_macro2::TokenStream),
}

impl Num {
    /// Expand as a const-generic argument: a bare literal, or a braced const expression
    /// (`{ Out::SIZE }`) so non-literal paths are accepted in const-generic position.
    fn expand(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Lit(n) => quote! { #n },
            Self::Const(tokens) => quote! { { #tokens } },
        }
    }
}

/// The extent a padding operator pads to. `Padding` takes it as a mapping, so a constant is
/// carried by `Broadcast` and an axis by `Symbol`. An extent named as an axis therefore never
/// reaches const-argument position, which is the only place a size read off a generic parameter
/// would need `generic_const_exprs`.
#[derive(Debug, Clone)]
pub enum Extent {
    /// A literal or escaped constant.
    Const(Num),
    /// An axis, named the same way a mapping names one.
    Axis(String),
    /// A parenthesized mapping, whose own extent is the factor. `Width / (Width / 512)` divides
    /// by however many 512-wide tiles `Width` holds without naming that count.
    Mapping(Box<Mapping>),
}

impl Extent {
    /// Expand as a mapping type, qualified by `path` so both the type position and the
    /// `map`-generating position can name it.
    fn expand(&self, path: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Self::Const(num) => {
                let value = num.expand();
                quote! { #path Broadcast<#value> }
            }
            Self::Axis(name) => {
                let ident = proc_macro2::Ident::new(name, proc_macro2::Span::call_site());
                quote! { #path Symbol<#ident> }
            }
            Self::Mapping(mapping) => mapping.expand(),
        }
    }
}

/// Parser-level representation of furiosa_mapping_types::PaddingKind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PaddingKind {
    /// Accessible padding holding arbitrary values.
    Top,
    /// Inaccessible padding. Emitted from the `m![A #{!} N]` syntax.
    Bottom,
    /// Accessible padding masked to a known zero. Emitted from `m![A #{0} N]`.
    Zero,
}

/// Representation of an index assignment (e.g., `A / 32 = 8` or `A = i`).
#[derive(Debug, Clone)]
pub struct IndexAssignment {
    /// The mapping expression.
    pub mapping: Mapping,
    /// The value expression.
    pub value: proc_macro2::TokenStream,
}

impl IndexAssignment {
    /// Expand the index assignment into code that adds it to an `Index`.
    pub fn expand(&self) -> proc_macro2::TokenStream {
        let value = &self.value;
        self.mapping.expand_as_index(value)
    }
}

/// Representation of a TCP mapping expression.
#[derive(Debug, Clone)]
pub enum Mapping {
    Broadcast {
        size: usize,
    },
    Symbol {
        symbol: String,
    },
    Stride {
        inner: Box<Self>,
        stride: Extent,
    },
    Modulo {
        inner: Box<Self>,
        modulo: Extent,
    },
    Resize {
        inner: Box<Self>,
        resize: Extent,
    },
    Padding {
        inner: Box<Self>,
        padding: Extent,
        kind: PaddingKind,
    },
    Pair {
        left: Box<Self>,
        right: Box<Self>,
    },
    Escaped {
        tokens: proc_macro2::TokenStream,
    },
}

impl Mapping {
    /// Expand the mapping into virtual ISA type representation.
    pub fn expand(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Broadcast { size } => {
                quote! { Broadcast<#size> }
            }
            Self::Symbol { symbol } => {
                let sym_ident = proc_macro2::Ident::new(symbol, proc_macro2::Span::call_site());
                quote! { Symbol<#sym_ident> }
            }
            Self::Stride {
                inner: left,
                stride: value,
            } => {
                let l = left.expand();
                let value = value.expand(&quote! {});
                quote! { Stride<#l, #value> }
            }
            Self::Modulo {
                inner: left,
                modulo: value,
            } => {
                let l = left.expand();
                let value = value.expand(&quote! {});
                quote! { Modulo<#l, #value> }
            }
            Self::Resize {
                inner: left,
                resize: value,
            } => {
                let l = left.expand();
                let value = value.expand(&quote! {});
                quote! { Resize<#l, #value> }
            }
            Self::Padding {
                inner: left,
                padding: value,
                kind,
            } => {
                let l = left.expand();
                let value = value.expand(&quote! {});
                let kind_ident = padding_kind_ident(*kind);
                quote! { Padding<#l, #value, { PaddingKind::#kind_ident }> }
            }
            Self::Pair { left, right } => {
                let l = left.expand();
                let r = right.expand();
                quote! { Pair<#l, #r> }
            }
            Self::Escaped { tokens } => {
                quote! { #tokens }
            }
        }
    }

    /// Expand the mapping into a cell contribution, preserving non-live results.
    pub fn expand_as_index(&self, value: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Self::Symbol { symbol } => {
                let sym_ident = proc_macro2::Ident::new(symbol, proc_macro2::Span::call_site());
                let size_expr = quote! { <#sym_ident as m::AxisName>::SIZE };
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        const SIZE: usize = #size_expr;
                        let mut index = m::Index::new();
                        index.add_term(
                            m::Term {
                                inner: m::Atom::Symbol {
                                    symbol: m::Ident::new(#symbol),
                                    size: SIZE,
                                },
                                stride: 1,
                                modulo: SIZE,
                            },
                            #value
                        );
                        cell = cell.combine(m::Cell::Index(index));
                    }
                }
            }
            Self::Broadcast { size } => {
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<m::Broadcast<#size> as m::M>::map(#value));
                    }
                }
            }
            Self::Stride { inner, stride } => {
                let inner_expanded = inner.expand();
                let stride = stride.expand(&quote! { m:: });
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<m::Stride<#inner_expanded, #stride> as m::M>::map(#value));
                    }
                }
            }
            Self::Modulo { inner, modulo } => {
                let inner_expanded = inner.expand();
                let modulo = modulo.expand(&quote! { m:: });
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<m::Modulo<#inner_expanded, #modulo> as m::M>::map(#value));
                    }
                }
            }
            Self::Resize { inner, resize } => {
                let inner_expanded = inner.expand();
                let resize = resize.expand(&quote! { m:: });
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<m::Resize<#inner_expanded, #resize> as m::M>::map(#value));
                    }
                }
            }
            Self::Padding { inner, padding, kind } => {
                let inner_expanded = inner.expand();
                let padding = padding.expand(&quote! { m:: });
                let kind_ident = padding_kind_ident(*kind);
                let pad_ty = quote! {
                    m::Padding<#inner_expanded, #padding, { m::PaddingKind::#kind_ident }>
                };
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<#pad_ty as m::M>::map(#value));
                    }
                }
            }
            Self::Pair { left, right } => {
                let left_expanded = left.expand();
                let right_expanded = right.expand();
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<m::Pair<#left_expanded, #right_expanded> as m::M>::map(#value));
                    }
                }
            }
            Self::Escaped { tokens } => {
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        cell = cell.combine(<#tokens as m::M>::map(#value));
                    }
                }
            }
        }
    }
}
