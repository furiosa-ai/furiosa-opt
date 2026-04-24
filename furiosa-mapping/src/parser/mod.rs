//! Macros for virtual ISA.

use crate::PaddingKind;
use lalrpop_util::lalrpop_mod;
use quote::quote;

mod lexer;
lalrpop_mod!(
    #[expect(missing_docs, missing_debug_implementations)]
    parser,
    "/parser/parser.rs"
);

pub use lexer::{Lexer, LexerMode};
pub use parser::{IndexParser, MappingParser};

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
#[expect(missing_docs)]
pub enum Mapping {
    Identity,
    Symbol {
        symbol: String,
    },
    Stride {
        inner: Box<Self>,
        stride: usize,
    },
    Modulo {
        inner: Box<Self>,
        modulo: usize,
    },
    Resize {
        inner: Box<Self>,
        resize: usize,
    },
    Padding {
        inner: Box<Self>,
        padding: usize,
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
            Self::Identity => {
                quote! { Identity }
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
                quote! { Stride<#l, #value> }
            }
            Self::Modulo {
                inner: left,
                modulo: value,
            } => {
                let l = left.expand();
                quote! { Modulo<#l, #value> }
            }
            Self::Resize {
                inner: left,
                resize: value,
            } => {
                let l = left.expand();
                quote! { Resize<#l, #value> }
            }
            Self::Padding {
                inner: left,
                padding: value,
                kind: _,
            } => {
                let l = left.expand();
                quote! { Padding<#l, #value> }
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

    /// Expand the mapping into an index addition operation.
    pub fn expand_as_index(&self, value: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Self::Symbol { symbol } => {
                let sym_ident = proc_macro2::Ident::new(symbol, proc_macro2::Span::call_site());
                let size_expr = quote! { <#sym_ident as m::AxisName>::SIZE };
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        const SIZE: usize = #size_expr;
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
                    }
                }
            }
            Self::Identity => {
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <m::Identity as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
            Self::Stride { inner, stride } => {
                let inner_expanded = inner.expand();
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <m::Stride<#inner_expanded, #stride> as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
            Self::Modulo { inner, modulo } => {
                let inner_expanded = inner.expand();
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <m::Modulo<#inner_expanded, #modulo> as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
            Self::Resize { inner, resize } => {
                let inner_expanded = inner.expand();
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <m::Resize<#inner_expanded, #resize> as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
            Self::Padding {
                inner,
                padding,
                kind: _,
            } => {
                let inner_expanded = inner.expand();
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <m::Padding<#inner_expanded, #padding> as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
            Self::Pair { left, right } => {
                let left_expanded = left.expand();
                let right_expanded = right.expand();
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <m::Pair<#left_expanded, #right_expanded> as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
            Self::Escaped { tokens } => {
                quote! {
                    {
                        use ::furiosa_mapping as m;
                        if let Some(mapped_index) = <#tokens as m::M>::map(#value) {
                            index.add(mapped_index);
                        } else {
                            index.mark_invalid();
                        }
                    }
                }
            }
        }
    }
}
