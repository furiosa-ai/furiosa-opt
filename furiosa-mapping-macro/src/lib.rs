//! Macros for mapping expressions.

#![feature(proc_macro_diagnostic)]

mod parser;

use std::collections::BTreeMap;

use proc_macro::{Diagnostic, Level, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::{Ident, LitInt, Token};

/// Macro to define shapes using axis names and sizes.
///
/// Each `NAME = SIZE` pair expands to a unit struct with an `AxisName` impl.
/// Duplicate names (e.g. `axes![A = 8, A = 16]`) emit a compile error.
///
/// # Examples
///
/// ```ignore
/// use furiosa_opt_std::prelude::*;
/// axes![A = 128, B = 64, C = 32];
/// ```
///
/// ## Compile Diagnostics
///
/// Duplicated axis names are rejected.
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// // This fails to compile
/// axes![A = 8, A = 16];
/// ```
///
/// Invalid comma placement is rejected.
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// // These fail to compile
/// axes![,];
/// ```
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// axes![A = 4,, B = 2];
/// ```
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// axes![,A = 4];
/// ```
#[proc_macro]
pub fn axes(input: TokenStream) -> TokenStream {
    struct AxisDecl {
        name: Ident,
        size: LitInt,
    }
    impl Parse for AxisDecl {
        fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
            let name: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let size: LitInt = input.parse()?;
            Ok(AxisDecl { name, size })
        }
    }

    let parser = Punctuated::<AxisDecl, Token![,]>::parse_terminated;
    let decls = match parser.parse(input) {
        Ok(d) => d,
        Err(e) => return e.to_compile_error().into(),
    };

    // Check duplicated axes
    let mut first_seen = BTreeMap::new();
    let mut has_errors = false;
    for d in &decls {
        let key = d.name.to_string();
        if let Some(&first_span) = first_seen.get(&key) {
            let duplicate_span = d.name.span().unwrap();
            Diagnostic::spanned(
                duplicate_span,
                Level::Error,
                format!("axis `{key}` is declared multiple times in this axes! invocation"),
            )
            .span_note(first_span, format!("first declaration of `{key}` is here"))
            .emit();
            has_errors = true;
        } else {
            first_seen.insert(key, d.name.span().unwrap());
        }
    }
    if has_errors {
        return TokenStream::new();
    }

    let items = decls.iter().map(|d| {
        let name = &d.name;
        let size = &d.size;
        quote! {
            #[allow(non_camel_case_types)]
            #[derive(Debug, Clone)]
            pub struct #name;
            impl AxisName for #name {
                const NAME: Ident = Ident::new(::core::stringify!(#name));
                const SIZE: usize = #size;
            }
        }
    });
    quote! { #(#items)* }.into()
}

/// Macro for mapping expressions.
///
/// See the documentation for `furiosa-opt-std` crate for details.
///
/// # Examples
///
/// ```ignore
/// use furiosa_opt_std::prelude::*;
/// axes![A = 512, B = 4];
/// type AB = m![A, B];
/// assert_eq!(AB::SIZE, 2048);
/// ```
///
/// ## Compile Diagnostics
///
/// Bare integer literals other than `1` are rejected.
/// Declare a named axis with `axes![NAME = N]` and use `m![NAME]`.
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// // This fails to compile
/// type _T = m![64];
/// ```
///
/// Invalid comma placement is rejected.
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// // These fail to compile
/// type _T = m![,];
/// ```
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// # axes![A = 4, B = 2];
/// type _T = m![A,,B];
/// ```
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// # axes![A = 4];
/// type _T = m![,A];
/// ```
#[proc_macro]
pub fn m(input: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = input.into();
    let lexer = parser::Lexer::new(input, parser::LexerMode::Mapping);
    let parser = parser::MappingParser::new();
    let mapping = match parser.parse(lexer) {
        Ok(mapping) => mapping,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e.to_string())
                .to_compile_error()
                .into();
        }
    };
    let expanded = mapping.expand();
    quote! { #expanded }.into()
}

/// Macro for index expressions.
///
/// See the documentation for `furiosa-opt-std` crate for details.
///
/// # Examples
///
/// ```ignore
/// use furiosa_opt_std::prelude::*;
/// axes![A = 512, B = 64];
/// let idx = i![A / 32 = 8, B = 10];
/// ```
///
/// ## Compile Diagnostics
///
/// Invalid comma placement is rejected.
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// // These fail to compile
/// let _ = i![,];
/// ```
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// # axes![A = 4, B = 2];
/// let _ = i![A,,B: 0];
/// ```
///
/// ```compile_fail
/// # use furiosa_mapping::*;
/// # axes![A = 4];
/// let _ = i![,A: 0];
/// ```
#[proc_macro]
pub fn i(input: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = input.into();
    let lexer = parser::Lexer::new(input, parser::LexerMode::Index);
    let parser = parser::IndexParser::new();
    let assignments = match parser.parse(lexer) {
        Ok(assignments) => assignments,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e.to_string())
                .to_compile_error()
                .into();
        }
    };

    let expansions = assignments.iter().map(|assignment| assignment.expand());

    quote! {
        {
            let mut index = ::furiosa_mapping::Index::new();
            #(#expansions)*
            index
        }
    }
    .into()
}
