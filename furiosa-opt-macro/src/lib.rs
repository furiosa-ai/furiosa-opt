//! Macros for virtual ISA.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Item, Type, parse_macro_input};

/// Macro for mapping expressions.
///
/// See the documentation for `furiosa-visa-std` crate for details.
///
/// # Examples
///
/// ```ignore
/// use furiosa_visa_std::prelude::*;
/// axes![A = 512, B = 4];
/// type AB = m![A, B];
/// assert_eq!(AB::SIZE, 2048);
/// ```
#[proc_macro]
pub fn m(input: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = input.into();
    let lexer = furiosa_mapping::parser::Lexer::new(input, furiosa_mapping::parser::LexerMode::Mapping);
    let parser = furiosa_mapping::parser::MappingParser::new();
    let mapping = match parser.parse(lexer) {
        Ok(mapping) => mapping,
        Err(e) => {
            let msg = format!("Parse error: {:?}", e);
            return syn::Error::new(proc_macro2::Span::call_site(), msg)
                .to_compile_error()
                .into();
        }
    };
    let expanded = mapping.expand();
    quote! { #expanded }.into()
}

/// Macro for index expressions.
///
/// See the documentation for `furiosa-visa-std` crate for details.
///
/// # Examples
///
/// ```ignore
/// use furiosa_visa_std::prelude::*;
/// axes![A = 512, B = 64];
/// let idx = i![A / 32 = 8, B = 10];
/// ```
#[proc_macro]
pub fn i(input: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = input.into();
    let lexer = furiosa_mapping::parser::Lexer::new(input, furiosa_mapping::parser::LexerMode::Index);
    let parser = furiosa_mapping::parser::IndexParser::new();
    let assignments = match parser.parse(lexer) {
        Ok(assignments) => assignments,
        Err(e) => {
            let msg = format!("Parse error: {:?}", e);
            return syn::Error::new(proc_macro2::Span::call_site(), msg)
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

/// Derive macro for DeviceSend trait.
///
/// Generates implementation with bounds requiring all fields to be `DeviceSend`.
///
/// # Compile-time Checks
///
/// All fields must implement `DeviceSend`. This ensures:
/// - Reference fields are rejected (references don't impl DeviceSend)
/// - Nested types must also be DeviceSend
///
/// # Example
///
/// ```ignore
/// #[derive(DeviceSend)]
/// struct MyTensor<D: Scalar, Chip: M, Element: M> {
///     inner: Tensor<D, Pair<Chip, Element>>,  // Tensor must impl DeviceSend
/// }
/// // Generates:
/// // impl<...> DeviceSend for MyTensor<...>
/// // where
/// //     Tensor<...>: DeviceSend,
/// // {}
/// ```
#[proc_macro_derive(DeviceSend)]
pub fn device_send(input: TokenStream) -> TokenStream {
    /// Collect field types from a struct for where bounds.
    fn field_types(data: &Data) -> Vec<&Type> {
        match data {
            Data::Struct(data) => match &data.fields {
                Fields::Named(f) => f.named.iter().map(|f| &f.ty).collect(),
                Fields::Unnamed(f) => f.unnamed.iter().map(|f| &f.ty).collect(),
                Fields::Unit => vec![],
            },
            Data::Enum(_) | Data::Union(_) => vec![],
        }
    }

    /// Build where predicates requiring fields to be DeviceSend.
    fn device_send_predicates(field_types: &[&Type]) -> Vec<TokenStream2> {
        field_types
            .iter()
            .map(|ty| quote! { #ty: crate::runtime::DeviceSend })
            .collect()
    }

    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let fields = field_types(&input.data);
    let predicates = device_send_predicates(&fields);
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = if let Some(wc) = where_clause {
        quote! {
            impl #impl_generics crate::runtime::DeviceSend for #name #ty_generics
            #wc, #(#predicates),*
            {}
        }
    } else {
        quote! {
            impl #impl_generics crate::runtime::DeviceSend for #name #ty_generics
            where #(#predicates),*
            {}
        }
    };

    expanded.into()
}

/// Marks a function as a device entry point for `launch()`.
///
/// Generates a unit struct implementing `DeviceFn` with `execute()`.
/// `cargo <subcommand>`: `execute()` calls the original function body (CPU).
/// `cargo furiosa-opt <subcommand>`: `execute()` loads the compiled EDF and runs on NPU.
#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    fn to_camel(s: &str) -> String {
        s.split('_')
            .map(|w| {
                let mut c = w.chars();
                c.next()
                    .map_or(String::new(), |ch| ch.to_uppercase().collect::<String>() + c.as_str())
            })
            .collect()
    }

    let attr_str = attr.to_string();
    let func = match parse_macro_input!(item as Item) {
        Item::Fn(f) => f,
        other => {
            return syn::Error::new_spanned(other, "#[device] can only be applied to functions")
                .to_compile_error()
                .into();
        }
    };

    let vis = &func.vis;
    let name = &func.sig.ident;
    let name_str = name.to_string();
    let hidden = syn::Ident::new(&format!("__tcp_{name}"), name.span());
    let struct_name = syn::Ident::new(&to_camel(&name_str), name.span());
    let syn::Signature {
        inputs,
        output,
        generics,
        ..
    } = &func.sig;

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Context,
        Tensor,
    }

    let params: Vec<_> = inputs
        .iter()
        .filter_map(|a| match a {
            syn::FnArg::Typed(pt) => Some(pt),
            _ => None,
        })
        .enumerate()
        .map(|(i, pt)| {
            let name = match pt.pat.as_ref() {
                syn::Pat::Ident(id) => id.ident.clone(),
                _ => syn::Ident::new(&format!("__arg_{i}"), proc_macro2::Span::call_site()),
            };
            let ty = &pt.ty;
            let s = quote!(#ty).to_string();
            // Heuristic: Context params (DmaContext, TuContext, etc.) are CPU-side scheduling
            // abstractions that don't exist on device — they'll be prefixed `_` in execute().
            let kind = if s.contains("Context") {
                Kind::Context
            } else {
                Kind::Tensor
            };
            (name, quote!(#ty), kind)
        })
        .collect();

    let types: Vec<_> = params.iter().map(|(_, t, _)| t).collect();

    // For each tensor param, convert to a DMA Buffer before passing to Kernel::run().
    // Reference params (`&HbmTensor`): `(&*name).into()` to reborrow.
    // Owned params (`HbmTensorView`): `(&name).into()` since there's nothing to deref.
    let (tensor_bufs, tensor_stmts): (Vec<syn::Ident>, Vec<TokenStream2>) = params
        .iter()
        .filter(|(_, _, k)| *k == Kind::Tensor)
        .enumerate()
        .map(|(i, (name, ty, _))| {
            let buf = syn::Ident::new(&format!("__tcp_{i}"), proc_macro2::Span::call_site());
            let is_ref = ty.to_string().starts_with('&');
            let conv = if is_ref {
                quote! { let #buf: furiosa_visa_std::runtime::Buffer = (&*#name).into(); }
            } else {
                quote! { let #buf: furiosa_visa_std::runtime::Buffer = (&(#name)).into(); }
            };
            (buf, conv)
        })
        .unzip();

    let run_body = match output {
        syn::ReturnType::Type(_, ty) => quote! {
            let __tcp_out = __tcp_kernel.alloc(<#ty>::size());
            __tcp_kernel.run(&[#(#tensor_bufs),*], &[__tcp_out.clone()]).await;
            __tcp_out.into()
        },
        syn::ReturnType::Default => quote! {
            __tcp_kernel.run(&[#(#tensor_bufs),*], &[]).await;
        },
    };

    let tuple_type = if types.len() == 1 {
        quote!(#(#types)*)
    } else {
        quote!((#(#types),*))
    };
    let return_ty = match output {
        syn::ReturnType::Default => quote!(()),
        syn::ReturnType::Type(_, ty) => quote!(#ty),
    };
    let block = &func.block;

    // Destructure the tuple param of `execute()`. Context params are prefixed
    // with `_` because the NPU branch doesn't read them (kernels run on-device);
    // the CPU branch uses the _-prefixed names when calling the hidden fn.
    let param_names: Vec<syn::Ident> = params
        .iter()
        .map(|(n, _, k)| match k {
            Kind::Context => syn::Ident::new(&format!("_{n}"), n.span()),
            Kind::Tensor => n.clone(),
        })
        .collect();
    let body_destructure = if param_names.len() == 1 {
        quote!(#(#param_names)*)
    } else {
        quote!((#(#param_names),*))
    };

    let npu_body = quote! {
        static __TCP_KERNEL: furiosa_visa_std::OnceCell<furiosa_visa_std::runtime::Kernel> =
            furiosa_visa_std::OnceCell::const_new();
        let __tcp_kernel = __TCP_KERNEL.get_or_init(|| async {
            let __tcp_path = furiosa_visa_std::runtime::kernel_path(
                env!("FURIOSA_OPT_OUT_DIR"),
                env!("CARGO_PKG_NAME"),
                module_path!(),
                #name_str,
            );
            furiosa_visa_std::runtime::Kernel::load(&__tcp_path).await
        }).await;
        #(#tensor_stmts)*
        #run_body
    };
    let cpu_body = quote! { #hidden(#(#param_names),*) };

    quote! {
        #[tcp::device = #attr_str]
        // `#[allow]` (not `#[expect]`): the hidden fn may or may not trigger
        // each of these lints depending on how the user defined the device
        // function, and `#[expect]` fails when the lint doesn't fire.
        #[allow(dead_code, unused, clippy::too_many_arguments)]
        fn #hidden #generics (#inputs) #output #block

        #[derive(Debug)]
        #vis struct #struct_name;

        // `#[allow]`: `#[expect]` requires the lint to fire, which it does
        // for names like `my_fn` but NOT for capitalized device-function
        // names like `MatMul`.
        #[allow(non_upper_case_globals)]
        #vis const #name: #struct_name = #struct_name;

        impl #generics furiosa_visa_std::runtime::DeviceFn<#tuple_type> for #struct_name {
            type Output = #return_ty;
            fn execute(#body_destructure: #tuple_type) -> impl std::future::Future<Output = Self::Output> {
                async move {
                    #[cfg(furiosa_opt)]
                    { #npu_body }
                    #[cfg(not(furiosa_opt))]
                    { #cpu_body }
                }
            }
        }
    }
    .into()
}
