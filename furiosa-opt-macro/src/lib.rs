//! Macros for virtual ISA.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Item, Type, Variant, parse_macro_input, parse_quote};

#[proc_macro_attribute]
pub fn primitive(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_str = attr.to_string().trim_matches('"').to_owned();

    let mut item = parse_macro_input!(item as Item);
    if let Item::Enum(item_enum) = &mut item {
        for Variant { ident, attrs, .. } in &mut item_enum.variants {
            let variant_str = format!("{attr_str}::{ident}");
            attrs.push(parse_quote!(#[furiosa_opt::primitive = #variant_str]));
        }
    }

    let expanded = quote! {
        #[furiosa_opt::primitive = #attr_str]
        #item
    };
    expanded.into()
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
/// //
/// // impl<...> ExtendBuffers<MyTensor<...>> for Vec<Buffer>
/// // where
/// //     Vec<Buffer>: ExtendBuffers<...>,
/// // {
/// //     fn extend<__I: IntoIterator<Item = MyTensor<...>>(&mut self, iter: __I) {
/// //         for value in iter {
/// //             ExtendBuffers::extend(self, core::iter::once(value.accessor));
/// //             ...
/// //         }
/// //     }
/// // }
/// ```
#[proc_macro_derive(DeviceSend)]
pub fn device_send(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // DeviceSend models a device-function argument: a tensor, or a struct/tuple of
    // them flattened positionally into kernel inputs. Enums (variant-dependent layout)
    // and unions (no defined field set) have no positional flatten, so reject them.
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) | Data::Union(_) => {
            return syn::Error::new_spanned(name, "DeviceSend can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let tys: Vec<&Type> = fields.iter().map(|f| &f.ty).collect();
    let accessors: Vec<TokenStream2> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| match &f.ident {
            Some(ident) => quote!(#ident),
            None => {
                let index = syn::Index::from(i);
                quote!(#index)
            }
        })
        .collect();

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let augment = |bounds: Vec<TokenStream2>| match (where_clause, bounds.is_empty()) {
        (Some(clause), true) => quote!(#clause),
        (Some(clause), false) => quote!(#clause, #(#bounds),*),
        (None, true) => quote!(),
        (None, false) => quote!(where #(#bounds),*),
    };

    let sendable = augment(
        tys.iter()
            .map(|ty| quote!(#ty: ::furiosa_opt_std::runtime::DeviceSend))
            .collect(),
    );
    let bufferable = augment(
        tys.iter()
            .map(|ty| {
                quote! {
                    ::std::vec::Vec<::furiosa_opt_std::backend::npu::Buffer>:
                        ::furiosa_opt_std::backend::npu::ExtendBuffers<#ty>
                }
            })
            .collect(),
    );

    quote! {
        impl #impl_generics ::furiosa_opt_std::runtime::DeviceSend for #name #ty_generics
        #sendable {}

        // Flatten fields into buffers in declaration order (must match the compiler's
        // parameter lowering).
        impl #impl_generics ::furiosa_opt_std::backend::npu::ExtendBuffers<#name #ty_generics>
            for ::std::vec::Vec<::furiosa_opt_std::backend::npu::Buffer>
        #bufferable
        {
            fn extend<__I: ::core::iter::IntoIterator<Item = #name #ty_generics>>(&mut self, iter: __I) {
                for value in iter {
                    #(
                        ::furiosa_opt_std::backend::npu::ExtendBuffers::extend(
                            self,
                            ::core::iter::once(value.#accessors),
                        );
                    )*
                }
            }
        }
    }
    .into()
}

/// Marks a function as a device entry point for `launch()`.
///
/// Generates a unit struct implementing `DeviceFn` with `execute()`.
/// `cargo <subcommand>`: `execute()` calls the original function body (CPU).
/// `cargo furiosa-opt <subcommand>`: `execute()` loads the compiled EDF and runs on NPU.
#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_str = attr.to_string();
    let attr_int = |key: &str, default: usize| -> usize {
        attr_str
            .split(',')
            .filter_map(|kv| kv.split_once('='))
            .find(|(k, _)| k.trim() == key)
            .and_then(|(_, v)| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let device_chip = attr_int("chip", 1) as u8;
    let device_pe = attr_int("pe", 8) as u8;
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
    let hidden = syn::Ident::new(&format!("__furiosa_opt_{name}"), name.span());
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

    // Convert tensor params to DMA Buffers via `ExtendBuffers`, whose trait dispatch
    // recursively flattens tuple params (e.g. `(&HbmTensor, &HbmTensor)`) into one buffer
    // per leaf tensor, in field order.
    let tensor_param_names: Vec<&syn::Ident> = params
        .iter()
        .filter(|(_, _, k)| *k == Kind::Tensor)
        .map(|(name, _, _)| name)
        .collect();

    let tensor_stmts: TokenStream2 = quote! {
        let mut __furiosa_opt_bufs: ::std::vec::Vec<furiosa_opt_std::backend::npu::Buffer> =
            ::std::vec::Vec::new();
        furiosa_opt_std::backend::npu::ExtendBuffers::extend(
            &mut __furiosa_opt_bufs,
            ::std::iter::once((#(#tensor_param_names,)*)),
        );
    };

    // Allocate one buffer per output, run, then rebuild the return value from the filled buffers.
    let run_body = match output {
        syn::ReturnType::Type(_, ty) => quote! {
            let __furiosa_opt_outs =
                <#ty as furiosa_opt_std::backend::npu::KernelOutput>::alloc_outputs(__furiosa_opt_kernel);
            __furiosa_opt_kernel.run(&__furiosa_opt_bufs, &__furiosa_opt_outs).await;
            <#ty as furiosa_opt_std::backend::npu::KernelOutput>::from_buffers(__furiosa_opt_outs)
        },
        syn::ReturnType::Default => quote! {
            __furiosa_opt_kernel.run(&__furiosa_opt_bufs, &[]).await;
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

    // Under `furiosa_opt` (the driver's scan/compile passes) no `.bin` exists yet, so embed nothing.
    let npu_body = quote! {
        static __FURIOSA_OPT_KERNEL: furiosa_opt_std::OnceCell<furiosa_opt_std::backend::npu::Kernel> =
            furiosa_opt_std::OnceCell::const_new();
        #[cfg(furiosa_opt)]
        let __furiosa_opt_kernel = __FURIOSA_OPT_KERNEL
            .get_or_init(|| async { furiosa_opt_std::backend::npu::Kernel::load(&[]).await })
            .await;
        #[cfg(not(furiosa_opt))]
        let __furiosa_opt_kernel = __FURIOSA_OPT_KERNEL
            .get_or_init(|| async {
                furiosa_opt_std::backend::npu::Kernel::load(include_bytes!(concat!(
                    env!("FURIOSA_OPT_OUT_DIR"), "/", env!("CARGO_PKG_NAME"), "/",
                    module_path!(), "::", #name_str, ".bin"
                )))
                .await
            })
            .await;
        #tensor_stmts
        #run_body
    };
    let cpu_body = quote! { #hidden(#(#param_names),*) };

    quote! {
        #[furiosa_opt::device = #attr_str]
        // `#[allow]` (not `#[expect]`): the hidden fn may or may not trigger
        // each of these lints depending on how the user defined the device
        // function, and `#[expect]` fails when the lint doesn't fire.
        #[allow(dead_code, unused, clippy::too_many_arguments)]
        fn #hidden #generics (#inputs) #output #block

        // Marker struct: the `__furiosa_opt_` prefix dodges a same-named module, and the braced (non-unit)
        // form keeps it out of the value namespace so it coexists with the hidden fn; npu `scan` strips it.
        #[allow(non_camel_case_types)]
        #[derive(Debug)]
        #vis struct #hidden {}

        // `#[allow]`: the const keeps the snake device-fn name, which trips `non_upper_case_globals`.
        #[allow(non_upper_case_globals)]
        #vis const #name: #hidden = #hidden {};

        impl #hidden {
            /// The logical [`furiosa_opt_std::Device`] this kernel runs on, from `#[device(chip, pe)]`.
            /// Pass to `Context::acquire().bind(..)` before any host I/O.
            pub fn device(&self) -> furiosa_opt_std::Device {
                furiosa_opt_std::Device { chip: #device_chip, pe: #device_pe }
            }
        }

        impl #generics furiosa_opt_std::runtime::DeviceFn<#tuple_type> for #hidden {
            type Output = #return_ty;
            fn execute(#body_destructure: #tuple_type) -> impl std::future::Future<Output = Self::Output> {
                async move {
                    #[cfg(backend = "npu")]
                    { #npu_body }
                    #[cfg(not(backend = "npu"))]
                    { #cpu_body }
                }
            }
        }
    }
    .into()
}
