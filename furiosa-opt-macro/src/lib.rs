//! Macros for virtual ISA.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::visit_mut::{self, VisitMut};
use syn::{
    Attribute, Data, DeriveInput, Error, ExprForLoop, Item, Meta, Type, Variant, parse_macro_input, parse_quote,
};

/// The rev of this macro's source tree, read at first expansion so no build script has to
/// stamp it (a build-script trigger on the macro would rebuild every dependent crate).
/// `FURIOSA_OPT_REV` overrides it (a pipeline building outside the origin tree). A published
/// snapshot carries its origin rev in the crate version (`0.5.1+g<sha>`), and that stamp wins
/// over git: a mirror checkout's own sha is the MIRROR commit, not the rev the driver was
/// built from. A registry checkout has neither and reads `unknown`, which the driver skips.
fn rev() -> &'static str {
    static REV: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    REV.get_or_init(|| {
        std::env::var("FURIOSA_OPT_REV")
            .ok()
            .or_else(|| {
                env!("CARGO_PKG_VERSION")
                    .split_once("+g")
                    .map(|(_, sha)| sha.to_owned())
                    .filter(|sha| !sha.is_empty())
            })
            .or_else(|| {
                let out = std::process::Command::new("git")
                    .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--short=12", "HEAD"])
                    .output()
                    .ok()?;
                out.status
                    .success()
                    .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            })
            .unwrap_or_else(|| "unknown".into())
    })
}

/// Fully unrolls the annotated `for` loop before device scheduling.
///
/// The loop may be in a [`device`] function or any function reachable from one.
#[proc_macro_attribute]
pub fn unroll(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new_spanned(TokenStream2::from(attr), "`unroll` takes no arguments")
            .to_compile_error()
            .into();
    }

    let mut loop_expr = parse_macro_input!(item as ExprForLoop);
    let mut errors = Vec::new();
    apply_unroll(&mut loop_expr, true, &mut errors);
    if !errors.is_empty() {
        let errors = errors.iter().map(Error::to_compile_error);
        return quote!(#(#errors)*).into();
    }
    quote!(#loop_expr).into()
}

fn insert_unroll_marker(loop_expr: &mut ExprForLoop) {
    loop_expr
        .body
        .stmts
        .insert(0, parse_quote!(::furiosa_opt_std::__private::__loop_hint_unroll();));
}

#[derive(Default)]
struct UnrollMarkerInserter {
    errors: Vec<Error>,
}

impl VisitMut for UnrollMarkerInserter {
    fn visit_expr_for_loop_mut(&mut self, loop_expr: &mut ExprForLoop) {
        // Consume loop-owned attributes before the generic visitor rejects misplaced ones.
        apply_unroll(loop_expr, false, &mut self.errors);
        visit_mut::visit_expr_for_loop_mut(self, loop_expr);
    }

    fn visit_attribute_mut(&mut self, attribute: &mut Attribute) {
        if is_unroll(attribute) {
            self.errors.push(Error::new_spanned(
                attribute,
                "`unroll` goes on a `for` loop, and this is not one",
            ));
        }
    }
}

fn apply_unroll(loop_expr: &mut ExprForLoop, mut requested: bool, errors: &mut Vec<Error>) {
    loop_expr.attrs.retain(|attribute| {
        if !is_unroll(attribute) {
            return true;
        }
        match attribute.meta {
            Meta::Path(_) if !requested => requested = true,
            Meta::Path(_) => errors.push(Error::new_spanned(attribute, "duplicate `unroll` attribute")),
            _ => errors.push(Error::new_spanned(attribute, "`unroll` takes no arguments")),
        }
        false
    });
    if requested {
        insert_unroll_marker(loop_expr);
    }
}

fn is_unroll(attribute: &Attribute) -> bool {
    attribute.path().is_ident("unroll")
}

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
/// Generates an owned implementation for structs whose fields are `DeviceSend`.
///
/// # Compile-time Checks
///
/// Nested derived structs and arrays flatten their fields recursively.
///
#[proc_macro_derive(DeviceSend)]
pub fn device_send(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // DeviceSend models a device-function argument: a tensor, or a struct/tuple of
    // them flattened positionally into launch inputs. Enums (variant-dependent layout)
    // and unions (no defined field set) have no positional flatten, so reject them.
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) | Data::Union(_) => {
            return syn::Error::new_spanned(name, "DeviceSend can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let tys: Vec<_> = fields.iter().map(|f| &f.ty).collect();
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
    let augment =
        |where_clause: Option<&syn::WhereClause>, bounds: Vec<TokenStream2>| match (where_clause, bounds.is_empty()) {
            (Some(clause), true) => quote!(#clause),
            (Some(clause), false) => quote!(#clause, #(#bounds),*),
            (None, true) => quote!(),
            (None, false) => quote!(where #(#bounds),*),
        };

    let sendable = augment(
        where_clause,
        tys.iter()
            .map(|ty| quote!(#ty: ::furiosa_opt_std::runtime::DeviceSend))
            .collect(),
    );

    quote! {
        impl #impl_generics ::furiosa_opt_std::runtime::DeviceSend for #name #ty_generics
        #sendable {
            fn bind(&self, buffers: &mut ::furiosa_opt_std::runtime::Buffers) -> Result<(), ::furiosa_opt_std::Error> {
                #(::furiosa_opt_std::runtime::DeviceSend::bind(&self.#accessors, buffers)?;)*
                Ok(())
            }
        }
    }
    .into()
}

/// Marks a function as a device entry point for `launch()`.
///
/// Generates a unit struct implementing `DeviceFn` with `execute()`.
/// `cargo <subcommand>`: `execute()` calls the original function body (CPU).
/// `cargo furiosa-opt <subcommand>`: `execute()` loads the compiled registry entry and runs on NPU.
#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_device(&attr.to_string(), parse_macro_input!(item as Item)).into()
}

/// The `proc_macro` bridge exists only inside a real expansion, so the attribute's work takes
/// parsed input: unit tests drive it on `syn::parse_quote!` items.
fn expand_device(attr: &str, item: Item) -> TokenStream2 {
    let attr_int = |key: &str, default: usize| -> usize {
        attr.split(',')
            .filter_map(|kv| kv.split_once('='))
            .find(|(k, _)| k.trim() == key)
            .and_then(|(_, v)| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let devices = attr_int("chip", 1) as u8;
    let pes = attr_int("pe", 8) as u8;
    let mut func = match item {
        Item::Fn(f) => f,
        other => {
            return syn::Error::new_spanned(other, "#[device] can only be applied to functions").to_compile_error();
        }
    };

    let mut unroll_markers = UnrollMarkerInserter::default();
    unroll_markers.visit_block_mut(&mut func.block);
    if !unroll_markers.errors.is_empty() {
        let errors = unroll_markers.errors.iter().map(Error::to_compile_error);
        return quote!(#(#errors)*);
    }

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
    let is_context = |ty: &Type| {
        let Type::Reference(reference) = ty else {
            return false;
        };
        if reference.mutability.is_none() {
            return false;
        }
        let Type::Path(path) = reference.elem.as_ref() else {
            return false;
        };
        path.qself.is_none()
            && path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Device")
    };
    let Some(syn::FnArg::Typed(context)) = inputs.first() else {
        return syn::Error::new_spanned(&func.sig, "the first #[device] parameter must be &mut Device")
            .to_compile_error();
    };
    if !is_context(&context.ty) {
        return syn::Error::new_spanned(&context.ty, "the first #[device] parameter must be &mut Device")
            .to_compile_error();
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
            (name, quote!(#ty))
        })
        .collect();

    let types: Vec<_> = params.iter().map(|(_, ty)| ty).collect();

    // The leading `Device` names the runtime the function loads into; every other parameter,
    // lowering contexts included, binds positionally (contexts bind to nothing).
    let context = &params[0].0;
    let arg_names: Vec<&syn::Ident> = params.iter().skip(1).map(|(name, _)| name).collect();

    let bind_stmts: TokenStream2 = quote! {
        let mut __furiosa_opt_inputs = furiosa_opt_std::runtime::Buffers::new();
        furiosa_opt_std::runtime::DeviceSend::bind(&(#(#arg_names,)*), &mut __furiosa_opt_inputs)?;
        let mut __furiosa_opt_outputs = furiosa_opt_std::runtime::Buffers::new();
    };

    let return_ty = match output {
        syn::ReturnType::Default => quote!(()),
        syn::ReturnType::Type(_, ty) => quote!(#ty),
    };
    let run_body = quote! {
        <#return_ty as furiosa_opt_std::__private::DeviceOutput>::alloc_into(
            &__furiosa_opt_function,
            &mut __furiosa_opt_outputs,
        )?;
        __furiosa_opt_function.run(&__furiosa_opt_inputs, &__furiosa_opt_outputs).await?;
        Ok(<#return_ty as furiosa_opt_std::__private::DeviceOutput>::take(&mut __furiosa_opt_outputs.into_iter()))
    };
    let run_into_body = quote! {
        furiosa_opt_std::runtime::DeviceSend::bind(&*__furiosa_opt_destination, &mut __furiosa_opt_outputs)?;
        __furiosa_opt_function.run(&__furiosa_opt_inputs, &__furiosa_opt_outputs).await
    };

    let tuple_type = if types.len() == 1 {
        quote!(#(#types)*)
    } else {
        quote!((#(#types),*))
    };
    let block = &func.block;

    let param_names: Vec<&syn::Ident> = params.iter().map(|(name, _)| name).collect();
    let body_destructure = if param_names.len() == 1 {
        quote!(#(#param_names)*)
    } else {
        quote!((#(#param_names),*))
    };

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let generic_keys: Vec<_> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Lifetime(_) => None,
            syn::GenericParam::Type(param) => {
                let ident = &param.ident;
                Some(quote!(<#ident as ::furiosa_opt_std::prelude::AxisName>::SIZE))
            }
            syn::GenericParam::Const(param) => {
                let ident = &param.ident;
                Some(quote!(#ident))
            }
        })
        .collect();

    // The launch selects its image from the binary's `furiosa_kernels` registry by the fn's
    // path and its numeric axis/const values. A concrete fn is the empty-key case.
    let kernel_stmts = quote! {
        let __furiosa_opt_function = furiosa_opt_std::backend::npu::function(
            #context,
            concat!(module_path!(), "::", #name_str),
            &[#(#generic_keys),*],
        )
        .await?;
    };
    let npu_body = quote! {
        #kernel_stmts
        #bind_stmts
        #run_body
    };
    let npu_into_body = quote! {
        #kernel_stmts
        #bind_stmts
        #run_into_body
    };
    let cpu_body = quote! { Ok(self::#hidden(#(#param_names),*)) };
    let cpu_into_body = quote! { *__furiosa_opt_destination = self::#hidden(#(#param_names),*); Ok(()) };

    let rev = rev();
    quote! {
        #[furiosa_opt::rev = #rev]
        #[furiosa_opt::device = #attr]
        // `#[allow]` (not `#[expect]`): the hidden fn may or may not trigger
        // each of these lints depending on how the user defined the device
        // function, and `#[expect]` fails when the lint doesn't fire.
        #[allow(dead_code, unused, clippy::too_many_arguments)]
        fn #hidden #impl_generics (#inputs) #output #where_clause #block

        // Marker struct: the `__furiosa_opt_` prefix dodges a same-named module, and the braced (non-unit)
        // form keeps it out of the value namespace so it coexists with the hidden fn; npu `scan` strips it.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug)]
        #vis struct #hidden {}

        // `#[allow]`: the const keeps the snake device-fn name, which trips `non_upper_case_globals`.
        #[allow(non_upper_case_globals)]
        #vis const #name: #hidden = #hidden {};

        impl #hidden {
            /// The device this function runs on, as `#[device(chip, pe)]` declares it.
            pub fn topology(&self) -> furiosa_opt_std::Topology {
                furiosa_opt_std::Topology { chips: #devices, pes: #pes }
            }

            /// The name the compiler uses for this function's registry entry.
            /// A caller that filters functions takes it from here rather than restating the path.
            #[doc(hidden)]
            pub fn path(&self) -> &'static str {
                concat!(module_path!(), "::", #name_str)
            }
        }

        impl #impl_generics furiosa_opt_std::runtime::DeviceFn<#tuple_type> for #hidden #where_clause {
            type Output = #return_ty;
            fn execute(#body_destructure: #tuple_type) -> impl std::future::Future<Output = Result<Self::Output, furiosa_opt_std::Error>> {
                async move {
                    #[cfg(backend = "npu")]
                    { #npu_body }
                    #[cfg(not(backend = "npu"))]
                    { #cpu_body }
                }
            }

            fn execute_into(
                #body_destructure: #tuple_type,
                __furiosa_opt_destination: &mut Self::Output,
            ) -> impl std::future::Future<Output = Result<(), furiosa_opt_std::Error>> {
                async move {
                    #[cfg(backend = "npu")]
                    { #npu_into_body }
                    #[cfg(not(backend = "npu"))]
                    { #cpu_into_body }
                }
            }
        }
    }
}

#[cfg(test)]
mod device_tests {
    use syn::{Item, parse_quote};

    use super::expand_device;

    #[test]
    fn accepts_only_mutable_context_references() {
        let rejects = |item: Item| expand_device("chip = 1", item).to_string().contains("compile_error");

        assert!(!rejects(parse_quote!(
            fn f(device: &'a mut furiosa_opt_std::Device) {}
        )));
        assert!(!rejects(parse_quote!(
            fn f(device: &mut Device<furiosa_opt_std::backend::Npu>) {}
        )));
        assert!(rejects(parse_quote!(
            fn f(device: &Device) {}
        )));
        assert!(rejects(parse_quote!(
            fn f(device: &mut TuContext<{ Tu::Main }>) {}
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_unroll_is_rejected_during_macro_expansion() {
        let mut block = parse_quote!({
            #[unroll]
            #[unroll]
            for _ in 0..4 {}
        });
        let mut inserter = UnrollMarkerInserter::default();

        inserter.visit_block_mut(&mut block);

        assert_eq!(inserter.errors.len(), 1);
        assert_eq!(inserter.errors[0].to_string(), "duplicate `unroll` attribute");
    }
}
