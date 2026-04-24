use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, Variant, parse_macro_input, parse_quote};

#[proc_macro_attribute]
pub fn primitive(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_str = attr.to_string().trim_matches('"').to_owned();

    let mut item = parse_macro_input!(item as Item);
    if let Item::Enum(item_enum) = &mut item {
        for Variant { ident, attrs, .. } in &mut item_enum.variants {
            let variant_str = format!("{attr_str}::{ident}");
            attrs.push(parse_quote!(#[tcp::primitive = #variant_str]));
        }
    }

    let expanded = quote! {
        #[tcp::primitive = #attr_str]
        #item
    };
    expanded.into()
}
