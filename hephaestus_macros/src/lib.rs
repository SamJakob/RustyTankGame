mod helpers;

use helpers::camel_to_snake_case;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, AttrStyle, Data, DeriveInput, Field, Ident};

/// empty_token_stream is a sugar that does [TokenStream::new] and then uses [TokenStream::into] to
/// convert it to the context-specific type.
macro_rules! empty_token_stream {
    () => {
        TokenStream::new().into()
    };
}

#[proc_macro]
pub fn vector_field(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Ident);
    let input_name = input.to_string();

    let is_float = input_name.starts_with("f") || input_name.starts_with("Float");
    let is_nan = if is_float {
        quote!(self == #input::NAN)
    } else {
        quote!(false)
    };

    TokenStream::from(quote!(
        impl VectorField for #input {

            fn is_nan(self) -> bool {
                #is_nan
            }

            fn abs(self) -> Self {
                self.abs()
            }

            fn sqrt(self) -> Self {
                self.sqrt()
            }

        }
    ))
}

#[proc_macro_derive(VectorBase, attributes(dim))]
pub fn derive_vector_base(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    if let Data::Struct(data) = input.data {
        // Get the name and fields of the struct.
        let name = input.ident;

        let fields = data.fields;
        let fields_with_dim = fields
            .iter()
            .filter(|field| {
                field
                    .attrs
                    .iter()
                    .any(|attr| attr.style == AttrStyle::Outer && attr.path().is_ident("dim"))
            })
            .collect::<Vec<&Field>>();

        // If there is at least one dimension attribute, include only those fields. Otherwise,
        // consider all fields to be dimension attributes.
        let use_attributes = !fields_with_dim.is_empty();
        let effective_fields = if use_attributes {
            fields_with_dim
        } else {
            fields.iter().collect()
        };

        let size = effective_fields.len();

        fn build_fields(fields: &Vec<&Field>, prefix: Option<&str>) -> Vec<syn::Expr> {
            fields
                .iter()
                .map(|field| {
                    syn::parse_str::<syn::Expr>(
                        format!("{}{}", prefix.unwrap_or(""), field.ident.clone().unwrap(),)
                            .as_str(),
                    )
                    .unwrap()
                })
                .collect()
        }

        fn build_entries(
            fields: &Vec<&Field>,
            prefix: Option<&str>,
            definition: impl Fn(usize) -> String,
        ) -> Vec<proc_macro2::TokenStream> {
            fields
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    let k = syn::parse_str::<syn::Expr>(
                        format!("{}{}", prefix.unwrap_or(""), field.ident.clone().unwrap(),)
                            .as_str(),
                    )
                    .unwrap();

                    let v = syn::parse_str::<syn::Expr>(definition(i).as_str()).unwrap();

                    quote!(#k: #v)
                })
                .collect()
        }

        let get_vector_fields = build_fields(&effective_fields, Some("self."));
        let for_each_vector_field = build_fields(&effective_fields, Some("&mut self."));
        let construct_from_vector_fields =
            build_entries(&effective_fields, None, |i| format!("fields[{}]", i));

        return TokenStream::from(quote!(
            impl<T> VectorBase<T, #size> for #name<T>
            where
                T: VectorField,
            {
                fn get_vector_fields(self) -> [T; #size] {
                    [#(#get_vector_fields),*]
                }

                fn for_each_vector_field(&mut self, callback: impl Fn(usize, &mut T)) {
                    [#(#for_each_vector_field),*]
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, field)| callback(i, *field));
                }

                fn construct_from_vector_fields(fields: [T; #size]) -> Self {
                    #name {
                        #(#construct_from_vector_fields),*
                    }
                }
            }
        ));
    }

    TokenStream::from(
        syn::Error::new(
            input.ident.span(),
            "Only structs with named fields can derive `VectorBase'",
        )
        .to_compile_error(),
    )
}

#[proc_macro_derive(VectorBaseImpl, attributes(foo))]
pub fn derive_impl_from_vector_base(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Ensure the structure annotated with derive is a struct.
    let Data::Struct(_) = input.data else {
        return TokenStream::from(
            syn::Error::new(
                input.ident.span(),
                "Only structs with named fields can derive `VectorBaseImpl'",
            )
            .to_compile_error(),
        );
    };

    // Get the name of the struct.
    let name = input.ident;

    // Define the implementations for the type.
    let impls = vec![
        build_arithmetic_mapping_impl_for("std::ops::Add", &name, quote!(+)),
        build_arithmetic_mapping_impl_for("std::ops::AddAssign", &name, quote!(+=)),
        build_arithmetic_mapping_impl_for("std::ops::Sub", &name, quote!(-)),
        build_arithmetic_mapping_impl_for("std::ops::SubAssign", &name, quote!(-=)),
        build_scalar_mapping_impl_for("std::ops::Mul", &name, quote!(*)),
        build_scalar_mapping_impl_for("std::ops::MulAssign", &name, quote!(*=)),
        build_scalar_mapping_impl_for("std::ops::Div", &name, quote!(/)),
        build_scalar_mapping_impl_for("std::ops::DivAssign", &name, quote!(/=)),
    ];

    TokenStream::from(quote!(
        #(#impls)*
    ))
}

/// build_arithmetic_mapping_impl_for the struct with the given name. The mapping tokens are the
/// mapping performed on the LHS of the operation (for example "quote!(+)").
///
/// The logic is automatically switched based on whether the trait is an Assign trait or not.
fn build_arithmetic_mapping_impl_for(
    raw_trait_name: &str,
    name: &Ident,
    mapping: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    build_mapping_impl_for(raw_trait_name, name, mapping, |options| {
        let name = options.name.into_token_stream();
        let trait_name = options.trait_name.into_token_stream();
        let mapping = options.context;

        if options.is_assign {
            MappingImpl {
                trait_name: trait_name.clone(),
                bound: trait_name,
                output_type: empty_token_stream!(),
                self_ref: quote!(&mut self),
                rhs_type: quote!(Self),
                output: empty_token_stream!(),
                field_mapping: quote!(
                    let rhs_fields = rhs.get_vector_fields();

                    self.for_each_vector_field(|i, field| {
                        *field #mapping rhs_fields[i];
                    })
                ),
            }
        } else {
            MappingImpl {
                trait_name: trait_name.clone(),
                bound: quote!(#trait_name<Output = T>),
                output_type: quote!(type Output = #name<T>;),
                self_ref: quote!(self),
                rhs_type: quote!(Self),
                output: quote!( -> Self),
                field_mapping: quote!(
                    Self::construct_from_vector_fields(
                        self.get_vector_fields()
                            .iter()
                            .zip(rhs.get_vector_fields())
                            .map(|(x, y)| *x #mapping y)
                            .collect::<Vec<T>>()
                            .try_into()
                            .unwrap_or_else(|v: Vec<T>| panic!(
                                "Expected a Vec of len {}, got len {}",
                                Self::DIMENSION_COUNT,
                                v.len()
                            ))
                    )
                ),
            }
        }
    })
}

fn build_scalar_mapping_impl_for(
    raw_trait_name: &str,
    name: &Ident,
    mapping: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    build_mapping_impl_for(raw_trait_name, name, mapping, |options| {
        let name = options.name.into_token_stream();
        let mapping = options.context;

        let bound = options.trait_name.clone().into_token_stream();
        let base_trait_name = options.trait_name;
        let trait_name = quote!(#base_trait_name<T>);

        if options.is_assign {
            MappingImpl {
                trait_name,
                bound,
                output_type: empty_token_stream!(),
                self_ref: quote!(&mut self),
                rhs_type: quote!(T),
                output: empty_token_stream!(),
                field_mapping: quote!(
                    self.for_each_vector_field(|_, field| {
                        *field #mapping rhs;
                    })
                ),
            }
        } else {
            MappingImpl {
                trait_name,
                bound: quote!(#base_trait_name<Output = T>),
                output_type: quote!(type Output = #name<T>;),
                self_ref: quote!(self),
                rhs_type: quote!(T),
                output: quote!( -> Self),
                field_mapping: quote!(
                    Self::construct_from_vector_fields(
                        self.get_vector_fields()
                            .iter()
                            .map(|x| *x #mapping rhs)
                            .collect::<Vec<T>>()
                            .try_into()
                            .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of len {}, got len {}", Self::DIMENSION_COUNT, v.len()))
                    )
                ),
            }
        }
    })
}

struct MappingImplOptions<'a, T> {
    /// name is the type that the mapping trait is being implemented for.
    name: &'a Ident,

    /// is_assign is true if and only if the trait being implemented is an assign trait (i.e., where
    /// the target struct, Self, should be updated in-place).
    is_assign: bool,

    /// trait_name is the qualified raw_trait_name parsed as a [syn::Type] (syntax tree type
    /// element).
    trait_name: syn::Type,

    /// context is additional metadata that might be supplied to the [MappingImplBuilder].
    context: T,
}

type MappingImplBuilder<T> = fn(MappingImplOptions<T>) -> MappingImpl;

struct MappingImpl {
    /// trait_name can be used to specialize the trait_name supplied to the [MappingImplBuilder].
    /// This must **always** include the trait_name (or the same value in an equivalent form) as was
    /// supplied to the [MappingImplOptions].
    trait_name: proc_macro2::TokenStream,

    /// bound is the type constraint bound on the VectorField of the Vector type that the trait is
    /// being implemented for (this ensures that arithmetic operations are only implemented on
    /// vectors where their fields also support that arithmetic operation).
    bound: proc_macro2::TokenStream,

    /// output_type is the type that is expected to be returned by the impl. Assign mappings should
    /// simply return [`empty_token_stream!()`] here.
    output_type: proc_macro2::TokenStream,

    /// self_ref is the token that should be used to get a reference to the self instance in the
    /// implementation function.
    self_ref: proc_macro2::TokenStream,

    /// rhs_type is the type for the right-hand side (RHS) operator. For arithmetic mappings, this
    /// is the same as the type that the impl is over.
    rhs_type: proc_macro2::TokenStream,

    /// output type declaration for the function. Assign mappings should simply return
    /// [`empty_token_stream!()`] here.
    output: proc_macro2::TokenStream,

    /// field_mapping is the body of the impl function that actually performs the mapping on each of
    /// the fields in the vector.
    field_mapping: proc_macro2::TokenStream,
}

fn build_mapping_impl_for<T>(
    raw_trait_name: &str,
    name: &Ident,
    context: T,
    builder: MappingImplBuilder<T>,
) -> proc_macro2::TokenStream {
    let is_assign = raw_trait_name.ends_with("Assign");
    let base_trait_name = raw_trait_name
        .split("::")
        .last()
        .unwrap_or_else(|| panic!("Invalid trait name."));

    let trait_name = syn::parse_str::<syn::Type>(raw_trait_name)
        .unwrap_or_else(|_| panic!("Failed to parse Trait name: {}", raw_trait_name));

    let impl_fn_name = syn::parse_str::<syn::Type>(&camel_to_snake_case(base_trait_name))
        .unwrap_or_else(|_| {
            panic!(
                "Failed to derive implementation function name for trait {}.",
                raw_trait_name
            )
        });

    let options = MappingImplOptions {
        name,
        is_assign,
        trait_name,
        context,
    };

    let MappingImpl {
        trait_name,
        bound,
        output_type,
        self_ref,
        rhs_type,
        output,
        field_mapping,
    } = builder(options);

    quote!(
        impl<T: VectorField> #trait_name for #name<T>
        where
            T: #bound
        {
            #output_type

            fn #impl_fn_name(#self_ref, rhs: #rhs_type)#output {
                #field_mapping
            }
        }
    )
}
