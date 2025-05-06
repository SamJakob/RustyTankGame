/// camel_to_snake_case converts a string from camelCase or UpperCamelCase to snake_case.
///
/// For example:
///
///     assert_eq!(camel_to_snake_case("UpperCamelCase"), "upper_camel_case");
///     assert_eq!(camel_to_snake_case("lowerCamelCase"), "lower_camel_case");
pub(crate) fn camel_to_snake_case(input: &str) -> String {
    input
        .chars()
        .enumerate()
        .map(|(i, x)| match x {
            'A'..='Z' => {
                if i == 0 {
                    format!("{}", x.to_lowercase())
                } else {
                    format!("_{}", x.to_lowercase())
                }
            }
            _ => x.into(),
        })
        .collect::<String>()
}
