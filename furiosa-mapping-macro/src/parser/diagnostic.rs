use super::ParseError;
use super::lexer::{LexerMode, LexicalError};
use lalrpop_util::ParseError as LalrpopError;

pub(super) fn from_parse_error(error: ParseError, mode: LexerMode) -> syn::Error {
    match error {
        LalrpopError::InvalidToken { location } => {
            syn::Error::new(location.span(), format!("invalid {} syntax", mode.name()))
        }
        LalrpopError::UnrecognizedEof { location, expected: _ }
            if matches!(mode, LexerMode::Mapping) && location.is_start() =>
        {
            syn::Error::new(
                location.span(),
                "empty mapping expression `m![]` is not supported; use `m![1]` for a unit mapping",
            )
        }
        LalrpopError::UnrecognizedEof { location, expected } => syn::Error::new(
            location.span(),
            format!(
                "unexpected end of {} expression; expected {}",
                mode.name(),
                format_expected(&expected)
            ),
        ),
        LalrpopError::UnrecognizedToken {
            token: (location, token, _),
            expected,
        } => syn::Error::new(
            location.span(),
            format!("unexpected token {token}; expected {}", format_expected(&expected)),
        ),
        LalrpopError::ExtraToken {
            token: (location, token, _),
        } => syn::Error::new(location.span(), format!("unexpected token {token}")),
        LalrpopError::User {
            error: LexicalError::InvalidToken { token, span },
        } => syn::Error::new(span, format!("invalid token `{token}`")),
        LalrpopError::User {
            error: LexicalError::UnrecognizedToken { token, span },
        } => syn::Error::new(
            span,
            format!(
                "unsupported token `{token}` in {} syntax; use mapping operators or wrap a Rust expression in braces",
                mode.name()
            ),
        ),
    }
}

fn format_expected(expected: &[String]) -> String {
    let expected = expected.iter().map(|token| display_token(token)).collect::<Vec<_>>();
    match expected.as_slice() {
        [] => "valid syntax".to_string(),
        [only] => only.to_string(),
        [left, right] => format!("{left} or {right}"),
        [prefix @ .., last] => format!("{}, or {last}", prefix.join(", ")),
    }
}

fn display_token(token: &str) -> &str {
    let token = token.trim_matches('"');
    match token {
        "Symbol" => "an axis name",
        "Nat" => "an integer",
        "Expr" => "an index expression",
        "Escaped" => "a braced Rust expression",
        "Slash" => "`/`",
        "Percent" => "`%`",
        "Eq" => "`=`",
        "Hash" => "`#`",
        "HashFill" => "a padding marker",
        "Comma" => "`,`",
        "Colon" => "`:`",
        "LParen" => "`(`",
        "RParen" => "`)`",
        "LBracket" => "`[`",
        "RBracket" => "`]`",
        _ => token,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{parse_index, parse_mapping};

    fn tokens(input: &str) -> proc_macro2::TokenStream {
        input.parse().expect("test input must be tokenizable")
    }

    fn assert_mapping_error(input: &str, span: std::ops::Range<usize>, message: &str) {
        let error = parse_mapping(tokens(input)).expect_err("mapping must be rejected");
        assert_eq!(error.to_string(), message);
        assert_eq!(error.span().byte_range(), span);
    }

    fn assert_index_error(input: &str, span: std::ops::Range<usize>, message: &str) {
        let error = parse_index(tokens(input)).expect_err("index must be rejected");
        assert_eq!(error.to_string(), message);
        assert_eq!(error.span().byte_range(), span);
    }

    #[test]
    fn empty_mapping_suggests_unit_mapping() {
        assert_mapping_error(
            "",
            0..0,
            "empty mapping expression `m![]` is not supported; use `m![1]` for a unit mapping",
        );
    }

    #[test]
    fn malformed_mapping_uses_dsl_vocabulary() {
        assert_mapping_error(
            "A,, B",
            2..3,
            "unexpected token `,`; expected an axis name, an integer, a braced Rust expression, `(`, or `[`",
        );
    }

    #[test]
    fn incomplete_operator_uses_dsl_vocabulary() {
        assert_mapping_error(
            "A /",
            2..3,
            "unexpected end of mapping expression; expected an integer or a braced Rust expression",
        );
    }

    #[test]
    fn lexical_error_suggests_valid_syntax() {
        assert_mapping_error(
            "A + B",
            2..3,
            "unsupported token `+` in mapping syntax; use mapping operators or wrap a Rust expression in braces",
        );
    }

    #[test]
    fn malformed_index_uses_index_diagnostic() {
        assert_index_error(
            "A,, B: 0",
            2..3,
            "unexpected token `,`; expected an axis name, an integer, a braced Rust expression, `:`, `(`, or `[`",
        );
    }

    #[test]
    fn incomplete_index_uses_index_diagnostic() {
        assert_index_error(
            "A:",
            1..2,
            "unexpected end of index expression; expected an index expression",
        );
    }
}
