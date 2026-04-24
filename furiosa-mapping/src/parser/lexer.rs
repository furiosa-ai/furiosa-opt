use proc_macro2::{Literal, TokenStream, TokenTree};
use std::collections::VecDeque;
use std::fmt;

/// Mode for the lexer, determining how expressions are tokenized.
#[derive(Debug, Clone, Copy)]
pub enum LexerMode {
    /// Parse mapping expressions (for `m!` macro)
    Mapping,
    /// Parse index expressions with expression capture (for `i!` macro)
    Index,
}

#[derive(Debug, Clone)]
pub enum Token {
    // --- Literals ---
    Symbol(String),
    Nat(usize),
    Expr(proc_macro2::TokenStream),
    Escaped(proc_macro2::TokenStream),

    // --- Operators & Punctuation ---
    Slash,   // /
    Percent, // %
    Eq,      // =
    Hash,    // #
    Comma,   // ,
    Colon,   // :

    // --- Delimiters ---
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]
}

#[derive(Debug, Clone)]
pub enum LexicalError {
    InvalidToken(String),
    UnrecognizedToken(String),
}

impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexicalError::InvalidToken(s) => write!(f, "Invalid token: {}", s),
            LexicalError::UnrecognizedToken(s) => write!(f, "Unrecognized token: {}", s),
        }
    }
}

/// Lexer for tokenizing input TokenStream.
#[derive(Debug)]
pub struct Lexer {
    /// Iterator over TokenTree.
    iter: proc_macro2::token_stream::IntoIter,
    /// Pending TokenTrees to be processed.
    pending: VecDeque<TokenTree>,
    /// Lexer mode (Mapping or Index)
    mode: LexerMode,
    /// Whether a Colon was seen in Index mode
    after_colon: bool,
}

impl Lexer {
    /// Creates a new Lexer from the given TokenStream with the specified mode.
    pub fn new(input: TokenStream, mode: LexerMode) -> Self {
        Lexer {
            iter: input.into_iter(),
            pending: VecDeque::new(),
            mode,
            after_colon: false,
        }
    }

    fn next_tree(&mut self) -> Option<TokenTree> {
        self.pending.pop_front().or_else(|| self.iter.next())
    }

    fn capture_expr(&mut self, first: TokenTree) -> proc_macro2::TokenStream {
        let mut tokens = vec![first];

        loop {
            match self.next_tree() {
                None => break,
                Some(tree) => {
                    if let TokenTree::Punct(ref p) = tree
                        && p.as_char() == ','
                    {
                        // Push comma back for the next token
                        self.pending.push_front(tree);
                        break;
                    }
                    tokens.push(tree);
                }
            }
        }

        tokens.into_iter().collect()
    }
}

impl Iterator for Lexer {
    type Item = Result<(usize, Token, usize), LexicalError>;

    fn next(&mut self) -> Option<Self::Item> {
        let tree = self.next_tree()?;
        let (span_start, span_end) = (0, 0);

        // In Index mode, after we see Colon, capture the expression
        if matches!(self.mode, LexerMode::Index) && self.after_colon {
            self.after_colon = false;
            let expr = self.capture_expr(tree);
            return Some(Ok((span_start, Token::Expr(expr), span_end)));
        }

        let token = match tree {
            TokenTree::Ident(ident) => Token::Symbol(ident.to_string()),

            TokenTree::Literal(lit) => {
                let s = lit.to_string();
                if let Ok(n) = s.parse::<usize>() {
                    Token::Nat(n)
                } else if s == "\")\"" {
                    Token::RParen
                } else if s == "\"]\"" {
                    Token::RBracket
                } else {
                    return Some(Err(LexicalError::InvalidToken(s)));
                }
            }

            TokenTree::Punct(punct) => {
                let ch = punct.as_char();

                match ch {
                    '/' => Token::Slash,
                    '%' => Token::Percent,
                    '#' => Token::Hash,
                    ',' => Token::Comma,
                    ':' => {
                        if matches!(self.mode, LexerMode::Index) {
                            self.after_colon = true;
                        }
                        Token::Colon
                    }
                    '=' => Token::Eq,
                    _ => return Some(Err(LexicalError::UnrecognizedToken(ch.to_string()))),
                }
            }

            TokenTree::Group(group) => {
                let (open_token, close_lit_string) = match group.delimiter() {
                    proc_macro2::Delimiter::Parenthesis => (Token::LParen, ")"),
                    proc_macro2::Delimiter::Bracket => (Token::LBracket, "]"),
                    proc_macro2::Delimiter::Brace => {
                        let inner_stream: proc_macro2::TokenStream = group.stream();
                        return Some(Ok((span_start, Token::Escaped(inner_stream), span_end)));
                    }
                    proc_macro2::Delimiter::None => {
                        let inner_stream: Vec<TokenTree> = group.stream().into_iter().collect();
                        for token in inner_stream.into_iter().rev() {
                            self.pending.push_front(token);
                        }
                        return self.next();
                    }
                };

                let mut fake_close = TokenTree::Literal(Literal::string(close_lit_string));
                fake_close.set_span(group.span_close());
                self.pending.push_front(fake_close);

                let inner_stream: Vec<TokenTree> = group.stream().into_iter().collect();
                for token in inner_stream.into_iter().rev() {
                    self.pending.push_front(token);
                }

                open_token
            }
        };

        Some(Ok((span_start, token, span_end)))
    }
}
