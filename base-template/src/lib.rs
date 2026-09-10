#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod kernel;
