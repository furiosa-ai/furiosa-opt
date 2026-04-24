//! Virtual ISA programs.

#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(tcp)]

pub mod aligner_assertions;
pub mod attention;
pub mod binary_add;
pub mod cluster_chip_shuffle_slice;
pub mod fetch_assertions;
pub mod fetch_commit;
pub mod matmul;
pub mod mnist;
pub mod reshape;
pub mod scatter_gather;
pub mod switch_assertions;
pub mod tile;
pub mod transformer;
pub mod transpose;
pub mod vector_engine;
pub mod view;
pub mod vrf_add;
