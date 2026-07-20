//! Virtual ISA programs.

#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod at_primitives;
pub mod attention;
pub mod bias_partition_broadcast;
pub mod binary_add;
pub mod cluster_chip_shuffle_slice;
pub mod contract_element_types;
pub mod contract_outer_assertions;
pub mod dma;
pub mod fetch_assertions;
pub mod fetch_commit;
pub mod generic_device;
pub mod matmul;
pub mod memory_op;
pub mod mnist;
pub mod param;
pub mod reshape;
pub mod runtime_panic;
pub mod scalar;
pub mod scalar_cast_diag;
pub mod scatter_gather;
pub mod switch_assertions;
pub mod switch_engine;
pub mod tile;
pub mod transformer;
pub mod transpose;
pub mod typelevel_const;
pub mod unevaluated_const;
pub mod vector_engine;
pub mod view;
pub mod vrf_add;
pub mod vrf_add_segmented;
