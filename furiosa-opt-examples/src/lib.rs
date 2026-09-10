//! Virtual ISA programs.
//!
//! Every concrete kernel outside [`negative`] compiles end-to-end to an EDF; [`generic`]'s
//! fns compile from a linked root's launches instead, and [`negative`] holds the rejection
//! fixtures, which must fail. Legal programs the compiler cannot lower yet are not here
//! at all: they live in the private `npu-opt-examples` crate under `unsupported`.

#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod bias_partition_broadcast;
pub mod binary_add;
pub mod chip_reduce;
pub mod cluster_chip_shuffle_slice;
pub mod cluster_reduce;
pub mod commit_view_tile;
pub mod contract_element_types;
pub mod contract_outer_assertions;
pub mod dma;
pub mod double_buffering;
pub mod f4_indexing;
pub mod fetch_assertions;
pub mod fetch_commit;
pub mod fetch_lift;
pub mod fetch_table_lookup;
pub mod generic;
pub mod host_tile_view;
pub mod matmul;
pub mod memory_op;
pub mod memset;
pub mod mnist;
pub mod negative;
pub mod padded_axis_tile;
pub mod param;
pub mod pe_count;
pub mod reshape;
pub mod runtime_if;
pub mod runtime_if_scalar;
pub mod scalar;
pub mod scatter_gather;
pub mod stotrf_table_lookup;
pub mod switch_assertions;
pub mod switch_engine;
pub mod tile;
pub mod to_vrf_assertions;
pub mod transformer;
pub mod transpose;
pub mod typelevel_const;
pub mod unevaluated_const;
pub mod unroll_loop;
pub mod vector_engine;
pub mod view;
pub mod vrf_add_segmented;
pub mod vrf_operand;
