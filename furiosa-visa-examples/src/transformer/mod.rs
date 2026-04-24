//! Qwen 2.5 0.5B composable transformer kernels (4PE, W16A16KV16, bf16).
//!
//! Each module maps to a composable kernel stage:
//!
//! | Module      | Layer range                                               |
//! |-------------|-----------------------------------------------------------|
//! | `embedding` | `embedding..transformer_0:qkv_projection`                 |
//! | `attention` | `transformer_N:attention` (prefill, S=1024)               |
//! | `decoder`   | `transformer_N:output_projection..transformer_N+1:qkv`    |
//! | `head`      | `transformer_-1:output_projection..output` (+ LM head)    |
//!
//! Shared building blocks (O-proj, MLP, residual+norm, RoPE) live in `common/`.

pub mod attention;
pub mod axes;
mod common;
pub mod decoder;
pub mod embedding;
pub mod head;
