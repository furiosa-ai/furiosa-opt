//! Whole-kernel simulation wall-clock harness (NOT criterion): runs a representative decode-path
//! kernel end to end on the active backend (default `emulation` → `BufStorage`), timed, so it can
//! be driven under `perf record` / `top -H` to profile the compute critical path.
//!
//! ```sh
//! cargo build -p furiosa-opt-examples --bench kernel_sim --release
//! perf record -g --call-graph dwarf -- \
//!   ./target/release/deps/kernel_sim-<hash> attention 3
//! ```
//!
//! Args: `<kernel> [iters]`. `<kernel>` ∈ {attention, decoder, matmul}. Prints per-iter wall time.
//! Deliberately not a criterion bench: a plain `perf`-driveable entry point, not a statistical bench.

use std::time::Instant;

use furiosa_opt_examples::matmul::matmul_4096;
use furiosa_opt_examples::transformer::axes::{C_kvcache, D, H, K, M as M_mlp, N, P, Q, R, S_decode, S_prefill, T, V};
use furiosa_opt_examples::transformer::{attention, decoder};
use furiosa_opt_std::prelude::*;

async fn run_matmul(ctx: &mut Context) {
    use matmul_4096::{A, B};
    let lhs = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let rhs = HostTensor::<i8, m![B]>::zero()
        .to_hbm::<m![1], m![B]>(&mut ctx.pdma)
        .await;
    let _ = launch(matmul_4096::matmul_4096, (ctx, &lhs, &rhs)).await;
}

async fn run_attention(ctx: &mut Context) {
    let attn_q = HostTensor::<bf16, m![S_prefill, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let attn_k = HostTensor::<bf16, m![T, K]>::zero().to_hbm(&mut ctx.pdma).await;
    let attn_v = HostTensor::<bf16, m![T, K]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_scatter_index = HostTensor::<i32, m![1, T]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_scatter_index = HostTensor::<i32, m![1, T]>::zero().to_hbm(&mut ctx.pdma).await;
    let attention_mask = HostTensor::<i32, m![1, S_prefill, T]>::zero()
        .to_hbm(&mut ctx.pdma)
        .await;

    let mut out_attn = unsafe { HbmTensor::<bf16, m![1], m![S_prefill, H]>::from_addr(0x288800) };
    let mut out_k_cache = unsafe { HbmTensor::<bf16, m![1], m![C_kvcache, 1, N, D]>::from_addr(0x448800) };
    let mut out_v_cache = unsafe { HbmTensor::<bf16, m![1], m![C_kvcache, 1, N, D]>::from_addr(0x40000) };

    launch(
        attention::forward,
        (
            ctx,
            &attn_q,
            &attn_k,
            &attn_v,
            &k_scatter_index,
            &v_scatter_index,
            &attention_mask,
            &mut out_attn,
            &mut out_k_cache,
            &mut out_v_cache,
        ),
    )
    .await;
}

async fn run_decoder(ctx: &mut Context) {
    let matmul_score_v = HostTensor::<bf16, m![S_decode, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let hidden_states = HostTensor::<bf16, m![S_decode, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let o_proj_weight = HostTensor::<bf16, m![H, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight_0 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let gate_weight = HostTensor::<bf16, m![M_mlp, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let up_weight = HostTensor::<bf16, m![M_mlp, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let down_weight = HostTensor::<bf16, m![H, M_mlp]>::zero().to_hbm(&mut ctx.pdma).await;
    let norm_weight_1 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_weight = HostTensor::<bf16, m![Q, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_bias = HostTensor::<bf16, m![Q]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_weight = HostTensor::<bf16, m![K, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_bias = HostTensor::<bf16, m![K]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_weight = HostTensor::<bf16, m![V, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_bias = HostTensor::<bf16, m![V]>::zero().to_hbm(&mut ctx.pdma).await;
    let rope_table = HostTensor::<bf16, m![P, D / 2, R, R]>::zero()
        .to_hbm(&mut ctx.pdma)
        .await;
    let position_ids = HostTensor::<i32, m![S_decode]>::zero().to_hbm(&mut ctx.pdma).await;

    let mut out_q = unsafe { HbmTensor::<bf16, m![1], m![S_decode, Q]>::from_addr(0x20036000) };
    let mut out_k = unsafe { HbmTensor::<bf16, m![1], m![S_decode, K]>::from_addr(0x20436000) };
    let mut out_v = unsafe { HbmTensor::<bf16, m![1], m![S_decode, V]>::from_addr(0x20536000) };
    let mut out_hidden = unsafe { HbmTensor::<bf16, m![1], m![S_decode, H]>::from_addr(0x20636000) };

    launch(
        decoder::forward,
        (
            ctx,
            &matmul_score_v,
            &o_proj_weight,
            &hidden_states,
            &norm_weight_0,
            &gate_weight,
            &up_weight,
            &down_weight,
            &norm_weight_1,
            &q_weight,
            &q_bias,
            &k_weight,
            &k_bias,
            &v_weight,
            &v_bias,
            &rope_table,
            &position_ids,
            &mut out_q,
            &mut out_k,
            &mut out_v,
            &mut out_hidden,
        ),
    )
    .await;
}

/// The kernel this harness can drive. Parsed once from argv so the dispatch `match` is exhaustive and
/// the name <-> runner table lives in one place.
enum Kernel {
    Attention,
    Decoder,
    Matmul,
}

impl std::str::FromStr for Kernel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "attention" => Ok(Self::Attention),
            "decoder" => Ok(Self::Decoder),
            "matmul" => Ok(Self::Matmul),
            other => Err(format!("unknown kernel {other:?}; use attention|decoder|matmul")),
        }
    }
}

impl Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::Decoder => "decoder",
            Self::Matmul => "matmul",
        }
    }

    async fn run(&self, ctx: &mut Context) {
        match self {
            Self::Attention => run_attention(ctx).await,
            Self::Decoder => run_decoder(ctx).await,
            Self::Matmul => run_matmul(ctx).await,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kernel: Kernel = args
        .get(1)
        .map_or(Ok(Kernel::Attention), |s| s.parse())
        .unwrap_or_else(|e| panic!("{e}"));
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    // The kernels are `#[device(chip = 1)]` with the `#[device]` default `pe = 8`, so this harness
    // runs the 1chip/8PE topology. The emulation backend derives storage layout from the tensor
    // `m!` shapes alone and never reads a global NPU config, so no config pin is needed here.

    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(async {
        for i in 0..iters {
            let mut ctx = Context::acquire();
            let t = Instant::now();
            kernel.run(&mut ctx).await;
            let dt = t.elapsed();
            println!("{} iter {i}: {:.3} s", kernel.name(), dt.as_secs_f64());
        }
    });
}
