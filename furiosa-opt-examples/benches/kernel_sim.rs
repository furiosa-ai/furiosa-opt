//! Whole-kernel simulation wall-clock harness (NOT criterion): runs a representative decode-path
//! kernel end to end on the CPU backend (`BufStorage`), timed, so it can
//! be driven under `perf record` / `top -H` to profile the compute critical path.
//!
//! ```sh
//! cargo build -p furiosa-opt-examples --bench kernel_sim --release
//! perf record -g --call-graph dwarf -- \
//!   ./target/release/deps/kernel_sim-<hash> matmul 3
//! ```
//!
//! Args: `<kernel> [iters]`. `<kernel>` ∈ {matmul}. Prints per-iter wall time.
//! Deliberately not a criterion bench: a plain `perf`-driveable entry point, not a statistical bench.

use std::time::Instant;

use furiosa_opt_examples::matmul::matmul_4096;
use furiosa_opt_std::prelude::*;

async fn run_matmul(device: &mut Device) {
    use matmul_4096::{A, B};
    let lhs = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let rhs = HostTensor::<i8, m![B]>::zero()
        .to_hbm::<m![1], m![B]>(&mut device.pdma)
        .await
        .unwrap();
    let _ = launch(matmul_4096::matmul_4096, (device, &lhs, &rhs)).await.unwrap();
}

/// The kernel this harness can drive. Parsed once from argv so the dispatch `match` is exhaustive and
/// the name <-> runner table lives in one place.
enum Kernel {
    Matmul,
}

impl std::str::FromStr for Kernel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "matmul" => Ok(Self::Matmul),
            other => Err(format!("unknown kernel {other:?}; use matmul")),
        }
    }
}

impl Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
        }
    }

    async fn run(&self, device: &mut Device) {
        match self {
            Self::Matmul => run_matmul(device).await,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kernel: Kernel = args
        .get(1)
        .map_or(Ok(Kernel::Matmul), |s| s.parse())
        .unwrap_or_else(|e| panic!("{e}"));
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    // The kernels are `#[device(chip = 1)]` with the `#[device]` default `pe = 8`, so this harness
    // runs the 1chip/8PE topology. The CPU backend derives storage layout from the tensor
    // `m!` shapes alone and never reads a global NPU config, so no config pin is needed here.

    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(async {
        for i in 0..iters {
            let mut device = Device::new(matmul_4096::matmul_4096.topology()).unwrap();
            let t = Instant::now();
            kernel.run(&mut device).await;
            let dt = t.elapsed();
            println!("{} iter {i}: {:.3} s", kernel.name(), dt.as_secs_f64());
        }
    });
}
