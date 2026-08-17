//! Storage-loop microbenchmarks for the host tensor backends.
//!
//! Each bench drives one `Tensor` op straight into the active backend's storage hot loop
//! (`map` / `zip_with` / `reduce` / `transpose`, plus the fused `contraction`), with no scheduler or
//! DMA in the path. The op code is backend-agnostic, so it measures whichever backend is compiled in:
//! the CPU backend (`BufStorage`).
//!
//! ```sh
//! cargo bench -p furiosa-opt-examples --bench storage
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Element-wise / reduce / transpose run on a 512-square; contraction (and its breakdown) on a
// 128-cube (2.1M cells). Sized so the whole suite runs in ~10s under the interpreter.
axes![A = 512, B = 512, M = 128, K = 128, N = 128];

// The `out_size ≫ contracted` corner: a wide output (2^20 cells) reduced over a thin (4-cell) axis.
// The old per-worker-accumulator `BufStorage::reduce` allocated + combined a `vec![identity; out_size]`
// per rayon leaf here (`O(out_size × leaves)`), which could run SLOWER than serial; the per-output-cell
// rewrite writes each cell once. This bench pins that the parallel reduce is not slower than serial in
// the corner (a plain cargo run exercises `BufStorage`).
axes![Wide = 1048576, Thin = 4];

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

fn bench_map(c: &mut Criterion) {
    let t = Tensor::<f32, m![A, B]>::rand(&mut rng());
    c.bench_function("map 512x512", |b| b.iter(|| black_box(&t).map(|x| x * 2.0)));
}

fn bench_zip(c: &mut Criterion) {
    let lhs = Tensor::<f32, m![A, B]>::rand(&mut rng());
    let rhs = Tensor::<f32, m![A, B]>::rand(&mut rng());
    c.bench_function("zip_with 512x512", |b| {
        b.iter(|| black_box(&lhs).zip_with(black_box(&rhs), |x, y| x + y))
    });
}

fn bench_reduce(c: &mut Criterion) {
    let t = Tensor::<f32, m![A, B]>::rand(&mut rng());
    c.bench_function("reduce_add 512x512->512", |b| {
        b.iter(|| black_box(&t).reduce_add::<m![A]>())
    });
}

fn bench_reduce_wide_thin(c: &mut Criterion) {
    let t = Tensor::<f32, m![Wide, Thin]>::rand(&mut rng());
    c.bench_function("reduce_add 2^20x4->2^20 (wide out, thin contracted)", |b| {
        b.iter(|| black_box(&t).reduce_add::<m![Wide]>())
    });
}

fn bench_transpose(c: &mut Criterion) {
    let t = Tensor::<f32, m![A, B]>::rand(&mut rng());
    c.bench_function("transpose 512x512", |b| {
        b.iter(|| black_box(&t).transpose::<m![B, A]>(false))
    });
}

fn bench_contraction(c: &mut Criterion) {
    let lhs = Tensor::<f32, m![M, K]>::rand(&mut rng());
    let rhs = Tensor::<f32, m![K, N]>::rand(&mut rng());
    c.bench_function("contraction 128x128x128", |b| {
        b.iter(|| {
            Tensor::<f32, m![M, N]>::contraction_prewidened::<m![M, K, N], _, _>(black_box(&lhs), black_box(&rhs))
        })
    });
}

// The composed alternative to the fused `contraction`, at the 128^3 = 2.1M-cell working set: the two
// broadcast transposes, the element-wise zip, and the reduce it would otherwise chain. Shows what the
// fused op avoids materializing, and which sub-op would dominate.
fn bench_contraction_parts(c: &mut Criterion) {
    let lhs = Tensor::<f32, m![M, K]>::rand(&mut rng());
    let rhs = Tensor::<f32, m![K, N]>::rand(&mut rng());
    c.bench_function("part: transpose_bcast MK->MKN", |b| {
        b.iter(|| black_box(&lhs).transpose::<m![M, K, N]>(true))
    });
    c.bench_function("part: transpose_bcast KN->MKN", |b| {
        b.iter(|| black_box(&rhs).transpose::<m![M, K, N]>(true))
    });
    let lhs_b = lhs.transpose::<m![M, K, N]>(true);
    let rhs_b = rhs.transpose::<m![M, K, N]>(true);
    c.bench_function("part: zip 128^3", |b| {
        b.iter(|| black_box(&lhs_b).zip_with(black_box(&rhs_b), |a, b| a * b))
    });
    let prod = lhs_b.zip_with(&rhs_b, |a, b| a * b);
    c.bench_function("part: reduce 128^3->128^2", |b| {
        b.iter(|| black_box(&prod).reduce_add::<m![M, N]>())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = bench_map, bench_zip, bench_reduce, bench_reduce_wide_thin, bench_transpose, bench_contraction, bench_contraction_parts
}
criterion_main!(benches);
