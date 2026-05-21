use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::gemm_kernel::{I, J, K, gemm_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let mut rng = SmallRng::seed_from_u64(42);
    let a = HostTensor::<bf16, m![I, K]>::rand(&mut rng);
    let b = HostTensor::<bf16, m![J, K]>::rand(&mut rng);
    let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
    println!("GEMM: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut ctx = Context::acquire();

        let mut rng = SmallRng::seed_from_u64(42);
        let a = HostTensor::<bf16, m![I, K]>::rand(&mut rng);
        let b = HostTensor::<bf16, m![J, K]>::rand(&mut rng);

        let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;

        // Reference: C[i, j] = sum_k A[i, k] * B[j, k] in f32, rounded to bf16.
        let a_buf: Vec<bf16> = a.to_buf();
        let b_buf: Vec<bf16> = b.to_buf();
        let expected: Vec<bf16> = a_buf
            .chunks(K::SIZE)
            .flat_map(|a_row| {
                b_buf.chunks(K::SIZE).map(move |b_row| {
                    let acc: f32 = a_row
                        .iter()
                        .zip(b_row)
                        .map(|(&a, &b)| f32::from(a) * f32::from(b))
                        .sum();
                    bf16::from_f32(acc)
                })
            })
            .collect();

        let out_hbm = launch(gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;

        let actual: Vec<bf16> = out_hbm.to_host::<m![I, J]>(&mut ctx.pdma).await.to_buf();
        for (idx, (&e, &av)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(av) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "gemm mismatch at idx={idx}: expected {e:?}, actual {av:?}");
        }
    }
}
