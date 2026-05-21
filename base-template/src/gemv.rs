use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::gemv_kernel::{I, J, gemv_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let mut rng = SmallRng::seed_from_u64(42);
    let matrix = HostTensor::<bf16, m![I, J]>::rand(&mut rng);
    let vector = HostTensor::<bf16, m![J]>::rand(&mut rng);
    let matrix_hbm = matrix.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let vector_hbm = vector.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(gemv_kernel, (&mut ctx, &matrix_hbm, &vector_hbm)).await;
    println!("GEMV: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut ctx = Context::acquire();

        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = HostTensor::<bf16, m![I, J]>::rand(&mut rng);
        let vector = HostTensor::<bf16, m![J]>::rand(&mut rng);

        let matrix_hbm = matrix.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let vector_hbm = vector.to_hbm(&mut ctx.pdma, 1 << 28).await;

        // Reference: y[i] = sum_j matrix[i, j] * vector[j] in f32, rounded to bf16.
        let mat_buf: Vec<bf16> = matrix.to_buf();
        let vec_buf: Vec<bf16> = vector.to_buf();
        let expected: Vec<bf16> = mat_buf
            .chunks(J::SIZE)
            .map(|row| {
                let acc: f32 = row
                    .iter()
                    .zip(&vec_buf)
                    .map(|(&a, &b)| f32::from(a) * f32::from(b))
                    .sum();
                bf16::from_f32(acc)
            })
            .collect();

        let out_hbm = launch(gemv_kernel, (&mut ctx, &matrix_hbm, &vector_hbm)).await;

        let actual: Vec<bf16> = out_hbm.to_host::<m![I]>(&mut ctx.pdma).await.to_buf();
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(a) - f32::from(e)).abs();
            let tol = (0.02 * f32::from(e).abs()).max(0.5);
            assert!(
                diff <= tol,
                "gemv mismatch at i={i}: expected {e:?}, actual {a:?}, diff {diff} > tol {tol}"
            );
        }
    }
}
