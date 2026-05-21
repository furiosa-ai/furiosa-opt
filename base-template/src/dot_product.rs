use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::dot_product_kernel::{A, dot_product_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Dot Product: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut ctx = Context::acquire();

        let mut rng = SmallRng::seed_from_u64(42);
        let lhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
        let rhs = HostTensor::<bf16, m![A]>::rand(&mut rng);

        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

        // Reference: sum_i lhs[i] * rhs[i] in f32, then round to bf16.
        let lhs_buf: Vec<bf16> = lhs.to_buf();
        let rhs_buf: Vec<bf16> = rhs.to_buf();
        let expected_f32: f32 = lhs_buf
            .iter()
            .zip(&rhs_buf)
            .map(|(&a, &b)| f32::from(a) * f32::from(b))
            .sum();
        let expected = bf16::from_f32(expected_f32);

        let out_hbm = launch(dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;

        let actual_buf: Vec<bf16> = out_hbm.to_host::<m![1]>(&mut ctx.pdma).await.to_buf();
        if let Some(&actual) = actual_buf.first() {
            let diff = (f32::from(actual) - f32::from(expected)).abs();
            let tol = (0.02 * f32::from(expected).abs()).max(0.5);
            assert!(
                diff <= tol,
                "dot_product mismatch: expected {expected:?}, actual {actual:?}, diff {diff} > tol {tol}"
            );
        }
    }
}
