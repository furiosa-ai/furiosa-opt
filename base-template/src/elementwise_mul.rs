use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::elementwise_mul_kernel::{A, elementwise_mul_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(elementwise_mul_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Elementwise Mul: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut ctx = Context::acquire();

        let mut rng = SmallRng::seed_from_u64(42);
        let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
        let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

        // Reference: out[i] = lhs[i] * rhs[i].
        let lhs_buf: Vec<i32> = lhs.to_buf();
        let rhs_buf: Vec<i32> = rhs.to_buf();
        let expected: Vec<i32> = lhs_buf.iter().zip(&rhs_buf).map(|(&a, &b)| a.wrapping_mul(b)).collect();

        let out_hbm = launch(elementwise_mul_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;

        let actual: Vec<i32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(e, a, "elementwise_mul mismatch at i={i}: expected {e}, actual {a}");
        }
    }
}
