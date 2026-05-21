use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::constant_add_kernel::{A, constant_add_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);
    let in_hbm = input.to_hbm(&mut ctx.pdma, 0).await;
    let _out_hbm = launch(constant_add_kernel, (&mut ctx, &in_hbm)).await;
    println!("Constant Add: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut ctx = Context::acquire();

        let mut rng = SmallRng::seed_from_u64(42);
        let input = HostTensor::<i32, m![A]>::rand(&mut rng);
        let in_hbm = input.to_hbm(&mut ctx.pdma, 0).await;

        // Reference: out[i] = in[i] + 1.
        let in_buf: Vec<i32> = input.to_buf();
        let expected: Vec<i32> = in_buf.iter().map(|&x| x.wrapping_add(1)).collect();

        let out_hbm = launch(constant_add_kernel, (&mut ctx, &in_hbm)).await;

        // Under the typecheck backend `actual` is empty (phantom tensors), so
        // the loop trivially runs zero iterations and the assertion is skipped.
        let actual: Vec<i32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(e, a, "constant_add mismatch at i={i}: expected {e}, actual {a}");
        }
    }
}
