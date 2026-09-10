use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::dot_product_kernel::{A, dot_product_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(dot_product_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await?;
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await?;
    let _out_hbm = device.launch(dot_product_kernel, (&lhs_hbm, &rhs_hbm)).await?;
    println!("Dot Product: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(dot_product_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let lhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
        let rhs = HostTensor::<bf16, m![A]>::rand(&mut rng);

        let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
        let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: sum_i lhs[i] * rhs[i] in f32, then round to bf16.
        let lhs_buf: Vec<bf16> = lhs.into_vec();
        let rhs_buf: Vec<bf16> = rhs.into_vec();
        let expected_f32: f32 = lhs_buf
            .iter()
            .zip(&rhs_buf)
            .map(|(&a, &b)| f32::from(a) * f32::from(b))
            .sum();
        let expected = bf16::from_f32(expected_f32);

        let out_hbm = device.launch(dot_product_kernel, (&lhs_hbm, &rhs_hbm)).await.unwrap();

        let actual_buf: Vec<bf16> = out_hbm.to_host::<m![1]>(&mut device.pdma).await.unwrap().into_vec();
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
