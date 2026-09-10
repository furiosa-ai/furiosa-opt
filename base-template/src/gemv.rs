use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::gemv_kernel::{I, J, gemv_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(gemv_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let matrix = HostTensor::<bf16, m![I, J]>::rand(&mut rng);
    let vector = HostTensor::<bf16, m![J]>::rand(&mut rng);
    let matrix_hbm = matrix.to_hbm(&mut device.pdma).await?;
    let vector_hbm = vector.to_hbm(&mut device.pdma).await?;
    let _out_hbm = launch(gemv_kernel, (&mut device, &matrix_hbm, &vector_hbm)).await?;
    println!("GEMV: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(gemv_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = HostTensor::<bf16, m![I, J]>::rand(&mut rng);
        let vector = HostTensor::<bf16, m![J]>::rand(&mut rng);

        let matrix_hbm = matrix.to_hbm(&mut device.pdma).await.unwrap();
        let vector_hbm = vector.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: y[i] = sum_j matrix[i, j] * vector[j] in f32, rounded to bf16.
        let mat_buf: Vec<bf16> = matrix.into_vec();
        let vec_buf: Vec<bf16> = vector.into_vec();
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

        let out_hbm = launch(gemv_kernel, (&mut device, &matrix_hbm, &vector_hbm)).await.unwrap();

        let actual: Vec<bf16> = out_hbm.to_host::<m![I]>(&mut device.pdma).await.unwrap().into_vec();
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
