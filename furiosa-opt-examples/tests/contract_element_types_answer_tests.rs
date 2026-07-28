//! Answer-key tests for the per-element-type contraction kernels: run the
//! `#[device]` body under the default (simulation) backend and compare against
//! an independent hand-computed true-math oracle. Unlike the `npu-visa-test`
//! `compare_edf` tests (VISA-sim vs LIR/EDF), this pins whether the VISA/device
//! simulation itself is correct, localizing a divergence to either the
//! eDSL/example or the LIR lowering.

use furiosa_opt_examples::contract_element_types::{A, K8, R, i8_contract};
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Chip = m![1];

// Same seeded input as `npu-visa-test`'s `test_i8_contract`, so a pass here
// (VISA-sim == true math) means the sub-mac contraction bug is in the LIR
// lowering, not the eDSL/simulation.
#[tokio::test]
async fn answer_i8_contract() {
    const AN: usize = 8;
    const RN: usize = 8;
    const KN: usize = 32;

    let mut rng = SmallRng::seed_from_u64(42);
    let input_vals: Vec<i8> = (0..<m![A, K8]>::SIZE).map(|_| rng.random_range(-8..8)).collect();
    let trf_vals: Vec<i8> = (0..<m![R, K8]>::SIZE).map(|_| rng.random_range(-8..8)).collect();

    // Hand-computed i32 oracle: out[a, r] = sum_k input[a, k] * trf[r, k],
    // laid out a-major r-minor to match `m![A, R # 8]`.
    let mut expected = vec![0i32; AN * RN];
    for a in 0..AN {
        for r in 0..RN {
            let mut s = 0i32;
            for k in 0..KN {
                s += input_vals[a * KN + k] as i32 * trf_vals[r * KN + k] as i32;
            }
            expected[a * RN + r] = s;
        }
    }

    let mut ctx = Context::acquire();
    let input = HostTensor::<i8, m![A, K8]>::from_vec(input_vals);
    let trf = HostTensor::<i8, m![R, K8]>::from_vec(trf_vals);
    let input_hbm = input.to_hbm::<Chip, m![A, K8]>(&mut ctx.pdma).await;
    let trf_hbm = trf.to_hbm::<Chip, m![R, K8]>(&mut ctx.pdma).await;

    let out = launch(i8_contract, (&mut *ctx, &input_hbm, &trf_hbm)).await;

    assert_eq!(
        expected,
        out.to_host::<m![A, R # 8]>(&mut ctx.pdma).await.into_vec(),
        "true-math oracle (left) vs VISA-sim (right)"
    );
}
