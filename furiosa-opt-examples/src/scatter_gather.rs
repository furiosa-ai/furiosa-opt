//! Scatter/gather test kernels.
//!
//! `K = 512` keys scatter into a `C = 612` cache (sparse: `C > K`).
//! The non-power-of-2, larger-than-`K` cache stresses non-aligned scatter coverage.
//!
//! Note: in the `renegade-8pe` config (`num_slices = 256`), the gather kernel fails
//! the `dma_gather` partitioning check `power_of_two_aligned(C/2) == num_slices`
//! because `C / 2 = 306` rounds up to `512`, not `256`. A test environment with a
//! larger `num_slices` (or a smaller cache) is required to compile the gather kernel
//! through `cargo furiosa-opt`. The corresponding `#[ignore]` markers in
//! `crates/npu-visa-test/tests/scatter_gather.rs` document a separate LIR-runtime
//! limitation (in-place `&mut output` not supported by the LIR executor).

use furiosa_opt_std::prelude::*;

axes![
    K = 512, // Scatter key
    D = 128, // Payload per key
    C = 612  // Cache length
];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Scatter values into cache at index positions.
#[device(chip = 1)]
pub fn scatter_minimal(
    ctx: &mut Context,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, Cluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut ctx.tdma, 0x0);

    data_dm.dma_scatter::<m![K], _, _>(index, output, true);
}

/// Gather rows from `table` at positions given by `index` into a fresh HBM output.
///
/// Inverse of [`scatter_minimal`]. Produces a `DmTensor` via [`HbmTensor::dma_gather`],
/// then writes it back to HBM via [`DmTensor::to_hbm`]. Returns the result by value
/// because `dma_gather` does not write into an existing `&mut HbmTensor`; the natural
/// write-back primitive (`to_hbm`) returns a freshly-addressed `HbmTensor`.
#[device(chip = 1)]
pub fn gather_minimal(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![C]>,
) -> HbmTensor<bf16, Chip, m![C, D]> {
    let values_dm: DmTensor<bf16, Chip, Cluster, m![C / 2], m![C % 2, D]> = table.dma_gather(index, 0x0, true);

    values_dm.to_hbm(&mut ctx.tdma, 0x1000)
}
