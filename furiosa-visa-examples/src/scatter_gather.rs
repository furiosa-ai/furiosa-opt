//! Scatter/gather test kernels.

use furiosa_visa_std::prelude::*;

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
