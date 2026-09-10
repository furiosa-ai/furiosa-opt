//! An array index must resolve to one value: a constant or a loop index, never a branch.

use furiosa_opt_std::prelude::*;

axes![A = 64];

type Chip = m![1];

#[device(chip = 1, pe = 1)]
pub fn branched(device: &mut Device, first: &HbmTensor<bf16, Chip, m![A]>, second: &HbmTensor<bf16, Chip, m![A]>) {
    let inputs = [first, second];
    for i in 0..2 {
        let _: DmTensor<bf16, Chip, m![1], m![1 # 64], m![A]> =
            inputs[if i == 0 { 0 } else { 1 }].to_dm(&mut device.tdma);
    }
}
