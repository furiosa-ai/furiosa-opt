//! Gather and scatter kernels sized for each CI device topology.

use furiosa_opt_std::prelude::*;

axes![
    TK = 512,  // table rows every gather reads and every scatter writes into
    TD = 64,   // payload elements per row (128 B, lane-aligned)
    TC = 1024, // scatter cache length, wider than any config's update count
    // Row counts, one per topology: (chips x) clusters x slices x 2.
    R1 = 128, // 1 PE
    R2 = 256, // 2 PE
    R4 = 512, // 4 PE
    R8 = 512, // 8 PE, and each chip of the 2- and 4-chip grids
    // Named chip axes keep each chip's rows distinct; `m![2]` would broadcast.
    C2 = 2,
    C4 = 4
];

type OneCluster = m![1];
type PaddedCluster = m![1 # 2];
type OneChip = m![1];
type TwoChips = m![C2];
type FourChips = m![C4];

/// Gathers `R1` rows at a 1-PE device (1 cluster x 64 slices).
#[device(chip = 1, pe = 1)]
pub fn one_pe_gather(
    device: &mut Device,
    table: &HbmTensor<bf16, OneChip, m![TK, TD]>,
    index: &HbmTensor<i32, OneChip, m![R1]>,
) -> HbmTensor<bf16, OneChip, m![R1, TD]> {
    let values: DmTensor<bf16, OneChip, OneCluster, m![R1 / 2], m![R1 % 2, TD]> = table
        .gather::<m![TK], m![TD]>()
        .by_byte_offsets::<m![R1]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Scatters `R1` rows at a 1-PE device.
#[device(chip = 1, pe = 1)]
pub fn one_pe_scatter(
    device: &mut Device,
    data: &HbmTensor<bf16, OneChip, m![R1, TD]>,
    index: &HbmTensor<i32, OneChip, m![R1]>,
    output: &mut HbmTensor<bf16, OneChip, m![TC, TD]>,
) {
    let data_dm: DmTensor<bf16, OneChip, OneCluster, m![R1 / 2], m![R1 % 2, TD]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![TC], m![TD]>()
        .by_byte_offsets::<m![R1]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers `R2` rows at a 2-PE device (1 cluster x 128 slices).
#[device(chip = 1, pe = 2)]
pub fn two_pe_gather(
    device: &mut Device,
    table: &HbmTensor<bf16, OneChip, m![TK, TD]>,
    index: &HbmTensor<i32, OneChip, m![R2]>,
) -> HbmTensor<bf16, OneChip, m![R2, TD]> {
    let values: DmTensor<bf16, OneChip, OneCluster, m![R2 / 2], m![R2 % 2, TD]> = table
        .gather::<m![TK], m![TD]>()
        .by_byte_offsets::<m![R2]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Scatters `R2` rows at a 2-PE device.
#[device(chip = 1, pe = 2)]
pub fn two_pe_scatter(
    device: &mut Device,
    data: &HbmTensor<bf16, OneChip, m![R2, TD]>,
    index: &HbmTensor<i32, OneChip, m![R2]>,
    output: &mut HbmTensor<bf16, OneChip, m![TC, TD]>,
) {
    let data_dm: DmTensor<bf16, OneChip, OneCluster, m![R2 / 2], m![R2 % 2, TD]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![TC], m![TD]>()
        .by_byte_offsets::<m![R2]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers `R4` rows at a 4-PE device (1 cluster x 256 slices).
#[device(chip = 1, pe = 4)]
pub fn four_pe_gather(
    device: &mut Device,
    table: &HbmTensor<bf16, OneChip, m![TK, TD]>,
    index: &HbmTensor<i32, OneChip, m![R4]>,
) -> HbmTensor<bf16, OneChip, m![R4, TD]> {
    let values: DmTensor<bf16, OneChip, OneCluster, m![R4 / 2], m![R4 % 2, TD]> = table
        .gather::<m![TK], m![TD]>()
        .by_byte_offsets::<m![R4]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Scatters `R4` rows at a 4-PE device.
#[device(chip = 1, pe = 4)]
pub fn four_pe_scatter(
    device: &mut Device,
    data: &HbmTensor<bf16, OneChip, m![R4, TD]>,
    index: &HbmTensor<i32, OneChip, m![R4]>,
    output: &mut HbmTensor<bf16, OneChip, m![TC, TD]>,
) {
    let data_dm: DmTensor<bf16, OneChip, OneCluster, m![R4 / 2], m![R4 % 2, TD]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![TC], m![TD]>()
        .by_byte_offsets::<m![R4]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers `R8` rows at an 8-PE device (2 clusters x 256 slices).
#[device(chip = 1, pe = 8)]
pub fn eight_pe_gather(
    device: &mut Device,
    table: &HbmTensor<bf16, OneChip, m![TK, TD]>,
    index: &HbmTensor<i32, OneChip, m![R8]>,
) -> HbmTensor<bf16, OneChip, m![R8, TD]> {
    let values: DmTensor<bf16, OneChip, PaddedCluster, m![R8 / 2], m![R8 % 2, TD]> = table
        .gather::<m![TK], m![TD]>()
        .by_byte_offsets::<m![R8]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Scatters `R8` rows at an 8-PE device.
#[device(chip = 1, pe = 8)]
pub fn eight_pe_scatter(
    device: &mut Device,
    data: &HbmTensor<bf16, OneChip, m![R8, TD]>,
    index: &HbmTensor<i32, OneChip, m![R8]>,
    output: &mut HbmTensor<bf16, OneChip, m![TC, TD]>,
) {
    let data_dm: DmTensor<bf16, OneChip, PaddedCluster, m![R8 / 2], m![R8 % 2, TD]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![TC], m![TD]>()
        .by_byte_offsets::<m![R8]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers `R8` rows independently on each of two chips.
#[device(chip = 2, pe = 8)]
pub fn two_chip_gather(
    device: &mut Device,
    table: &HbmTensor<bf16, TwoChips, m![TK, TD]>,
    index: &HbmTensor<i32, TwoChips, m![R8]>,
) -> HbmTensor<bf16, TwoChips, m![R8, TD]> {
    let values: DmTensor<bf16, TwoChips, PaddedCluster, m![R8 / 2], m![R8 % 2, TD]> = table
        .gather::<m![TK], m![TD]>()
        .by_byte_offsets::<m![R8]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Scatters `R8` rows on each of two chips.
#[device(chip = 2, pe = 8)]
pub fn two_chip_scatter(
    device: &mut Device,
    data: &HbmTensor<bf16, TwoChips, m![R8, TD]>,
    index: &HbmTensor<i32, TwoChips, m![R8]>,
    output: &mut HbmTensor<bf16, TwoChips, m![TC, TD]>,
) {
    let data_dm: DmTensor<bf16, TwoChips, PaddedCluster, m![R8 / 2], m![R8 % 2, TD]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![TC], m![TD]>()
        .by_byte_offsets::<m![R8]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers `R8` rows on each of four chips, the widest grid.
#[device(chip = 4, pe = 8)]
pub fn four_chip_gather(
    device: &mut Device,
    table: &HbmTensor<bf16, FourChips, m![TK, TD]>,
    index: &HbmTensor<i32, FourChips, m![R8]>,
) -> HbmTensor<bf16, FourChips, m![R8, TD]> {
    let values: DmTensor<bf16, FourChips, PaddedCluster, m![R8 / 2], m![R8 % 2, TD]> = table
        .gather::<m![TK], m![TD]>()
        .by_byte_offsets::<m![R8]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Scatters `R8` rows on each of four chips.
#[device(chip = 4, pe = 8)]
pub fn four_chip_scatter(
    device: &mut Device,
    data: &HbmTensor<bf16, FourChips, m![R8, TD]>,
    index: &HbmTensor<i32, FourChips, m![R8]>,
    output: &mut HbmTensor<bf16, FourChips, m![TC, TD]>,
) {
    let data_dm: DmTensor<bf16, FourChips, PaddedCluster, m![R8 / 2], m![R8 % 2, TD]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![TC], m![TD]>()
        .by_byte_offsets::<m![R8]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}
