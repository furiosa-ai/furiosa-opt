//! Scatter and gather kernels covering index units, layouts, and runtime prefixes.

use furiosa_opt_std::prelude::*;

axes![
    A = 2,     // Payload axis outside the gather key
    K = 512,   // Scatter key
    D = 128,   // Payload per key
    C = 612,   // Cache length (non-power-of-2, > K: stresses unaligned coverage)
    G = 512,   // Slice-aligned gather count (G / 2 = 256)
    U = 768,   // Unaligned gather count (non-power-of-2, > K: 3 rows/slice, U / 3 = 256)
    V = 10240, // Large gather capacity (40 rows/slice)
    VO = 256,  // Outer index axis of the consecutive-axes sparse gather
    VI = 2,    // Inner index axis of the consecutive-axes sparse gather
    CL = 2,    // Real cluster partition (hardware has 2 clusters/chip): placed, not broadcast
    SC = 2,    // Chip partition of the two-chip sparse gather; `m![2]` would broadcast instead
    SF = 4     // Chip partition of the four-chip sparse gather
];

// 2048 i32 indices take 8 KiB, above one PE's 4 KiB SPM limit.
axes![
    Rows = 12,     // table rows the raw index selects (non-power-of-2, coprime to the index's 37)
    IdxRows = 16,  // index rows; `IdxRows * (Indices / 8) = 256` fills the hardware slices
    Indices = 128, // indices per index row
    Width = 8      // bf16 payload ELEMENTS per table row; the row's byte stride is `Width * 2 = 16`
];

// NBlocks * PBlock = 256 fills the slices of the paged-KV gather.
axes![
    NBlocks = 16,  // gathered block count
    PBlock = 16,   // block size (lane-aligned); NBlocks * PBlock = 256
    KvHeads = 8,   // outer payload axis
    HeadDim = 128  // inner payload axis
];

type Chip = m![1];
type PaddedCluster = m![1 # 2];
type BroadcastCluster = m![2];
type PlacedCluster = m![CL];

/// Scatter values into cache at index positions.
#[device(chip = 1)]
pub fn scatter_minimal(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, PaddedCluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![C], m![D]>()
        .by_byte_offsets::<m![K]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Scatters values at unscaled positions supplied in HBM.
#[device(chip = 1)]
pub fn scatter_spm(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, PaddedCluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut device.tdma);
    let index_dm: DmTensor<i32, Chip, PaddedCluster, m![1 # 256], m![K]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, PaddedCluster, m![1 # 4], m![K]> = index_dm.to_spm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![C], m![D]>()
        .by_positions::<m![K]>(&index_spm)
        .from_dm(&mut device.tdma, data_dm);
}

/// Scatters byte offsets into one consecutive multi-axis destination region.
#[device(chip = 1)]
pub fn scatter_contiguous_axes(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![A, C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, PaddedCluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![A, C], m![D]>()
        .by_byte_offsets::<m![K]>(index.view())
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers `G` indexed rows from `table` into a fresh HBM tensor.
#[device(chip = 1)]
pub fn gather_minimal(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Gathers over an indexed region spelled as two consecutive table axes.
///
/// An index value selects a position of `[A, K]` together, so the region is `A * K` rows wide and
/// the payload is the `D` that survives. The rule is that the region is consecutive, not that it is
/// one axis: lowering reshapes whatever it gets into the single axis a descriptor addresses.
#[device(chip = 1)]
pub fn gather_consecutive_indexed_axes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![A, K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let values: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![A, K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers a key that lies between two preserved table axes.
#[device(chip = 1)]
pub fn gather_middle_key(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![A, K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![A, G, D]> {
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![A, G % 2, D]> = table
        .gather::<m![K], m![A, D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Gathers only the runtime-valid prefix of `G` indices.
#[device(chip = 1)]
pub fn sparse_gather_runtime_length(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers from a table placed per chip through one index list every chip shares.
///
/// The index chip mapping is a broadcast, so one list sits in HBM and each chip reads the same
/// rows out of the table its own chip holds. A list spelled `m![SC]` would take a copy per chip.
#[device(chip = 2)]
pub fn gather_broadcast_index(
    device: &mut Device,
    table: &HbmTensor<bf16, m![SC], m![K, D]>,
    index: &HbmTensor<i32, m![2], m![G]>,
) -> HbmTensor<bf16, m![SC], m![G, D]> {
    let values: DmTensor<bf16, m![SC], PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers a runtime-valid prefix on each of two chips, from that chip's own table and list.
///
/// The prefix counts the entries one chip holds, so the same `valid_length` shortens every chip's
/// list by itself. Each chip reads only the table its own chip holds.
#[device(chip = 2)]
pub fn sparse_gather_two_chips(
    device: &mut Device,
    table: &HbmTensor<bf16, m![SC], m![K, D]>,
    index: &HbmTensor<i32, m![SC], m![G]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, m![SC], m![G, D]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, m![SC], PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers a runtime-valid prefix on each of four chips, the widest grid.
#[device(chip = 4)]
pub fn sparse_gather_four_chips(
    device: &mut Device,
    table: &HbmTensor<bf16, m![SF], m![K, D]>,
    index: &HbmTensor<i32, m![SF], m![G]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, m![SF], m![G, D]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, m![SF], PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers a runtime-valid prefix spanning the consecutive axes `VO` and `VI`.
#[device(chip = 1)]
pub fn sparse_gather_consecutive_axes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![VO, VI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![VO, VI, D]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![VO, VI, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![VO, VI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

// QD keeps the innermost table and output transfer at the same eight-byte width.
axes![QO = 128, QI = 8, QP = 4, QD = 4];
axes![QA = 512];

/// Places the outer factor of one index axis across slices.
#[device(chip = 1)]
pub fn sparse_gather_outer_factor_in_slices(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QA]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QA, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![QA / 2 # 256], m![QA % 2, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QA]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Places the inner factor of one index axis across slices.
#[device(chip = 1)]
pub fn sparse_gather_inner_factor_in_slices(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QA]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QA, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![QA % 2 # 256], m![QA / 2, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QA]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with both index axes in their declared order in elements.
#[device(chip = 1)]
pub fn sparse_gather_index_axes_in_elements(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QO, QI, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with its two index axes reversed in data memory.
#[device(chip = 1)]
pub fn sparse_gather_permuted_index_axes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QI, QO, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with one payload axis between its index axes.
#[device(chip = 1)]
pub fn sparse_gather_payload_between_index_axes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QP, QI, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QO, QP, QI, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with the index axes reversed around a payload axis.
#[device(chip = 1)]
pub fn sparse_gather_reversed_around_payload(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QI, QP, QO, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QI, QP, QO, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with the outer index axis placed across slices.
#[device(chip = 1)]
pub fn sparse_gather_outer_index_in_slices(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![QO # 256], m![QI, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with the inner index axis placed across slices.
#[device(chip = 1)]
pub fn sparse_gather_inner_index_in_slices(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![QI # 256], m![QO, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with one index axis split across slices and elements.
#[device(chip = 1)]
pub fn sparse_gather_split_outer_index(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![QO / 2 # 256], m![QO % 2, QI, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with a factorized index axis kept together in elements.
#[device(chip = 1)]
pub fn sparse_gather_factorized_index_in_elements(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QO / 2, QO % 2, QI, QP, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with payload placed between two parts of an index axis.
#[device(chip = 1)]
pub fn sparse_gather_payload_between_index_parts(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QO / 2, QP, QO % 2, QI, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with two parts of a payload axis around an index axis.
#[device(chip = 1)]
pub fn sparse_gather_split_payload_around_index(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QO, QI, QP, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QO, QP / 2, QI, QP % 2, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a sparse prefix with payload outside both index axes.
#[device(chip = 1)]
pub fn sparse_gather_payload_before_index_axes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![QP, QO, QI, QD]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QP, QO, QI, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers the same split payload without a sparse prefix for comparison.
#[device(chip = 1)]
pub fn gather_payload_between_index_axes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, QP, QD]>,
    index: &HbmTensor<i32, Chip, m![QO, QI]>,
) -> HbmTensor<bf16, Chip, m![QO, QP, QI, QD]> {
    let values: DmTensor<bf16, Chip, PaddedCluster, m![1 # 256], m![QO, QP, QI, QD]> = table
        .gather::<m![K], m![QP, QD]>()
        .by_byte_offsets::<m![QO, QI]>(index)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

/// Gathers a runtime-valid prefix of 768 indices.
#[device(chip = 1)]
pub fn sparse_gather_unaligned_runtime_length(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![U]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![U, D]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    // Three positions per slice give the required 256-slice placement.
    let values: DmTensor<bf16, Chip, PaddedCluster, m![U / 3], m![U % 3, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![U]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers up to 10,240 indexed rows using a runtime valid length.
#[device(chip = 1)]
pub fn sparse_gather_large_runtime_length(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![V]>,
    valid_length: &HbmScalar<i32>,
) -> HbmTensor<bf16, Chip, m![V, D]> {
    let valid_length = valid_length.to_spm(&mut device.tdma);
    let values: DmTensor<bf16, Chip, PaddedCluster, m![V / 40], m![V % 40, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![V]>(index)
        .sparse_prefix(valid_length)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers 768 indexed rows from a 512-row table.
#[device(chip = 1)]
pub fn gather_unaligned(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![U]>,
) -> HbmTensor<bf16, Chip, m![U, D]> {
    // Three rows per slice keep the non-power-of-2 count on 256 slices.
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![U / 3], m![U % 3, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![U]>(index)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Gathers a paged KV pool while reusing `NBlocks` in the index and output.
#[device(chip = 1)]
pub fn gather_paged_kv(
    device: &mut Device,
    pool: &HbmTensor<bf16, Chip, m![NBlocks, KvHeads, PBlock, HeadDim]>,
    block_table: &HbmTensor<i32, Chip, m![NBlocks]>,
) -> HbmTensor<bf16, Chip, m![NBlocks, KvHeads, PBlock, HeadDim]> {
    let gathered: DmTensor<bf16, Chip, PaddedCluster, m![NBlocks, PBlock], m![KvHeads, HeadDim]> = pool
        .gather::<m![NBlocks], m![KvHeads, PBlock, HeadDim]>()
        .by_byte_offsets::<m![NBlocks]>(block_table)
        .to_dm(&mut device.tdma);

    gathered.to_hbm(&mut device.tdma)
}

/// Gathers rows selected by unscaled indices supplied in HBM.
#[device(chip = 1)]
pub fn gather_aligned_spm(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    type Slice = m![G / 2];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, BroadcastCluster, Slice, m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, DummyPe, m![G]> = index_dm.to_spm(&mut device.tdma);
    let gather = table.gather::<m![K], m![D]>().by_positions::<m![G]>(&index_spm);
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, Slice, m![G % 2, D]> = gather.to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Gathers rows selected by unscaled indices beyond one PE's SPM capacity.
#[device(chip = 1)]
pub fn gather_byte_offsets_from_indices(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![Rows, Width]>,
    index: &HbmTensor<i32, Chip, m![IdxRows, Indices]>,
) -> HbmTensor<bf16, Chip, m![IdxRows, Indices, Width]> {
    type Slice = m![IdxRows, Indices / 8];
    type Packet = m![Indices % 8];
    const ROW_BYTES: i32 = (<m![Width]>::SIZE * size_of::<bf16>()) as i32;

    let raw: DmTensor<i32, Chip, PaddedCluster, Slice, Packet> = index.to_dm(&mut device.tdma);
    // Convert raw positions to byte offsets using the bf16 row width.
    let scaled: DmTensor<i32, Chip, PaddedCluster, Slice, Packet> = device
        .main
        .begin(raw.view())
        .fetch::<m![1], Packet>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![Indices % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::MulInt, ROW_BYTES)
        .vector_final()
        .commit_trim::<Packet>()
        .commit();
    let offsets: HbmTensor<i32, Chip, m![IdxRows, Indices]> = scaled.to_hbm(&mut device.tdma);

    let values: DmTensor<bf16, Chip, PaddedCluster, Slice, m![Indices % 8, Width]> = table
        .gather::<m![Rows], m![Width]>()
        .by_byte_offsets::<m![IdxRows, Indices]>(&offsets)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}

/// Gathers rows selected by a large unscaled index supplied in HBM.
#[device(chip = 1)]
pub fn gather_spm_split_index(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![Rows, Width]>,
    index: &HbmTensor<i32, Chip, m![IdxRows, Indices]>,
) -> HbmTensor<bf16, Chip, m![IdxRows, Indices, Width]> {
    type FullSlice = m![IdxRows # 256];
    type Slice = m![IdxRows, Indices % 32 / 2];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, BroadcastCluster, FullSlice, m![Indices]> = index.to_dm(&mut device.tdma);
    let mut out = HbmTensor::<bf16, Chip, m![IdxRows, Indices, Width]>::new();
    // Four 512-entry SPM windows fit the per-PE budget; tiled writes reassemble the result.
    for c in 0..<m![Indices / 32]>::SIZE {
        let chunk_spm: SpmTensor<i32, Chip, BroadcastCluster, DummyPe, m![IdxRows, Indices % 32]> = index_dm
            .view()
            .tile::<m![Indices / 32], 1, m![1 # 4, Indices % 32]>(c)
            .to_spm::<DummyPe, m![IdxRows, Indices % 32]>(&mut device.tdma);
        let values: DmTensor<bf16, Chip, PaddedCluster, Slice, m![Indices % 2, Width]> = table
            .gather::<m![Rows], m![Width]>()
            .by_positions::<m![IdxRows, Indices % 32]>(&chunk_spm)
            .to_dm(&mut device.tdma);

        values.view().to_hbm_view(
            &mut device.tdma,
            out.view_mut()
                .tile::<m![Indices / 32], 1, m![IdxRows, 1 #{!} 4, Indices % 32, Width]>(c),
        );
    }

    out
}

/// Gathers through a placed SPM index, with a distinct list on each cluster.
#[device(chip = 1)]
pub fn gather_placed_spm(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![CL, G]>,
) -> HbmTensor<bf16, Chip, m![CL, G, D]> {
    type Slice = m![G / 2];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, PlacedCluster, Slice, m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, PlacedCluster, DummyPe, m![G]> = index_dm.to_spm(&mut device.tdma);
    let values_dm: DmTensor<bf16, Chip, PlacedCluster, Slice, m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_positions::<m![CL, G]>(&index_spm)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Scatters through a per-cluster index list, with the cluster partition carried as an ordinary
/// axis of the updates rather than in the cluster mapping.
///
/// A scatter's domain cuts the updates, so it is matched against their slice and element axes. A
/// cluster partition placed in the cluster mapping is not among those, but the same partition
/// spelled as an axis of `Slice` is, and then `m![CL, G]` is one consecutive region of the updates
/// and each cluster keeps its own list.
#[device(chip = 1)]
pub fn scatter_cluster_axis_in_slice(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![CL, G, D]>,
    index: &HbmTensor<i32, Chip, m![CL, G]>,
    output: &mut HbmTensor<bf16, Chip, m![V, D]>,
) {
    type Slice = m![CL, G / 4];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, BroadcastCluster, Slice, m![G % 4]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, DummyPe, m![CL, G]> = index_dm.to_spm(&mut device.tdma);
    let data_dm: DmTensor<bf16, Chip, BroadcastCluster, Slice, m![G % 4, D]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![V], m![D]>()
        .by_positions::<m![CL, G]>(&index_spm)
        .from_dm(&mut device.tdma, data_dm);
}

/// Scatters through a per-cluster index list kept in the cluster mapping, so the domain spans the
/// cluster placement and the slice axes at once.
///
/// `m![CL, G]` takes `CL` from the updates' cluster mapping and `G` from their slice and element
/// axes. The three are one placement to the sequencer, which is what lets the domain cross them.
#[device(chip = 1)]
pub fn scatter_placed_cluster_domain(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![CL, G, D]>,
    index: &HbmTensor<i32, Chip, m![CL, G]>,
    output: &mut HbmTensor<bf16, Chip, m![V, D]>,
) {
    type Slice = m![G / 2];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, PlacedCluster, Slice, m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, PlacedCluster, DummyPe, m![G]> = index_dm.to_spm(&mut device.tdma);
    let data_dm: DmTensor<bf16, Chip, PlacedCluster, Slice, m![G % 2, D]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![V], m![D]>()
        .by_positions::<m![CL, G]>(&index_spm)
        .from_dm(&mut device.tdma, data_dm);
}

/// Scatters through an index whose domain is split around a payload axis in the updates.
///
/// The updates are laid out `[G / 2, A, G % 2, D]`, so neither the domain `m![G]` nor the payload
/// `m![A, D]` is one consecutive region: each is two runs interleaved with the other. `D` stays
/// innermost to keep every stride byte-aligned.
#[device(chip = 1)]
pub fn scatter_payload_between_index_axes(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![G, A, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
    output: &mut HbmTensor<bf16, Chip, m![V, A, D]>,
) {
    type Slice = m![G / 2];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, BroadcastCluster, Slice, m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, DummyPe, m![G]> = index_dm.to_spm(&mut device.tdma);
    let data_dm: DmTensor<bf16, Chip, PaddedCluster, Slice, m![A, G % 2, D]> = data.to_dm(&mut device.tdma);

    output
        .view_mut()
        .scatter::<m![V], m![A, D]>()
        .by_positions::<m![G]>(&index_spm)
        .from_dm(&mut device.tdma, data_dm);
}

/// Gathers by one digit of a table axis, carrying the digit it leaves behind as payload.
///
/// `m![K / 2]` is a consecutive run of `m![K, D]`, so an index entry steps two table rows at a
/// time and `m![K % 2, D]` rides along. Nothing may fuse the two digits back together.
#[device(chip = 1)]
pub fn gather_by_table_axis_digit(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, K % 2, D]> {
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, K % 2, D]> = table
        .gather::<m![K / 2], m![K % 2, D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Gathers through a per-cluster index list whose cluster partition is an ordinary axis of the
/// result rather than the cluster mapping.
///
/// The twin of `gather_placed_spm`: the same `m![CL, G]` domain, spelled into `Slice` instead of
/// the cluster placement, so the partition is mixed in with the rest of the layout.
#[device(chip = 1)]
pub fn gather_cluster_axis_in_slice(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![CL, G]>,
) -> HbmTensor<bf16, Chip, m![CL, G, D]> {
    type Slice = m![CL, G / 4];
    type DummyPe = m![1 # { Slice::SIZE / SLICES_PER_PE }];

    let index_dm: DmTensor<i32, Chip, BroadcastCluster, Slice, m![G % 4]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, DummyPe, m![CL, G]> = index_dm.to_spm(&mut device.tdma);
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, Slice, m![G % 4, D]> = table
        .gather::<m![K], m![D]>()
        .by_positions::<m![CL, G]>(&index_spm)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Gathers through an index whose element mapping scrambles the digits of one axis.
///
/// The domain is `[G / 4, G % 2, G / 2 % 2]`, so the stored order of the list is a permutation of
/// `m![G]` rather than a re-spelling of it. Nothing may fuse the digits back together on the way
/// down.
#[device(chip = 1)]
pub fn gather_scrambled_index_digits(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G / 4, G % 2, G / 2 % 2]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G / 4, G % 2, G / 2 % 2]>(index)
        .to_dm(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Binds a scatter plan to a name before supplying its values.
#[device(chip = 1)]
pub fn scatter_rebound_plan(
    device: &mut Device,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, PaddedCluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut device.tdma);

    let plan = output.view_mut().scatter::<m![C], m![D]>();
    let plan = plan.by_byte_offsets::<m![K]>(index.view());
    plan.from_dm(&mut device.tdma, data_dm);
}
