//! MoE primitives for runtime-scalar execution guards.

use furiosa_opt_std::prelude::*;

axes![
    Slice = 64,
    Group = 4,
    Width = 8,
    Expert = 1024,
    MatmulOut = 8,
    MatmulRed = 16,
    ScalarOutput = 128,
    ChipScalarOutput = 1024
];

type Chip = m![1];
type Cluster = m![1];

// ANCHOR: runtime_work_skipping
/// Copies requested expert groups through a guarded vector fetch/commit with a static `Group` bound.
#[device(chip = 1, pe = 1)]
pub fn bounded_dynamic_expert_loop(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![Slice, Group, Width]>,
    active_experts: &HbmScalar<usize>,
) -> HbmTensor<i32, Chip, m![Slice, Group, Width]> {
    let input = input.to_dm::<Cluster, m![Slice], m![Group, Width]>(&mut device.tdma);
    let mut output = DmTensor::<i32, Chip, Cluster, m![Slice], m![Group, Width]>::new();
    output.view_mut().memset(0, &mut device.sub);

    let active_experts = active_experts.to_spm(&mut device.tdma);
    for group in 0..Group::SIZE {
        if group < active_experts {
            let source = input.view().tile::<m![Group], 1, m![1 # { Group::SIZE }, Width]>(group);
            let target = output
                .view_mut()
                .tile::<m![Group], 1, m![1 #{!} { Group::SIZE }, Width]>(group);
            device
                .main
                .begin(source)
                .fetch::<m![1], m![Width]>()
                .collect::<m![1], m![Width]>()
                .commit_trim::<m![Width]>()
                .commit_view(target);
        }
    }

    output.to_hbm(&mut device.tdma)
}
// ANCHOR_END: runtime_work_skipping

// ANCHOR: unrolled_runtime_work_skipping
/// Copies requested expert groups after unrolling the static `Group` loop.
#[device(chip = 1, pe = 1)]
pub fn unrolled_bounded_dynamic_expert_loop(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![Slice, Group, Width]>,
    active_experts: &HbmScalar<usize>,
) -> HbmTensor<i32, Chip, m![Slice, Group, Width]> {
    let input = input.to_dm::<Cluster, m![Slice], m![Group, Width]>(&mut device.tdma);
    let mut output = DmTensor::<i32, Chip, Cluster, m![Slice], m![Group, Width]>::new();
    output.view_mut().memset(0, &mut device.sub);

    let active_experts = active_experts.to_spm(&mut device.tdma);
    #[unroll]
    for group in 0..Group::SIZE {
        if group < active_experts {
            let source = input.view().tile::<m![Group], 1, m![1 # { Group::SIZE }, Width]>(group);
            let target = output
                .view_mut()
                .tile::<m![Group], 1, m![1 #{!} { Group::SIZE }, Width]>(group);
            device
                .main
                .begin(source)
                .fetch::<m![1], m![Width]>()
                .collect::<m![1], m![Width]>()
                .commit_trim::<m![Width]>()
                .commit_view(target);
        }
    }

    output.to_hbm(&mut device.tdma)
}
// ANCHOR_END: unrolled_runtime_work_skipping

/// Copies one expert group selected by an HBM runtime index.
/// Execution rejects indices outside `0..Group::SIZE`.
#[device(chip = 1, pe = 1)]
pub fn runtime_scalar_indexed_copy(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![Slice, Group, Width]>,
    selected_group: &HbmScalar<usize>,
) -> HbmTensor<i32, Chip, m![Slice, Width]> {
    let input = input.to_dm::<Cluster, m![Slice], m![Group, Width]>(&mut device.tdma);
    let mut output = DmTensor::<i32, Chip, Cluster, m![Slice], m![Width]>::new();
    let selected_group = selected_group.to_spm(&mut device.tdma);
    let source = input
        .view()
        .tile::<m![Group], 1, m![1 # { Group::SIZE }, Width]>(selected_group);
    device
        .main
        .begin(source)
        .fetch::<m![1], m![Width]>()
        .collect::<m![1], m![Width]>()
        .commit_trim::<m![Width]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}

// ANCHOR: runtime_matmul_guard
/// Runs matmul for the requested experts and leaves the remaining experts zero.
/// Every `Expert` iteration is scheduled and its body is guarded at runtime.
#[device(chip = 1, pe = 1)]
pub fn bounded_dynamic_matmul(
    device: &mut Device,
    activation: &HbmTensor<bf16, Chip, m![Slice, MatmulRed]>,
    weight: &HbmTensor<bf16, Chip, m![Slice, Expert, MatmulOut, MatmulRed]>,
    active_experts: &HbmScalar<usize>,
) -> HbmTensor<f32, Chip, m![Slice, Expert, MatmulOut]> {
    let activation: DmTensor<bf16, Chip, Cluster, m![Slice], m![MatmulRed]> = activation.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, m![Slice], m![Expert, MatmulOut, MatmulRed]> =
        weight.to_dm(&mut device.tdma);
    let mut output: DmTensor<f32, Chip, Cluster, m![Slice], m![Expert, MatmulOut]> = DmTensor::new();
    output.view_mut().memset(0.0, &mut device.sub);

    let active_experts = active_experts.to_spm(&mut device.tdma);
    for expert in 0..Expert::SIZE {
        if expert < active_experts {
            let weight_group = weight
                .view()
                .tile::<m![Expert], 1, m![1 # { Expert::SIZE }, MatmulOut, MatmulRed]>(expert);
            let weight_trf: TrfTensor<bf16, Chip, Cluster, m![Slice], m![MatmulOut], m![MatmulRed]> = device
                .sub
                .begin(weight_group)
                .fetch::<m![MatmulOut], m![MatmulRed]>()
                .collect::<m![MatmulOut], m![MatmulRed]>()
                .to_trf();

            device
                .main
                .begin(activation.view())
                .fetch::<m![1], m![MatmulRed]>()
                .collect::<m![1], m![MatmulRed]>()
                .contract_outer::<m![1], m![MatmulRed], _, _, _>(&weight_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![1]>()
                .contract_lane::<m![1], m![MatmulOut]>(LaneMode::Interleaved)
                .commit_trim::<m![MatmulOut]>()
                .commit_view(
                    output
                        .view_mut()
                        .tile::<m![Expert], 1, m![1 #{!} { Expert::SIZE }, MatmulOut]>(expert),
                );
        }
    }
    output.to_hbm(&mut device.tdma)
}
// ANCHOR_END: runtime_matmul_guard

/// Writes a scalar guard result across one chip.
#[device(chip = 1, pe = 1)]
pub fn scalar_guard_one_chip(device: &mut Device, value: HbmScalar<i32>) -> HbmTensor<i32, m![1], m![ScalarOutput]> {
    let mut output: DmTensor<i32, m![1], m![1], m![64], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let value = value.to_spm(&mut device.tdma);
    if value > 0 {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}

/// Writes one when an HBM boolean scalar is true.
#[device(chip = 1, pe = 1)]
pub fn bool_scalar_guard(device: &mut Device, enabled: &HbmScalar<bool>) -> HbmTensor<i32, m![1], m![ScalarOutput]> {
    let mut output: DmTensor<i32, m![1], m![1], m![64], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let enabled = enabled.to_spm(&mut device.tdma);
    if enabled {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}

/// Writes the same scalar guard result into every chip's output region.
#[device(chip = 4, pe = 8)]
pub fn scalar_guard_all_chips(
    device: &mut Device,
    value: &HbmScalar<i32>,
) -> HbmTensor<i32, m![4], m![ChipScalarOutput]> {
    let mut output: DmTensor<i32, m![4], m![2], m![256], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let value = value.to_spm(&mut device.tdma);
    if value > 0 {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}

/// Writes one when Rust's signed-to-unsigned cast makes the input positive.
#[device(chip = 1, pe = 1)]
pub fn signed_to_unsigned_guard(
    device: &mut Device,
    value: &HbmScalar<i32>,
) -> HbmTensor<i32, m![1], m![ScalarOutput]> {
    let mut output: DmTensor<i32, m![1], m![1], m![64], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let value = value.to_spm(&mut device.tdma);
    if (value as u32) > 0 {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}

/// Writes one when a `u32` survives zero-extension to the target-width `usize`.
#[device(chip = 1, pe = 1)]
pub fn u32_to_usize_guard(device: &mut Device, value: &HbmScalar<u32>) -> HbmTensor<i32, m![1], m![ScalarOutput]> {
    let mut output: DmTensor<i32, m![1], m![1], m![64], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let value = value.to_spm(&mut device.tdma) as usize;
    if value + 1 > u32::MAX as usize {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}

/// Writes one when an HBM scalar retains bits above the 32-bit range in SPM.
#[device(chip = 1, pe = 1)]
pub fn u64_scalar_guard(device: &mut Device, value: &HbmScalar<u64>) -> HbmTensor<i32, m![1], m![ScalarOutput]> {
    let mut output: DmTensor<i32, m![1], m![1], m![64], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let value = value.to_spm(&mut device.tdma);
    if value > u32::MAX as u64 {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}

/// Writes one when a signed 64-bit HBM scalar is negative.
#[device(chip = 1, pe = 1)]
pub fn i64_scalar_guard(device: &mut Device, value: &HbmScalar<i64>) -> HbmTensor<i32, m![1], m![ScalarOutput]> {
    let mut output: DmTensor<i32, m![1], m![1], m![64], m![2]> = DmTensor::new();
    output.view_mut().memset(0, &mut device.sub);
    let value = value.to_spm(&mut device.tdma);
    if value < 0 {
        output.view_mut().memset(1, &mut device.sub);
    }
    output.to_hbm(&mut device.tdma)
}
