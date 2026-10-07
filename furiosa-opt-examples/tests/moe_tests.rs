use furiosa_opt_examples::moe::{
    Expert, Group, MatmulOut, MatmulRed, ScalarOutput, Slice, Width, bool_scalar_guard, bounded_dynamic_expert_loop,
    bounded_dynamic_matmul, scalar_guard_one_chip, unrolled_bounded_dynamic_expert_loop,
};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn hbm_scalar_can_move_into_a_kernel() -> eyre::Result<()> {
    let mut device = Device::new(scalar_guard_one_chip.topology())?;
    let scalar = HbmScalar::from_host(1, &mut device.pdma).await?;
    let output = launch(scalar_guard_one_chip, (&mut device, scalar)).await?;
    let output = output.to_host::<m![ScalarOutput]>(&mut device.pdma).await?.into_vec();

    assert_eq!(output, vec![1; ScalarOutput::SIZE]);
    Ok(())
}

#[tokio::test]
async fn bool_hbm_scalar_controls_a_branch() -> eyre::Result<()> {
    let mut device = Device::new(bool_scalar_guard.topology())?;
    let mut enabled = HbmScalar::from_host(false, &mut device.pdma).await?;
    for value in [false, true] {
        enabled.write(value, &mut device.pdma).await?;
        let output = launch(bool_scalar_guard, (&mut device, &enabled)).await?;
        let output = output.to_host::<m![ScalarOutput]>(&mut device.pdma).await?.into_vec();
        assert_eq!(output, vec![i32::from(value); ScalarOutput::SIZE]);
    }
    Ok(())
}

#[tokio::test]
async fn bounded_dynamic_loop_handles_empty_partial_and_full_ranges() -> eyre::Result<()> {
    let mut device = Device::new(bounded_dynamic_expert_loop.topology())?;
    let input_values = (0..<m![Slice, Group, Width]>::SIZE)
        .map(|value| value as i32)
        .collect::<Vec<_>>();
    let input = HostTensor::<i32, m![Slice, Group, Width]>::from_vec(input_values.clone());
    let input = input.to_hbm(&mut device.pdma).await?;

    let mut active_experts_hbm = HbmScalar::<usize>::from_host(0, &mut device.pdma).await?;
    for active_experts in [0, 1, 2, Group::SIZE, Group::SIZE + 1, usize::MAX] {
        active_experts_hbm.write(active_experts, &mut device.pdma).await?;
        let output = launch(bounded_dynamic_expert_loop, (&mut device, &input, &active_experts_hbm)).await?;
        let output = output
            .to_host::<m![Slice, Group, Width]>(&mut device.pdma)
            .await?
            .into_vec();

        for slice in 0..Slice::SIZE {
            for group in 0..Group::SIZE {
                for width in 0..Width::SIZE {
                    let index = (slice * Group::SIZE + group) * Width::SIZE + width;
                    let expected = if group < active_experts { input_values[index] } else { 0 };
                    assert_eq!(
                        output[index], expected,
                        "active_experts {active_experts}, slice {slice}, group {group}, lane {width}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn unrolled_loop_keeps_the_runtime_guard() -> eyre::Result<()> {
    let mut device = Device::new(unrolled_bounded_dynamic_expert_loop.topology())?;
    let input_values = (0..<m![Slice, Group, Width]>::SIZE)
        .map(|value| value as i32)
        .collect::<Vec<_>>();
    let input = HostTensor::<i32, m![Slice, Group, Width]>::from_vec(input_values.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let mut active = HbmScalar::<usize>::from_host(0, &mut device.pdma).await?;

    for active_experts in [0, 1, 2, Group::SIZE, Group::SIZE + 1] {
        active.write(active_experts, &mut device.pdma).await?;
        let output = launch(unrolled_bounded_dynamic_expert_loop, (&mut device, &input, &active))
            .await?
            .to_host::<m![Slice, Group, Width]>(&mut device.pdma)
            .await?
            .into_vec();

        for (index, value) in output.into_iter().enumerate() {
            let group = (index / Width::SIZE) % Group::SIZE;
            let expected = if group < active_experts { input_values[index] } else { 0 };
            assert_eq!(value, expected, "active_experts {active_experts}, element {index}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn bounded_dynamic_matmul_skips_inactive_experts() -> eyre::Result<()> {
    let mut device = Device::new(bounded_dynamic_matmul.topology())?;
    let activation =
        HostTensor::<bf16, m![Slice, MatmulRed]>::from_vec(vec![bf16::from_f32(1.0); <m![Slice, MatmulRed]>::SIZE]);
    let weight_values = (0..<m![Slice]>::SIZE)
        .flat_map(|_| {
            (0..<m![Expert]>::SIZE).flat_map(|group| {
                std::iter::repeat_n(bf16::from_f32((group + 1) as f32), <m![MatmulOut, MatmulRed]>::SIZE)
            })
        })
        .collect::<Vec<_>>();
    let weight = HostTensor::<bf16, m![Slice, Expert, MatmulOut, MatmulRed]>::from_vec(weight_values);
    let activation = activation.to_hbm(&mut device.pdma).await?;
    let weight = weight.to_hbm(&mut device.pdma).await?;

    for active_experts in [0usize, 3, 8] {
        let active_experts_hbm = HbmScalar::<usize>::from_host(active_experts, &mut device.pdma).await?;
        let output = launch(
            bounded_dynamic_matmul,
            (&mut device, &activation, &weight, &active_experts_hbm),
        )
        .await?
        .to_host::<m![Slice, Expert, MatmulOut]>(&mut device.pdma)
        .await?
        .into_vec();

        for (slice_index, slice) in output.chunks_exact(Expert::SIZE * MatmulOut::SIZE).enumerate() {
            for group in 0..Expert::SIZE {
                let expected = if group < active_experts {
                    MatmulRed::SIZE as f32 * (group + 1) as f32
                } else {
                    0.0
                };
                for lane in 0..MatmulOut::SIZE {
                    assert_eq!(
                        slice[group * MatmulOut::SIZE + lane],
                        expected,
                        "active_experts {active_experts}, slice {slice_index}, expert {group}, lane {lane}"
                    );
                }
            }
        }
    }
    Ok(())
}
