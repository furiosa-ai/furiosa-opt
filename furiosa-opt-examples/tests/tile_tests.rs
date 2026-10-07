use furiosa_opt_examples::tile::{A, B, D, tile_simple, tile_window_commit};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Host function to test the device function.
#[tokio::test]
async fn test_tile_simple_host() -> eyre::Result<()> {
    // Host operations: create tensors, transfer to device
    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let mut device = Device::new(tile_simple.topology())?;
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    // Device operation via launch
    let output_hbm = launch(tile_simple, (&mut device, input_hbm.view())).await?;

    // Host operation: transfer back
    let output = output_hbm.to_host::<m![B, A]>(&mut device.pdma).await?;

    assert_eq!(
        input.into_inner().transpose::<m![B, A]>(false).into_vec(),
        output.into_vec(),
        "Transpose should not change the mathematical meaning of tensor"
    );

    Ok(())
}

/// Commits a fetched 32-wide window into the upper half of a 64-wide down-padded DM tile: the kernel
/// writes only `result[32..64]` (= `input[0..32]`) and leaves `result[0..32]` unwritten. A freshly
/// allocated destination starts as an all-zero blank canvas (`Backend::zeroed`) that only the
/// commit overwrites, so "unwritten" is the concrete, checkable claim "still zero".
#[tokio::test]
async fn test_tile_window_commit_host() -> eyre::Result<()> {
    let mut device = Device::new(tile_window_commit.topology())?;

    let input = HostTensor::<f32, m![D]>::from_vec((0..64).map(|x| x as f32).collect::<Vec<_>>())
        .to_hbm::<m![1], m![D]>(&mut device.pdma)
        .await?;

    let output = launch(tile_window_commit, (&mut device, &input)).await?;

    // result[32..64] is written from input[0..32]; result[0..32] (the out-of-tile down-pad cells)
    // must stay at the destination's zero-filled default.
    let actual: Vec<f32> = output.to_host::<m![D]>(&mut device.pdma).await?.into_vec();
    for i in 0..32 {
        assert_eq!(
            actual[i], 0.0,
            "result[{i}] (out-of-tile) must stay unwritten, got {:?}",
            actual[i]
        );
        assert_eq!(actual[32 + i], i as f32, "result[{}] should equal input[{}]", 32 + i, i);
    }

    Ok(())
}

#[tokio::test]
async fn test_dm_bottom_padding_does_not_overwrite_second_half() -> eyre::Result<()> {
    use furiosa_opt_examples::tile::dm_bottom_padding::{L, P, R, update_first_half};

    let update = f4e2m1::from_bits(0x2);
    let initial = f4e2m1::from_bits(0x4);
    let len = P::SIZE * R::SIZE * L::SIZE;
    let expected = (0..len)
        .map(|index| if index / L::SIZE % R::SIZE < 2 { update } else { initial })
        .collect::<Vec<_>>();

    let mut device = Device::new(update_first_half.topology())?;
    let updates = HostTensor::<f4e2m1, m![P, R, L]>::from_vec(vec![update; len])
        .to_hbm::<m![1], m![P, R, L]>(&mut device.pdma)
        .await?;
    let initial = HostTensor::<f4e2m1, m![P, R, L]>::from_vec(vec![initial; len])
        .to_hbm::<m![1], m![P, R, L]>(&mut device.pdma)
        .await?;

    let output = launch(update_first_half, (&mut device, &updates, &initial)).await?;
    let actual = output.to_host::<m![P, R, L]>(&mut device.pdma).await?.into_vec();

    assert_eq!(expected, actual);
    Ok(())
}
