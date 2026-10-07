use furiosa_opt_examples::contract_outer_assertions::{A, B, E, R, T, U, V};
use furiosa_opt_std::prelude::*;

type Chip = m![1];

mod lane_size {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::lane_size::*;

    #[tokio::test]
    async fn test_valid_size_8() -> eyre::Result<()> {
        let mut device = Device::new(valid_size_8.topology())?;

        let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_vec((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, R # 8]>::new();

        launch(valid_size_8, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }
}

mod cpacket_size {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::cpacket_size::*;

    #[tokio::test]
    async fn test_valid_size_64() -> eyre::Result<()> {
        let mut device = Device::new(valid_size_64.topology())?;

        let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_vec((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, R # 8]>::new();

        launch(valid_size_64, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_size_32() -> eyre::Result<()> {
        let mut device = Device::new(valid_size_32.topology())?;

        let input =
            HostTensor::<i8, m![A, B / 2]>::from_vec((0..<m![A, B / 2]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B / 2]>(&mut device.pdma)
                .await?;

        let input_trf =
            HostTensor::<i8, m![R, B / 2]>::from_vec((0..<m![R, B / 2]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B / 2]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, R # 8]>::new();

        launch(valid_size_32, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }
}

mod cpacket_mapping {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::cpacket_mapping::*;

    #[tokio::test]
    async fn test_valid_one_collect_flit() -> eyre::Result<()> {
        let mut device = Device::new(valid_one_collect_flit.topology())?;

        let input = HostTensor::<i8, m![A, E]>::from_vec((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_vec((0..<m![R, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, R # 8]>::new();

        launch(valid_one_collect_flit, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_two_collect_flits() -> eyre::Result<()> {
        let mut device = Device::new(valid_two_collect_flits.topology())?;

        let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_vec((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, R # 8]>::new();

        launch(valid_two_collect_flits, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }
}

mod time_broadcast {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::time_broadcast::*;

    #[tokio::test]
    async fn test_valid_single_tiling() -> eyre::Result<()> {
        let mut device = Device::new(valid_single_tiling.topology())?;

        let input = HostTensor::<i8, m![A, E]>::from_vec((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, T, E]>::from_vec((0..<m![R, T, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, T, E]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, T, R # 8]>::new();

        launch(valid_single_tiling, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_double_tiling() -> eyre::Result<()> {
        let mut device = Device::new(valid_double_tiling.topology())?;

        let input = HostTensor::<i8, m![A, E]>::from_vec((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut device.pdma)
            .await?;

        let input_trf = HostTensor::<i8, m![R, U, T, E]>::from_vec(
            (0..<m![R, U, T, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, U, T, E]>(&mut device.pdma)
        .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, U, T, R # 8]>::new();

        launch(valid_double_tiling, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_tiling_not_in_trf() -> eyre::Result<()> {
        let mut device = Device::new(valid_tiling_not_in_trf.topology())?;

        let input = HostTensor::<i8, m![A, E]>::from_vec((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_vec((0..<m![R, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, T, R # 8]>::new();

        launch(valid_tiling_not_in_trf, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_transposed_tiling() -> eyre::Result<()> {
        let mut device = Device::new(valid_transposed_tiling.topology())?;

        let input = HostTensor::<i8, m![A, E]>::from_vec((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut device.pdma)
            .await?;

        let input_trf = HostTensor::<i8, m![R, T, V, E]>::from_vec(
            (0..<m![R, T, V, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, T, V, E]>(&mut device.pdma)
        .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, V, T, R # 8]>::new();

        launch(valid_transposed_tiling, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }
}

mod trf_mapping {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::trf_mapping::*;

    #[tokio::test]
    async fn test_valid_mapping() -> eyre::Result<()> {
        let mut device = Device::new(valid_mapping.topology())?;

        let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut device.pdma)
            .await?;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_vec((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i32, Chip, m![A, 1 # 8]>::new();

        launch(valid_mapping, (&mut device, &input, &input_trf, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_unit_time_lane() -> eyre::Result<()> {
        let mut device = Device::new(valid_unit_time_lane.topology())?;

        let input_trf = HostTensor::<i8, m![E]>::from_vec((0..<m![E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![E]>(&mut device.pdma)
            .await?;

        launch(valid_unit_time_lane, (&mut device, &input_trf)).await?;

        Ok(())
    }
}

mod trf_size {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::trf_size::*;

    #[tokio::test]
    async fn test_valid_to_trf_full() -> eyre::Result<()> {
        let mut device = Device::new(valid_to_trf_full.topology())?;

        let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut device.pdma)
            .await?;

        let mut output = HbmTensor::<i8, Chip, m![A, B]>::new();

        launch(valid_to_trf_full, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}
