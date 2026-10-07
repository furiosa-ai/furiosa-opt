use furiosa_opt_examples::switch_assertions::{A, B, C, D, E, F};
use furiosa_opt_std::prelude::*;

mod alignment {
    use super::*;
    use furiosa_opt_examples::switch_assertions::alignment::*;

    #[tokio::test]
    async fn test_aligned_fetch_packet_i8() -> eyre::Result<()> {
        let mut device = Device::new(aligned_fetch_packet_i8.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(aligned_fetch_packet_i8, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
    #[tokio::test]
    async fn test_aligned_fetch_packet_bf16() -> eyre::Result<()> {
        let mut device = Device::new(aligned_fetch_packet_bf16.topology())?;

        let input = HostTensor::<bf16, m![A, B]>::from_vec(
            (0..<m![A, B]>::SIZE)
                .map(|x| bf16::from_f32(x as f32))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await?;

        let mut output = HbmTensor::<bf16, m![1], m![A, B]>::new();

        launch(aligned_fetch_packet_bf16, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

pub mod packet {
    use super::*;
    use furiosa_opt_examples::switch_assertions::packet::*;

    #[tokio::test]
    async fn test_packet_padding_unchanged() -> eyre::Result<()> {
        let mut device = Device::new(packet_padding_unchanged.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(packet_padding_unchanged, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_packet_padding_added_in_switch() -> eyre::Result<()> {
        let mut device = Device::new(packet_padding_added_in_switch.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(packet_padding_added_in_switch, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_packet_nested_padding() -> eyre::Result<()> {
        let mut device = Device::new(packet_nested_padding.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(packet_nested_padding, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_packet_restructuring() -> eyre::Result<()> {
        let mut device = Device::new(packet_restructuring.topology())?;

        let input =
            HostTensor::<i8, m![A, C]>::from_vec((0..<m![A, C]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, C]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, C / 16, C % 16]>::new();

        launch(packet_restructuring, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_padding() -> eyre::Result<()> {
        let mut device = Device::new(valid_padding.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(valid_padding, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

pub mod slice {
    use super::*;
    use furiosa_opt_examples::switch_assertions::slice::*;

    #[tokio::test]
    async fn test_valid_matching_slice_sizes() -> eyre::Result<()> {
        let mut device = Device::new(valid_matching_slice_sizes.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(valid_matching_slice_sizes, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

mod broadcast1 {
    use super::*;
    use furiosa_opt_examples::switch_assertions::broadcast1::*;

    #[tokio::test]
    async fn test_valid_basic() -> eyre::Result<()> {
        let mut device = Device::new(valid_basic.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B]>::new();

        launch(valid_basic, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_degenerate() -> eyre::Result<()> {
        let mut device = Device::new(valid_degenerate.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C % 4, B]>::new();

        launch(valid_degenerate, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

mod broadcast01 {
    use super::*;
    use furiosa_opt_examples::switch_assertions::broadcast01::*;

    #[tokio::test]
    async fn test_valid_only_slice1() -> eyre::Result<()> {
        let mut device = Device::new(valid_only_slice1.topology())?;

        let input = HostTensor::<i8, m![B]>::from_vec((0..<m![B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
            .to_hbm::<m![1], m![B]>(&mut device.pdma)
            .await?;

        let mut output = HbmTensor::<i8, m![1], m![F / 4, E / 4, B]>::new();

        launch(valid_only_slice1, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_with_time0() -> eyre::Result<()> {
        let mut device = Device::new(valid_with_time0.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 4, D % 4, A / 2, C / 2 % 2, A % 2, C % 2, B]>::new();

        launch(valid_with_time0, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_broadcast_with_padding() -> eyre::Result<()> {
        let mut device = Device::new(valid_broadcast_with_padding.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B]>::new();

        launch(valid_broadcast_with_padding, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

mod transpose {
    use super::*;
    use furiosa_opt_examples::switch_assertions::transpose::*;

    #[tokio::test]
    async fn test_valid_single_axis() -> eyre::Result<()> {
        let mut device = Device::new(valid_single_axis.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 64, C % 2, C / 2 % 32, A, B]>::new();

        launch(valid_single_axis, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_three_axes() -> eyre::Result<()> {
        let mut device = Device::new(valid_three_axes.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 128, C % 8, C / 8 % 16, A, B]>::new();

        launch(valid_three_axes, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_split_inner() -> eyre::Result<()> {
        let mut device = Device::new(valid_split_inner.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 16, C % 4, C / 4 % 4, A, B]>::new();

        launch(valid_split_inner, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

mod inter_transpose {
    use super::*;
    use furiosa_opt_examples::switch_assertions::inter_transpose::*;

    #[tokio::test]
    async fn test_valid() -> eyre::Result<()> {
        let mut device = Device::new(valid.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![C / 32, A / 2 % 2, C % 16, A / 4, A % 2, C / 16 % 2, B]>::new();

        launch(valid, (&mut device, &input, &mut output)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_degenerate() -> eyre::Result<()> {
        let mut device = Device::new(valid_degenerate.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, C % 32, C / 32 % 8, B]>::new();

        launch(valid_degenerate, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}
