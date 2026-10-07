use furiosa_opt_examples::fetch_assertions::{A, B};
use furiosa_opt_std::prelude::*;

mod cluster_size {
    use super::*;
    use furiosa_opt_examples::fetch_assertions::cluster_size::*;

    #[tokio::test]
    async fn test_valid_cluster_size() -> eyre::Result<()> {
        let mut device = Device::new(valid_cluster_size.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(valid_cluster_size, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}

mod slice_size {
    use super::*;
    use furiosa_opt_examples::fetch_assertions::slice_size::*;

    #[tokio::test]
    async fn test_valid_slice_size() -> eyre::Result<()> {
        let mut device = Device::new(valid_slice_size.topology())?;

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
                .await?;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(valid_slice_size, (&mut device, &input, &mut output)).await?;

        Ok(())
    }
}
