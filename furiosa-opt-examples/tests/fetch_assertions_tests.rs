use furiosa_opt_examples::fetch_assertions::{A, B};
use furiosa_opt_std::prelude::*;

mod cluster_size {
    use super::*;
    use furiosa_opt_examples::fetch_assertions::cluster_size::*;

    #[tokio::test]
    async fn test_valid_cluster_size() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

        launch(valid_cluster_size, (&mut *ctx, &input, &mut output)).await;
    }
}

mod slice_size {
    use super::*;
    use furiosa_opt_examples::fetch_assertions::slice_size::*;

    #[tokio::test]
    async fn test_valid_slice_size() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

        launch(valid_slice_size, (&mut *ctx, &input, &mut output)).await;
    }
}
