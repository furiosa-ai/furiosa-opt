use furiosa_opt_examples::contract_outer_assertions::{A, B, E, R, T, U, V};
use furiosa_opt_std::prelude::*;

type Chip = m![1];

mod lane_size {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::lane_size::*;

    #[tokio::test]
    async fn test_valid_size_1() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf = HostTensor::<i8, m![B]>::from_buf((0..<m![B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![B]>(&mut ctx.pdma, 0)
            .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, 1 # 8]>::from_addr(0x1000) };

        launch(valid_size_1, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_2() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R / 4, B]>::from_buf((0..<m![R / 4, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R / 4, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R / 4 # 8]>::from_addr(0x1000) };

        launch(valid_size_2, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_4() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R / 2, B]>::from_buf((0..<m![R / 2, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R / 2, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R / 2 # 8]>::from_addr(0x1000) };

        launch(valid_size_4, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_8() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_size_8, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod cpacket_size {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::cpacket_size::*;

    #[tokio::test]
    async fn test_valid_size_64() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_size_64, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_32() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B / 2]>::from_buf((0..<m![A, B / 2]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B / 2]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B / 2]>::from_buf((0..<m![R, B / 2]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B / 2]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_size_32, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod cpacket_mapping {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::cpacket_mapping::*;

    #[tokio::test]
    async fn test_valid_one_collect_flit() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_buf((0..<m![R, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_one_collect_flit, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_two_collect_flits() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_two_collect_flits, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod time_broadcast {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::time_broadcast::*;

    #[tokio::test]
    async fn test_valid_single_tiling() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, T, E]>::from_buf((0..<m![R, T, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, T, E]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, T, R # 8]>::from_addr(0x1000) };

        launch(valid_single_tiling, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_double_tiling() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
            .await;

        let input_trf = HostTensor::<i8, m![R, U, T, E]>::from_buf(
            (0..<m![R, U, T, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, U, T, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, U, T, R # 8]>::from_addr(0x1000) };

        launch(valid_double_tiling, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_tiling_not_in_trf() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_buf((0..<m![R, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, T, R # 8]>::from_addr(0x1000) };

        launch(valid_tiling_not_in_trf, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_transposed_tiling() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
            .await;

        let input_trf = HostTensor::<i8, m![R, T, V, E]>::from_buf(
            (0..<m![R, T, V, E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, T, V, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, V, T, R # 8]>::from_addr(0x1000) };

        launch(valid_transposed_tiling, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod trf_mapping {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::trf_mapping::*;

    #[tokio::test]
    async fn test_valid_mapping() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, 1 # 8]>::from_addr(0x1000) };

        launch(valid_mapping, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_unit_time_lane() {
        let mut ctx = Context::acquire();

        let input_trf = HostTensor::<i8, m![E]>::from_buf((0..<m![E]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![E]>(&mut ctx.pdma, 0)
            .await;

        launch(valid_unit_time_lane, (&mut *ctx, &input_trf)).await;
    }
}

mod trf_size {
    use super::*;
    use furiosa_opt_examples::contract_outer_assertions::trf_size::*;

    #[tokio::test]
    async fn test_valid_to_trf_full() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

        launch(valid_to_trf_full, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_to_trf_half() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>())
            .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
            .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

        launch(valid_to_trf_half, (&mut *ctx, &input, &mut output)).await;
    }
}
