use furiosa_visa_examples::aligner_assertions::{A, B, C, E, F, R, T, U, V};
use furiosa_visa_std::prelude::*;

type Chip = m![1];

mod row_size {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::row_size::*;

    #[tokio::test]
    async fn test_valid_size_1() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![B]>::from_buf((0..<m![B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, 1 # 8]>::from_addr(0x1000) };

        launch(valid_size_1, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_2() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf = HostTensor::<i8, m![R / 4, B]>::from_buf(
            (0..<m![R / 4, B]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R / 4, B]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R / 4 # 8]>::from_addr(0x1000) };

        launch(valid_size_2, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_4() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf = HostTensor::<i8, m![R / 2, B]>::from_buf(
            (0..<m![R / 2, B]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R / 2, B]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R / 2 # 8]>::from_addr(0x1000) };

        launch(valid_size_4, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_size_8() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_size_8, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Row::SIZE must be 1, 2, 4, or 8, got 3")]
    async fn test_invalid_size_3() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(invalid_size_3, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Row::SIZE must be 1, 2, 4, or 8, got 16")]
    async fn test_invalid_size_16() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(invalid_size_16, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod cpacket_size {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::cpacket_size::*;

    #[tokio::test]
    async fn test_valid_size_64() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_size_64, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutPacket must be 64 bytes, got 32 bytes")]
    async fn test_invalid_size_32() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B / 2]>::from_buf(
            (0..<m![A, B / 2]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![A, B / 2]>(&mut ctx.pdma, 0)
        .await;

        let input_trf = HostTensor::<i8, m![R, B / 2]>::from_buf(
            (0..<m![R, B / 2]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, B / 2]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(invalid_size_32, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutPacket must be 64 bytes, got 128 bytes")]
    async fn test_invalid_size_128() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, C]>::from_buf((0..<m![A, C]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, C]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, C]>::from_buf((0..<m![R, C]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, C]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(invalid_size_128, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod cpacket_mapping {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::cpacket_mapping::*;

    #[tokio::test]
    async fn test_valid_one_collect_flit() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_buf((0..<m![R, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_one_collect_flit, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_two_collect_flits() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(valid_two_collect_flits, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Time does not divide OutTime")]
    async fn test_invalid_one_collect_flit_no_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(
            invalid_one_collect_flit_no_padding,
            (&mut *ctx, &input, &input_trf, &mut output),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "`align` packet mismatch")]
    async fn test_invalid_one_collect_flit_no_padding_reversed() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(
            invalid_one_collect_flit_no_padding_reversed,
            (&mut *ctx, &input, &input_trf, &mut output),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "`align` packet mismatch")]
    async fn test_invalid_mapping() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, R # 8]>::from_addr(0x1000) };

        launch(invalid_mapping, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }
}

mod time_broadcast {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::time_broadcast::*;

    #[tokio::test]
    async fn test_valid_single_tiling() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf = HostTensor::<i8, m![R, T, E]>::from_buf(
            (0..<m![R, T, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, T, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, T, R # 8]>::from_addr(0x1000) };

        launch(valid_single_tiling, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_double_tiling() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf = HostTensor::<i8, m![R, U, T, E]>::from_buf(
            (0..<m![R, U, T, E]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, U, T, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, U, T, R # 8]>::from_addr(0x1000) };

        launch(valid_double_tiling, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "tiling axes must be present in TRF")]
    async fn test_invalid_tiling_not_in_trf() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_buf((0..<m![R, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, T, R # 8]>::from_addr(0x1000) };

        launch(invalid_tiling_not_in_trf, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_transposed_tiling() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf = HostTensor::<i8, m![R, T, V, E]>::from_buf(
            (0..<m![R, T, V, E]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, T, V, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, V, T, R # 8]>::from_addr(0x1000) };

        launch(valid_transposed_tiling, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Time does not divide OutTime")]
    async fn test_invalid_time_mismatch() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, E]>::from_buf((0..<m![R, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, E]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![F, R # 8]>::from_addr(0x1000) };

        launch(invalid_time_mismatch, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Time axes are reordered in OutTime")]
    async fn test_invalid_swapped_time_axes() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, V, E]>::from_buf(
            (0..<m![A, V, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![A, V, E]>(&mut ctx.pdma, 0)
        .await;

        let input_trf = HostTensor::<i8, m![R, T, V, E]>::from_buf(
            (0..<m![R, T, V, E]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, T, V, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![V, A, T, R # 8]>::from_addr(0x1000) };

        launch(invalid_swapped_time_axes, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "tiling axes must be innermost in OutTime")]
    async fn test_invalid_tiling_not_innermost() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, E]>::from_buf((0..<m![A, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, E]>(&mut ctx.pdma, 0)
                .await;

        let input_trf = HostTensor::<i8, m![R, T, E]>::from_buf(
            (0..<m![R, T, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, T, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![T, A, R # 8]>::from_addr(0x1000) };

        launch(
            invalid_tiling_not_innermost,
            (&mut *ctx, &input, &input_trf, &mut output),
        )
        .await;
    }
}

mod trf_mapping {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::trf_mapping::*;

    #[tokio::test]
    async fn test_valid_mapping() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, 1 # 8]>::from_addr(0x1000) };

        launch(valid_mapping, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_unit_time_row() {
        let mut ctx = Context::acquire();

        let input_trf =
            HostTensor::<i8, m![E]>::from_buf((0..<m![E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![E]>(&mut ctx.pdma, 0)
                .await;

        launch(valid_unit_time_row, (&mut *ctx, &input_trf)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "`to_trf` row mismatch: time_outer != Row")]
    async fn test_invalid_row_mapping() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, 1 # 8]>::from_addr(0x1000) };

        launch(invalid_row_mapping, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "`to_trf` element mismatch: [time_inner, Packet] != Element")]
    async fn test_invalid_mapping() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let input_trf =
            HostTensor::<i8, m![R, B]>::from_buf((0..<m![R, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![R, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i32, Chip, m![A, 1 # 8]>::from_addr(0x1000) };

        launch(invalid_mapping, (&mut *ctx, &input, &input_trf, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Row::SIZE (4) does not divide Time::SIZE (6)")]
    async fn test_invalid_row_not_divisible_by_time() {
        let mut ctx = Context::acquire();

        let input_trf =
            HostTensor::<i8, m![F, E]>::from_buf((0..<m![F, E]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![F, E]>(&mut ctx.pdma, 0)
                .await;

        launch(invalid_row_not_divisible_by_time, (&mut *ctx, &input_trf)).await;
    }
}

mod trf_row_time {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::trf_row_time::*;

    #[tokio::test]
    #[should_panic(expected = "Row::SIZE must be <= Time::SIZE, got 4 > 2")]
    async fn test_invalid_row_exceeds_time() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A / 4, E]>::from_buf(
            (0..<m![A / 4, E]>::SIZE)
                .map(|x| Opt::Init(x as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![A / 4, E]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A / 4, E]>::from_addr(0x1000) };

        launch(invalid_row_exceeds_time, (&mut *ctx, &input, &mut output)).await;
    }
}

mod trf_size {
    use super::*;
    use furiosa_visa_examples::aligner_assertions::trf_size::*;

    #[tokio::test]
    async fn test_valid_to_trf_full() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

        launch(valid_to_trf_full, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_to_trf_half() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

        launch(valid_to_trf_half, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "TRF data (524288 bytes = 8 rows x 65536 bytes) exceeds register file capacity (65536 bytes for TrfAddress::Full)"
    )]
    async fn test_invalid_to_trf_full() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

        launch(invalid_to_trf_full, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "TRF data (65536 bytes = 8 rows x 8192 bytes) exceeds register file capacity (32768 bytes for TrfAddress::FirstHalf)"
    )]
    async fn test_invalid_to_trf_half() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| Opt::Init(x as i8)).collect::<Vec<_>>())
                .to_hbm::<Chip, m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, Chip, m![A, B]>::from_addr(0x1000) };

        launch(invalid_to_trf_half, (&mut *ctx, &input, &mut output)).await;
    }
}
