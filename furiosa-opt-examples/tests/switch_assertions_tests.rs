use furiosa_opt_examples::switch_assertions::{A, B, C, D, E, F};
use furiosa_opt_std::prelude::*;

mod alignment {
    use super::*;
    use furiosa_opt_examples::switch_assertions::alignment::*;

    #[tokio::test]
    async fn test_aligned_fetch_packet_i8() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(aligned_fetch_packet_i8, (&mut *ctx, &input, &mut output)).await;
    }
    #[tokio::test]
    async fn test_aligned_fetch_packet_bf16() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<bf16, m![A, B]>::from_vec(
            (0..<m![A, B]>::SIZE)
                .map(|x| bf16::from_f32(x as f32))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

        let mut output = HbmTensor::<bf16, m![1], m![A, B]>::new();

        launch(aligned_fetch_packet_bf16, (&mut *ctx, &input, &mut output)).await;
    }
}

pub mod packet {
    use super::*;
    use furiosa_opt_examples::switch_assertions::packet::*;

    #[tokio::test]
    async fn test_packet_padding_unchanged() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(packet_padding_unchanged, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_padding_added_in_switch() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(packet_padding_added_in_switch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_nested_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(packet_nested_padding, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_restructuring() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, C]>::from_vec((0..<m![A, C]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, C]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, C / 16, C % 16]>::new();

        launch(packet_restructuring, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(valid_padding, (&mut *ctx, &input, &mut output)).await;
    }
}

pub mod slice {
    use super::*;
    use furiosa_opt_examples::switch_assertions::slice::*;

    #[tokio::test]
    async fn test_valid_matching_slice_sizes() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, B]>::new();

        launch(valid_matching_slice_sizes, (&mut *ctx, &input, &mut output)).await;
    }
}

mod broadcast1 {
    use super::*;
    use furiosa_opt_examples::switch_assertions::broadcast1::*;

    #[tokio::test]
    async fn test_valid_basic() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B]>::new();

        launch(valid_basic, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_degenerate() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C % 4, B]>::new();

        launch(valid_degenerate, (&mut *ctx, &input, &mut output)).await;
    }
}

mod broadcast01 {
    use super::*;
    use furiosa_opt_examples::switch_assertions::broadcast01::*;

    #[tokio::test]
    async fn test_valid_only_slice1() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![B]>::from_vec((0..<m![B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
            .to_hbm::<m![1], m![B]>(&mut ctx.pdma)
            .await;

        let mut output = HbmTensor::<i8, m![1], m![F / 4, E / 4, B]>::new();

        launch(valid_only_slice1, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_with_time0() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 4, D % 4, A / 2, C / 2 % 2, A % 2, C % 2, B]>::new();

        launch(valid_with_time0, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_broadcast_with_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B]>::new();

        launch(valid_broadcast_with_padding, (&mut *ctx, &input, &mut output)).await;
    }
}

mod transpose {
    use super::*;
    use furiosa_opt_examples::switch_assertions::transpose::*;

    #[tokio::test]
    async fn test_valid_single_axis() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 64, C % 2, C / 2 % 32, A, B]>::new();

        launch(valid_single_axis, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_three_axes() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 128, C % 8, C / 8 % 16, A, B]>::new();

        launch(valid_three_axes, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_split_inner() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 16, C % 4, C / 4 % 4, A, B]>::new();

        launch(valid_split_inner, (&mut *ctx, &input, &mut output)).await;
    }
}

mod inter_transpose {
    use super::*;
    use furiosa_opt_examples::switch_assertions::inter_transpose::*;

    #[tokio::test]
    async fn test_valid() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![C / 32, A / 2 % 2, C % 16, A / 4, A % 2, C / 16 % 2, B]>::new();

        launch(valid, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_degenerate() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
                .await;

        let mut output = HbmTensor::<i8, m![1], m![A, C % 32, C / 32 % 8, B]>::new();

        launch(valid_degenerate, (&mut *ctx, &input, &mut output)).await;
    }
}
