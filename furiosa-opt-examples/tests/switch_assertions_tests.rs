use furiosa_opt_examples::switch_assertions::{A, B, C, D, E, F};
use furiosa_opt_std::prelude::*;

mod alignment {
    use super::*;
    use furiosa_opt_examples::switch_assertions::alignment::*;

    // Why buf-skipped: input mapping `m![A, B # 64]` has real padding (B=32, padded to 64) so
    // the host buffer holds `Opt::Uninit` slots. `from_opt_buf` is Simulation/Typecheck-only —
    // Npu / Emulation's native `Vec<D>` DMA staging has no `Opt::Uninit` representation, so this
    // test cannot even compile under those cfgs (`cfg_attr(.., ignore)` wouldn't be enough).
    #[cfg(not(backend = "npu"))]
    #[tokio::test]
    async fn test_aligned_fetch_packet_i4() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i4, m![A, B # 64]>::from_opt_buf(
            (0..<m![A, B # 64]>::SIZE)
                .map(|i| {
                    if i % 64 < <B>::SIZE {
                        let real = (i / 64) * <B>::SIZE + (i % 64);
                        Opt::Init(i4::from_i32((real % 16) as i32))
                    } else {
                        Opt::Uninit
                    }
                })
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 64]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i4, m![1], m![A, B # 64]>::from_addr(0x1000) };

        launch(aligned_fetch_packet_i4, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_aligned_fetch_packet_i8() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

        launch(aligned_fetch_packet_i8, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_aligned_fetch_packet_bf16() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<bf16, m![A, B]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| bf16::from_f32(x as f32))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<bf16, m![1], m![A, B]>::from_addr(0x1000) };

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
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

        launch(packet_padding_unchanged, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_padding_added_in_switch() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

        launch(packet_padding_added_in_switch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_nested_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

        launch(packet_nested_padding, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_restructuring() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, C]>::from_buf((0..<m![A, C]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, C]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, C / 16, C % 16]>::from_addr(0x1000) };

        launch(packet_restructuring, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

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
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x1000) };

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
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B]>::from_addr(0x1000) };

        launch(valid_basic, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_degenerate() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C % 4, B]>::from_addr(0x1000) };

        launch(valid_degenerate, (&mut *ctx, &input, &mut output)).await;
    }
}

mod broadcast01 {
    use super::*;
    use furiosa_opt_examples::switch_assertions::broadcast01::*;

    #[tokio::test]
    async fn test_valid_only_slice1() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![B]>::from_buf((0..<m![B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
            .to_hbm::<m![1], m![B]>(&mut ctx.pdma, 0)
            .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![F / 4, E / 4, B]>::from_addr(0x1000) };

        launch(valid_only_slice1, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_with_time0() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A / 2, C / 2 % 2, A % 2, C % 2, B]>::from_addr(0x1000) };

        launch(valid_with_time0, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_broadcast_with_padding() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B]>::from_addr(0x1000) };

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
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 64, C % 2, C / 2 % 32, A, B]>::from_addr(0x1000) };

        launch(valid_single_axis, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_three_axes() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 128, C % 8, C / 8 % 16, A, B]>::from_addr(0x1000) };

        launch(valid_three_axes, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_split_inner() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 16, C % 4, C / 4 % 4, A, B]>::from_addr(0x1000) };

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
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe {
            HbmTensor::<i8, m![1], m![C / 32, A / 2 % 2, C % 16, A / 4, A % 2, C / 16 % 2, B]>::from_addr(0x1000)
        };

        launch(valid, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_degenerate() {
        let mut ctx = Context::acquire();

        let input =
            HostTensor::<i8, m![A, B]>::from_buf((0..<m![A, B]>::SIZE).map(|x| (x % 256) as i8).collect::<Vec<_>>())
                .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
                .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, C % 32, C / 32 % 8, B]>::from_addr(0x1000) };

        launch(valid_degenerate, (&mut *ctx, &input, &mut output)).await;
    }
}
