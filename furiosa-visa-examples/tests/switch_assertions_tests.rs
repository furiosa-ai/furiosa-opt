use furiosa_visa_examples::switch_assertions::{A, B, C, D, E, F};
use furiosa_visa_std::prelude::*;

mod alignment {
    use super::*;
    use furiosa_visa_examples::switch_assertions::alignment::*;

    #[tokio::test]
    #[should_panic(expected = "Collect output packet must be exactly 32 bytes (one flit).")]
    async fn test_unaligned_fetch_packet_i4() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i4, m![A, B # 32]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init(i4::from_i32((x % 16) as i32)))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i4, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(unaligned_fetch_packet_i4, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_aligned_fetch_packet_i4() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i4, m![A, B # 64]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init(i4::from_i32((x % 16) as i32)))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 64]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i4, m![1], m![A, B # 64]>::from_addr(0x1000) };

        launch(aligned_fetch_packet_i4, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Collect output packet must be exactly 32 bytes (one flit).")]
    async fn test_unaligned_fetch_packet_i8() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B / 2]>::from_buf(
            (0..<m![A, B / 2]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B / 2]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B / 2]>::from_addr(0x1000) };

        launch(unaligned_fetch_packet_i8, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_aligned_fetch_packet_i8() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(aligned_fetch_packet_i8, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Collect output packet must be exactly 32 bytes (one flit).")]
    async fn test_unaligned_fetch_packet_bf16() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<bf16, m![A, B / 4]>::from_buf(
            (0..<m![A, B / 4]>::SIZE)
                .map(|x| Opt::Init(bf16::from_f32(x as f32)))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B / 4]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<bf16, m![1], m![A, B / 4]>::from_addr(0x1000) };

        launch(unaligned_fetch_packet_bf16, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_aligned_fetch_packet_bf16() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<bf16, m![A, B]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init(bf16::from_f32(x as f32)))
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
    use furiosa_visa_examples::switch_assertions::packet::*;

    #[tokio::test]
    async fn test_packet_padding_unchanged() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(packet_padding_unchanged, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_padding_added_in_switch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(packet_padding_added_in_switch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_nested_padding() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(packet_nested_padding, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_packet_restructuring() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, C]>::from_buf(
            (0..<m![A, C]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, C]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, C / 16, C % 16]>::from_addr(0x1000) };

        launch(packet_restructuring, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Collect packet mismatch.")]
    async fn test_packet_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(packet_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_padding() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(valid_padding, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Collect time mismatch.")]
    async fn test_collect_time_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<bf16, m![A, B]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init(bf16::from_f32(x as f32)))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<bf16, m![1], m![A, B % 16]>::from_addr(0x1000) };

        launch(collect_time_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Collect output packet must be exactly 32 bytes (one flit).")]
    async fn test_invalid_excessive_padding() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 64]>::from_buf(
            (0..<m![A, B]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 64]>::from_addr(0x1000) };

        launch(invalid_excessive_padding, (&mut *ctx, &input, &mut output)).await;
    }
}

pub mod slice {
    use super::*;
    use furiosa_visa_examples::switch_assertions::slice::*;

    #[tokio::test]
    async fn test_valid_matching_slice_sizes() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(valid_matching_slice_sizes, (&mut *ctx, &input, &mut output)).await;
    }
}

mod broadcast1 {
    use super::*;
    use furiosa_visa_examples::switch_assertions::broadcast1::*;

    #[tokio::test]
    async fn test_valid_basic() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B # 32]>::from_addr(0x1000) };

        launch(valid_basic, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_degenerate() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C % 4, B # 32]>::from_addr(0x1000) };

        launch(valid_degenerate, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "All dimensions must be greater than 0")]
    async fn test_invalid_slice1_zero() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, C % 4, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice1_zero, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "InSlice::SIZE must be divisible by (slice1 * slice0)")]
    async fn test_invalid_slice_size() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, C % 4, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice_size, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice must preserve slice2 from InSlice")]
    async fn test_invalid_slice2_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![D % 4, C % 64, A, C / 64, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice2_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice broadcast axes must be new axes")]
    async fn test_invalid_broadcast_axes_not_new() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, C % 4, A, C % 4, B # 32]>::from_addr(0x1000) };

        launch(invalid_broadcast_axes_not_new, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice must preserve slice0 from InSlice")]
    async fn test_invalid_slice0_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice0_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutTime does not match expected layout: [time0, slice1]")]
    async fn test_invalid_out_time() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, E % 4, B # 32]>::from_addr(0x1000) };

        launch(invalid_out_time, (&mut *ctx, &input, &mut output)).await;
    }
}

mod broadcast01 {
    use super::*;
    use furiosa_visa_examples::switch_assertions::broadcast01::*;

    #[tokio::test]
    async fn test_valid_only_slice1() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![B # 32]>::from_buf(
            (0..<m![B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![F / 4, E / 4, B # 32]>::from_addr(0x1000) };

        launch(valid_only_slice1, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_with_time0() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe {
            HbmTensor::<i8, m![1], m![C / 4, D % 4, A / 2, C / 2 % 2, A % 2, C % 2, B # 32]>::from_addr(0x1000)
        };

        launch(valid_with_time0, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_broadcast_with_padding() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(valid_broadcast_with_padding, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "All dimensions must be greater than 0")]
    async fn test_invalid_slice0_zero() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, C / 2 % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice0_zero, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "InSlice::SIZE must be divisible by (slice1 * slice0)")]
    async fn test_invalid_slice_size() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, C / 2 % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice_size, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutTime does not match expected layout")]
    async fn test_invalid_time_size() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, D % 4, A, C / 2 % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_time_size, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice must preserve slice2 from InSlice")]
    async fn test_invalid_slice2_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe {
            HbmTensor::<i8, m![1], m![C / 8, D % 8, A / 2, C / 2 % 2, A % 2, C % 2, B # 32]>::from_addr(0x1000)
        };

        launch(invalid_slice2_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice broadcast axes must be new axes")]
    async fn test_invalid_slice_axes_in_broadcast() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, C % 4, A, C / 2 % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice_axes_in_broadcast, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice broadcast axes must be new axes")]
    async fn test_invalid_time_axes_in_broadcast() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, A % 4, A, C / 2 % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_time_axes_in_broadcast, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutTime does not match expected layout")]
    async fn test_invalid_out_time() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 4, E % 4, A / 2, C / 2, A % 2, C % 2, B # 32]>::from_addr(0x1000) };

        launch(invalid_out_time, (&mut *ctx, &input, &mut output)).await;
    }
}

mod transpose {
    use super::*;
    use furiosa_visa_examples::switch_assertions::transpose::*;

    #[tokio::test]
    async fn test_valid_single_axis() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 64, C % 2, C / 2 % 32, A, B # 32]>::from_addr(0x1000) };

        launch(valid_single_axis, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_three_axes() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 128, C % 8, C / 8 % 16, A, B # 32]>::from_addr(0x1000) };

        launch(valid_three_axes, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_split_inner() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 16, C % 4, C / 4 % 4, A, B # 32]>::from_addr(0x1000) };

        launch(valid_split_inner, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Input and output time dimensions must have the same size")]
    async fn test_invalid_time_size() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![C / 64, C % 2, C / 2 % 32, D, B # 32]>::from_addr(0x1000) };

        launch(invalid_time_size, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Input and output time dimensions must match (excluding padding)")]
    async fn test_invalid_time_mapping() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output =
            unsafe { HbmTensor::<i8, m![1], m![C / 64, C % 2, C / 2 % 32, E % 8, B # 32]>::from_addr(0x1000) };

        launch(invalid_time_mapping, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice does not match expected layout")]
    async fn test_invalid_transpose_placement() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_transpose_placement, (&mut *ctx, &input, &mut output)).await;
    }
}

mod inter_transpose {
    use super::*;
    use furiosa_visa_examples::switch_assertions::inter_transpose::*;

    #[tokio::test]
    async fn test_valid() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe {
            HbmTensor::<i8, m![1], m![C / 32, A / 2 % 2, C % 16, A / 4, A % 2, C / 16 % 2, B # 32]>::from_addr(0x1000)
        };

        launch(valid, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    async fn test_valid_degenerate() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, C % 32, C / 32 % 8, B # 32]>::from_addr(0x1000) };

        launch(valid_degenerate, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "All dimensions must be greater than 0")]
    async fn test_invalid_time0() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_time0, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "InSlice::SIZE must be divisible by (slice1 * slice0)")]
    async fn test_invalid_dims() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_dims, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "InTime::SIZE must be divisible by (slice1 * time0)")]
    async fn test_invalid_time0_size() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_time0_size, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice must preserve slice2 from InSlice")]
    async fn test_invalid_slice2_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice2_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice must preserve slice0 from InSlice")]
    async fn test_invalid_slice0_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![B # 32]>::from_buf(
            (0..<m![B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![B # 32]>::from_addr(0x1000) };

        launch(invalid_slice0_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutSlice time1 must come from InTime")]
    async fn test_invalid_time1_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_time1_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutTime must preserve 'time2' from InTime")]
    async fn test_invalid_time2_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_time2_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutTime must preserve 'time0' from InTime")]
    async fn test_invalid_time0_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_time0_mismatch, (&mut *ctx, &input, &mut output)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutTime must preserve 'slice1' from InSlice")]
    async fn test_invalid_slice1_mismatch() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<i8, m![A, B # 32]>::from_buf(
            (0..<m![A, B # 32]>::SIZE)
                .map(|x| Opt::Init((x % 256) as i8))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![A, B # 32]>(&mut ctx.pdma, 0)
        .await;

        let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B # 32]>::from_addr(0x1000) };

        launch(invalid_slice1_mismatch, (&mut *ctx, &input, &mut output)).await;
    }
}
