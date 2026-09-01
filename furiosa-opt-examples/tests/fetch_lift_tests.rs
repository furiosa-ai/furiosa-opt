//! Answer keys for the `fetch_lift` example kernels: what each base makes the read
//! return, on the CPU backend.

use furiosa_opt_examples::fetch_lift as kernels;
use furiosa_opt_examples::fetch_lift::{A, Beat, C, Code, G, H, HV, L, Oct, Part, Q, Red, Row, Step, V, Win, Word, Z};
use furiosa_opt_std::prelude::*;

fn value(i: usize) -> bf16 {
    bf16::from_f32((i % 251) as f32)
}

fn byte(i: usize) -> i8 {
    (i % 127) as i8
}

fn code(i: usize) -> f4e2m1 {
    f4e2m1::from_bits((i % 16) as u8)
}

fn word(i: usize) -> i32 {
    (i % 251) as i32
}

const E2M1_F32: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn ramp(len: usize) -> Vec<bf16> {
    (0..len).map(value).collect()
}

/// Generates VISA tests from input generators and per-element answer keys.
macro_rules! visa_answer_tests {
    ($(
        $(#[$attr:meta])*
        $kernel:ident: $chip:ty, $ramp:path, $in:ty => $out:ty = |$i:ident| $cell:expr;
    )*) => {
        $(
            $(#[$attr])*
            #[tokio::test]
            async fn $kernel() {
                let mut ctx = Context::acquire();

                let input = HostTensor::<_, $in>::from_vec(
                    (0..<$in>::SIZE).map($ramp).collect::<Vec<_>>(),
                )
                .to_hbm::<$chip, $in>(&mut ctx.pdma)
                .await;

                let output = launch(kernels::$kernel, (&mut *ctx, &input)).await;

                let expected = (0..<$out>::SIZE).map(|$i: usize| $cell).collect::<Vec<_>>();
                assert_eq!(
                    output.to_host::<$out>(&mut ctx.pdma).await.into_inner(),
                    Tensor::<_, $out, CurrentBackend>::from_vec(expected)
                );
            }
        )*
    };
}

visa_answer_tests! {
    fetch_slice_lift_axis: m![1], value, m![A, H, V] => m![A, H, V] = |i| value(i);

    fetch_sub_slice_lift_axis: m![1], value, m![A, H, V] => m![A, H, V] = |i| value(i);

    fetch_slice_lift_digit: m![1], value, m![A, Q, V] => m![A, Q, V] = |i| value(i);

    fetch_cluster_lift_axis: m![1], value, m![A, H, V] => m![H, A, V] = |i| {
        let (h, a, v) = (i / (128 * 16), (i / 16) % 128, i % 16);
        value(a * 32 + h * 16 + v)
    };

    fetch_sub_cluster_lift_axis: m![1], value, m![A, H, V] => m![H, A, V] = |i| {
        let (h, a, v) = (i / (128 * 16), (i / 16) % 128, i % 16);
        value(a * 32 + h * 16 + v)
    };

    fetch_chip_lift_axis: m![4], value, m![A, Q, V] => m![Q, A, V] = |i| {
        let (q, a, v) = (i / (128 * 16), (i / 16) % 128, i % 16);
        value(a * 64 + q * 16 + v)
    };

    fetch_sub_chip_lift_axis: m![4], value, m![A, Q, V] => m![Q, A, V] = |i| {
        let (q, a, v) = (i / (128 * 16), (i / 16) % 128, i % 16);
        value(a * 64 + q * 16 + v)
    };

    fetch_slice_lift_broadcasts: m![1], value, m![Row, H, G, V] => m![H, Row, G, V] = |i| {
        let (h, row, g, v) = (i / (64 * 2 * 16), (i / (2 * 16)) % 64, (i / 16) % 2, i % 16);
        value(row * 64 + h * 32 + g * 16 + v)
    };

    fetch_lift_every_dimension: m![4], value, m![A, C, L, H, V] => m![C, L, A, H, V] = |i| {
        let (c, l, a, h, v) = (
            i / (2 * 128 * 2 * 16),
            (i / (128 * 2 * 16)) % 2,
            (i / (2 * 16)) % 128,
            (i / 16) % 2,
            i % 16,
        );
        value(a * 256 + c * 64 + l * 32 + h * 16 + v)
    };

    fetch_sub_slice_lift_custom_switch: m![1], value, m![Part, H, Beat, V]
        => m![Part, H, Step, Beat, V] = |i| {
        let (part, h, beat, v) = (
            i / (2 * 4 * 8 * 16),
            (i / (4 * 8 * 16)) % 2,
            (i / 16) % 8,
            i % 16,
        );
        value(((part * 2 + h) * 8 + beat) * 16 + v)
    };

    fetch_slice_lift_dynamic_view: m![1], value, m![A, H, Win] => m![A, H, Win] = |i| value(i);
    fetch_sub_slice_lift_dynamic_view: m![1], value, m![A, H, Win] => m![A, H, Win] = |i| value(i);

    fetch_slice_lift_reshaped_broadcast: m![1], value, m![A, HV] => m![A, H, V] = |i| value(i);

    fetch_slice_lift_reshaped_named_broadcast: m![1], value, m![A, H, V] => m![A, H, V] = |i| value(i);

    fetch_slice_lift_then_cast: m![1], byte, m![A, H, Oct] => m![A, H, Oct] = |i| i32::from(byte(i));

    fetch_slice_lift_table_lookup: m![1], code, m![A, H, Code] => m![A, H, Code] = |i| E2M1_F32[i % 16];

    fetch_cluster_lift_inter_slice_reduce: m![1], word, m![Row, Red, Z, Oct]
        => m![Z, Row, Oct] = |i| {
        let (z, row, oct) = (i / (64 * 8), (i / 8) % 64, i % 8);
        (0..4)
            .map(|red| word(((row * 4 + red) * 2 + z) * 8 + oct))
            .sum::<i32>()
    };
}

macro_rules! vrf_answer_tests {
    ($($kernel:ident;)*) => {
        $(
            #[tokio::test]
            async fn $kernel() {
                let mut ctx = Context::acquire();
                let input = HostTensor::<i32, m![A, H, Word]>::from_vec(
                    (0..<m![A, H, Word]>::SIZE).map(word).collect::<Vec<_>>(),
                )
                .to_hbm::<m![1], m![A, H, Word]>(&mut ctx.pdma)
                .await;
                let addend = HostTensor::<i32, m![A, H, Word]>::from_vec(
                    (0..<m![A, H, Word]>::SIZE)
                        .map(|i| (i % 97) as i32)
                        .collect::<Vec<_>>(),
                )
                .to_hbm::<m![1], m![A, H, Word]>(&mut ctx.pdma)
                .await;

                let output = launch(kernels::$kernel, (&mut *ctx, &input, &addend)).await;
                let expected = (0..<m![A, H, Word]>::SIZE)
                    .map(|i| word(i) + (i % 97) as i32)
                    .collect::<Vec<_>>();

                assert_eq!(
                    output.to_host::<m![A, H, Word]>(&mut ctx.pdma).await.into_inner(),
                    Tensor::<_, m![A, H, Word], CurrentBackend>::from_vec(expected)
                );
            }
        )*
    };
}

vrf_answer_tests! {
    fetch_sub_slice_lift_to_vrf;
    fetch_sub_slice_lift_bare_to_vrf;
}

mod mix {
    use furiosa_opt_examples::fetch_lift::mix as kernels;
    use furiosa_opt_examples::fetch_lift::mix::{K, P128, P256, R, S, T, U8, U32};

    use super::*;

    fn input_index(output_index: usize, outer: usize, live: usize, slice: usize, packet: usize) -> usize {
        let packet_index = output_index % packet;
        let output_index = output_index / packet;
        let slice_index = output_index % slice;
        let output_index = output_index / slice;
        let live_index = output_index % live;
        let outer_index = output_index / live;

        (((live_index * outer + outer_index) * slice + slice_index) * packet) + packet_index
    }

    visa_answer_tests! {
        mix_chip_plain: m![4], byte, m![P256, R, U32] => m![R, P256, U32] = |i| {
            byte(input_index(i, 4, 256, 1, 32))
        };
        mix_chip_cast: m![4], byte, m![P256, R, U8] => m![R, P256, U8] = |i| {
            i32::from(byte(input_index(i, 4, 256, 1, 8)))
        };
        mix_chip_lookup: m![4], code, m![P256, R, K] => m![R, P256, K] = |i| {
            E2M1_F32[input_index(i, 4, 256, 1, 16) % 16]
        };

        mix_cluster_plain: m![1], byte, m![P256, S, U32] => m![S, P256, U32] = |i| {
            byte(input_index(i, 2, 256, 1, 32))
        };
        mix_cluster_cast: m![1], byte, m![P256, S, U8] => m![S, P256, U8] = |i| {
            i32::from(byte(input_index(i, 2, 256, 1, 8)))
        };
        mix_cluster_lookup: m![1], code, m![P256, S, K] => m![S, P256, K] = |i| {
            E2M1_F32[input_index(i, 2, 256, 1, 16) % 16]
        };

        mix_slice_plain: m![1], byte, m![P128, T, U32] => m![P128, T, U32] = |i| {
            byte(input_index(i, 1, 128, 2, 32))
        };
        mix_slice_cast: m![1], byte, m![P128, T, U8] => m![P128, T, U8] = |i| {
            i32::from(byte(input_index(i, 1, 128, 2, 8)))
        };
        mix_slice_lookup: m![1], code, m![P128, T, K] => m![P128, T, K] = |i| {
            E2M1_F32[input_index(i, 1, 128, 2, 16) % 16]
        };

        mix_chip_cluster_plain: m![4], byte, m![P256, R, S, U32] => m![R, S, P256, U32] = |i| {
            byte(input_index(i, 8, 256, 1, 32))
        };
        mix_chip_cluster_cast: m![4], byte, m![P256, R, S, U8] => m![R, S, P256, U8] = |i| {
            i32::from(byte(input_index(i, 8, 256, 1, 8)))
        };
        mix_chip_cluster_lookup: m![4], code, m![P256, R, S, K] => m![R, S, P256, K] = |i| {
            E2M1_F32[input_index(i, 8, 256, 1, 16) % 16]
        };

        mix_chip_slice_plain: m![4], byte, m![P128, R, T, U32] => m![R, P128, T, U32] = |i| {
            byte(input_index(i, 4, 128, 2, 32))
        };
        mix_chip_slice_cast: m![4], byte, m![P128, R, T, U8] => m![R, P128, T, U8] = |i| {
            i32::from(byte(input_index(i, 4, 128, 2, 8)))
        };
        mix_chip_slice_lookup: m![4], code, m![P128, R, T, K] => m![R, P128, T, K] = |i| {
            E2M1_F32[input_index(i, 4, 128, 2, 16) % 16]
        };

        mix_cluster_slice_plain: m![1], byte, m![P128, S, T, U32] => m![S, P128, T, U32] = |i| {
            byte(input_index(i, 2, 128, 2, 32))
        };
        mix_cluster_slice_cast: m![1], byte, m![P128, S, T, U8] => m![S, P128, T, U8] = |i| {
            i32::from(byte(input_index(i, 2, 128, 2, 8)))
        };
        mix_cluster_slice_lookup: m![1], code, m![P128, S, T, K] => m![S, P128, T, K] = |i| {
            E2M1_F32[input_index(i, 2, 128, 2, 16) % 16]
        };

        mix_every_plain: m![4], byte, m![P128, R, S, T, U32] => m![R, S, P128, T, U32] = |i| {
            byte(input_index(i, 8, 128, 2, 32))
        };
        mix_every_cast: m![4], byte, m![P128, R, S, T, U8] => m![R, S, P128, T, U8] = |i| {
            i32::from(byte(input_index(i, 8, 128, 2, 8)))
        };
        mix_every_lookup: m![4], code, m![P128, R, S, T, K] => m![R, S, P128, T, K] = |i| {
            E2M1_F32[input_index(i, 8, 128, 2, 16) % 16]
        };
    }
}

mod zero_point {
    use furiosa_opt_examples::fetch_lift::{Act, Dot, Out, P128, Zp, lift_zero_point_sub_contract};

    use super::*;

    #[tokio::test]
    async fn lift_zero_point_sub_contract_matches_dot_product() {
        let mut ctx = Context::acquire();
        let input_values = (0..<m![P128, Zp, Act, Dot]>::SIZE)
            .map(|i| (i % 31) as i8)
            .collect::<Vec<_>>();
        let weight_values = (0..<m![P128, Zp, Out, Dot]>::SIZE)
            .map(|i| (i % 7) as i8 - 3)
            .collect::<Vec<_>>();
        let input = HostTensor::<i8, m![P128, Zp, Act, Dot]>::from_vec(input_values.clone())
            .to_hbm::<m![1], m![P128, Zp, Act, Dot]>(&mut ctx.pdma)
            .await;
        let weight = HostTensor::<i8, m![P128, Zp, Out, Dot]>::from_vec(weight_values.clone())
            .to_hbm::<m![1], m![P128, Zp, Out, Dot]>(&mut ctx.pdma)
            .await;

        let output = launch(lift_zero_point_sub_contract, (&mut *ctx, &input, &weight)).await;
        let expected = (0..<m![P128, Zp, Act, Out]>::SIZE)
            .map(|i| {
                let (p, z, act, out) = (i / (2 * 8 * 8), (i / (8 * 8)) % 2, (i / 8) % 8, i % 8);
                (0..32)
                    .map(|red| {
                        let input_index = (((p * 2 + z) * 8 + act) * 32) + red;
                        let weight_index = (((p * 2 + z) * 8 + out) * 32) + red;
                        (i32::from(input_values[input_index]) - 3) * i32::from(weight_values[weight_index])
                    })
                    .sum::<i32>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            output
                .to_host::<m![P128, Zp, Act, Out]>(&mut ctx.pdma)
                .await
                .into_inner(),
            Tensor::<_, m![P128, Zp, Act, Out], CurrentBackend>::from_vec(expected)
        );
    }
}

mod trf {
    use furiosa_opt_examples::fetch_lift::trf::{Act, Col, Dot, Grp, Row, lift_to_trf_contract};

    use super::*;

    #[tokio::test]
    async fn lift_to_trf_contract_matches_dot_product() {
        let mut ctx = Context::acquire();
        let act_values = (0..<m![Row, Grp, Act, Dot]>::SIZE)
            .map(|i| (i % 31) as i8)
            .collect::<Vec<_>>();
        let weight_values = (0..<m![Row, Grp, Col, Dot]>::SIZE)
            .map(|i| (i % 7) as i8 - 3)
            .collect::<Vec<_>>();
        let act = HostTensor::<i8, m![Row, Grp, Act, Dot]>::from_vec(act_values.clone())
            .to_hbm::<m![1], m![Row, Grp, Act, Dot]>(&mut ctx.pdma)
            .await;
        let weight = HostTensor::<i8, m![Row, Grp, Col, Dot]>::from_vec(weight_values.clone())
            .to_hbm::<m![1], m![Row, Grp, Col, Dot]>(&mut ctx.pdma)
            .await;

        let output = launch(lift_to_trf_contract, (&mut *ctx, &act, &weight)).await;
        let expected = (0..<m![Row, Grp, Act, Col]>::SIZE)
            .map(|i| {
                let (row, group, act, col) = (i / (2 * 8 * 8), (i / (8 * 8)) % 2, (i / 8) % 8, i % 8);
                (0..32)
                    .map(|dot| {
                        let act_index = (((row * 2 + group) * 8 + act) * 32) + dot;
                        let weight_index = (((row * 2 + group) * 8 + col) * 32) + dot;
                        i32::from(act_values[act_index]) * i32::from(weight_values[weight_index])
                    })
                    .sum::<i32>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            output
                .to_host::<m![Row, Grp, Act, Col]>(&mut ctx.pdma)
                .await
                .into_inner(),
            Tensor::<_, m![Row, Grp, Act, Col], CurrentBackend>::from_vec(expected)
        );
    }
}

#[tokio::test]
async fn reshaped_without_broadcast_uses_the_cpu_broadcast_origin() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<bf16, m![A, G, H, V]>::from_vec(ramp(<m![A, G, H, V]>::SIZE))
        .to_hbm::<m![1], m![A, G, H, V]>(&mut ctx.pdma)
        .await;

    let output = launch(
        kernels::reshape::fetch_slice_lift_reshaped_without_broadcast,
        (&mut *ctx, &input),
    )
    .await;
    let expected = (0..<m![A, H, V]>::SIZE)
        .map(|i| {
            let (a, h, v) = (i / (2 * 16), (i / 16) % 2, i % 16);
            value(a * 64 + h * 16 + v)
        })
        .collect::<Vec<_>>();

    assert_eq!(
        output.to_host::<m![A, H, V]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, H, V], CurrentBackend>::from_vec(expected)
    );
}
