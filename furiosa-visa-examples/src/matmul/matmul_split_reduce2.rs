//! Matrix multiplication of size [64, 1024] x [1024, 128] -> [64, 128] with split accumulation pattern.
//! This demonstrates the `test_compile_dfg_with_loop_and_index_access_ref` example in virtual ISA.

use furiosa_visa_std::prelude::*;

axes![M = 64, K = 1024, N = 128, I = 2];

#[device(chip = 1)]
pub fn matmul_with_split_reduce2(
    ctx: &mut Context,
    a: &HbmTensor<bf16, m![1], m![M, K]>,
    b: &HbmTensor<bf16, m![1], m![K, N]>,
    acc_zero: &HbmTensor<bf16, m![1], m![M, N]>,
) -> HbmTensor<bf16, m![1], m![M, N]> {
    type Chip = m![1];
    type Cluster = m![1 # 2];

    let mut acc: DmTensor<bf16, Chip, Cluster, m![1 # 4, M], m![N]> = acc_zero.to_dm(&mut ctx.tdma, 0x0000_0000);

    for i in 0..4 {
        let t87: DmTensor<bf16, Chip, Cluster, m![1 # 4, M], m![K % 128]> = a
            .view()
            .tile::<m![K / 128], 1, m![M, 1 # 8, K % 128]>(i * 2)
            .to_dm(&mut ctx.tdma, 0x0000_0100);

        let t95: DmTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, M % 8], m![K % 128 / 16, K % 128 % 16]> =
            unsafe { t87.reshape() };
        let t96: DmTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, K % 128 / 16], m![M % 8, K % 128 % 16]> = ctx
            .main
            .begin(t95.view())
            .fetch::<bf16, m![K % 128 / 16], m![K % 128 % 16]>()
            .switch::<m![1 # 4, M / 8, K % 128 / 16], m![M % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 1,
                time0: 1,
            })
            .collect::<m![M % 8], m![K % 128 % 16]>()
            .commit(0x0000_0200);
        let t72: DmTensor<bf16, Chip, Cluster, m![1 # 4, K % 128 / 2], m![K % 128 % 2, N]> = b
            .view()
            .tile::<m![K / 128], 1, m![1 # 8, K % 128, N]>(i * 2)
            .to_dm(&mut ctx.tdma, 0x0000_0300);
        let t89: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![1 # 4, M / 8, K % 128 / 16],
            m![M % 8, K % 128 % 16 / 2, K % 128 % 16 % 2],
        > = unsafe { t96.reshape() };
        let t94: TrfTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, K % 128 / 16], m![M % 8], m![K % 128 % 16]> = ctx
            .sub
            .begin(t89.view())
            .fetch::<bf16, m![M % 8], m![K % 128 % 16]>()
            .collect::<m![M % 8], m![K % 128 % 16]>()
            .to_trf(TrfAddress::FirstHalf);
        let t88: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![1 # 4, K % 128 / 2 / 8, K % 128 / 2 % 8],
            m![K % 128 % 2, N / 32, N % 32],
        > = unsafe { t72.reshape() };
        let t90: DmTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, M % 8], m![N / 32, N % 32]> = ctx
            .main
            .begin(t88.view())
            .fetch::<bf16, m![N / 32, K % 128 % 2], m![N % 32]>()
            .switch::<m![1 # 4, M / 8, K % 128 / 16], m![N / 32, K % 128 % 2, K % 128 / 2 % 8]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 8, slice0: 8 },
            )
            .collect::<m![N / 32, K % 128 % 2, K % 128 / 2 % 8, N % 32 / 16], m![N % 32 % 16]>()
            .align::<m![N / 32, K % 128 % 2, K % 128 / 2 % 8], m![N % 32], _, _>(&t94)
            .contract::<m![N % 32]>()
            .accumulate::<m![N / 32, M % 8, N % 32 / 8], m![N % 32 % 8]>(AccumulationKind::Sequential)
            .vector_init()
            .vector_inter_slice_reduce::<m![1 # 4, M / 8, M % 8], m![N / 32, N % 32 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![N % 32 % 8 # 16]>()
            .commit(0x0000_0100);

        let t108: DmTensor<bf16, Chip, Cluster, m![1 # 4, M], m![K % 128]> = a
            .view()
            .tile::<m![K / 128], 1, m![M, 1 # 8, K % 128]>(1 + i * 2)
            .to_dm(&mut ctx.tdma, 0x0000_0200);
        let t116: DmTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, M % 8], m![K % 128 / 16, K % 128 % 16]> =
            unsafe { t108.reshape() };
        let t103: DmTensor<bf16, Chip, Cluster, m![1 # 4, K % 128 / 2], m![K % 128 % 2, N]> = b
            .view()
            .tile::<m![K / 128], 1, m![1 # 8, K % 128, N]>(1 + i * 2)
            .to_dm(&mut ctx.tdma, 0x0000_0500);
        let t109: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![1 # 4, K % 128 / 2 / 8, K % 128 / 2 % 8],
            m![K % 128 % 2, N / 32, N % 32],
        > = unsafe { t103.reshape() };

        let t80: DmTensor<bf16, Chip, Cluster, m![1 # 4, M], m![N]> = unsafe { t90.reshape() };

        let t117: DmTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, K % 128 / 16], m![M % 8, K % 128 % 16]> = ctx
            .main
            .begin(t116.view())
            .fetch::<bf16, m![K % 128 / 16], m![K % 128 % 16]>()
            .switch::<m![1 # 4, M / 8, K % 128 / 16], m![M % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 1,
                time0: 1,
            })
            .collect::<m![M % 8], m![K % 128 % 16]>()
            .commit(0x0000_0300);
        let t110: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![1 # 4, M / 8, K % 128 / 16],
            m![M % 8, K % 128 % 16 / 2, K % 128 % 16 % 2],
        > = unsafe { t117.reshape() };

        let t81: DmTensor<bf16, Chip, Cluster, m![1 # 4, M], m![N]> = ctx
            .main
            .begin_interleaved::<I, _, _, _, _, _>(acc.view(), t80.view())
            .fetch::<f32, m![I, N / 8], m![N % 8]>()
            .collect::<m![I, N / 8], m![N % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<I, m![1 # 2, N / 8], m![N / 8]>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .cast::<bf16, m![N % 8 # 16]>()
            .commit(0x0000_0200);

        let t115: TrfTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, K % 128 / 16], m![M % 8], m![K % 128 % 16]> = ctx
            .sub
            .begin(t110.view())
            .fetch::<bf16, m![M % 8], m![K % 128 % 16]>()
            .collect::<m![M % 8], m![K % 128 % 16]>()
            .to_trf(TrfAddress::SecondHalf);

        let t111: DmTensor<bf16, Chip, Cluster, m![1 # 4, M / 8, M % 8], m![N / 32, N % 32]> = ctx
            .main
            .begin(t109.view())
            .fetch::<bf16, m![N / 32, K % 128 % 2], m![N % 32]>()
            .switch::<m![1 # 4, M / 8, K % 128 / 16], m![N / 32, K % 128 % 2, K % 128 / 2 % 8]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 8, slice0: 8 },
            )
            .collect::<m![N / 32, K % 128 % 2, K % 128 / 2 % 8, N % 32 / 16], m![N % 32 % 16]>()
            .align::<m![N / 32, K % 128 % 2, K % 128 / 2 % 8], m![N % 32], _, _>(&t115)
            .contract::<m![N % 32]>()
            .accumulate::<m![N / 32, M % 8, N % 32 / 8], m![N % 32 % 8]>(AccumulationKind::Sequential)
            .vector_init()
            .vector_inter_slice_reduce::<m![1 # 4, M / 8, M % 8], m![N / 32, N % 32 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![N % 32 % 8 # 16]>()
            .commit(0x0000_0100);
        let t105: DmTensor<bf16, Chip, Cluster, m![1 # 4, M], m![N]> = unsafe { t111.reshape() };

        acc = ctx
            .main
            .begin_interleaved::<I, _, _, _, _, _>(t81.view(), t105.view())
            .fetch::<f32, m![I, N / 8], m![N % 8]>()
            .collect::<m![I, N / 8], m![N % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<I, m![1 # 2, N / 8], m![N / 8]>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .cast::<bf16, m![N % 8 # 16]>()
            .commit(0x0000_0000);
    }

    acc.to_hbm(&mut ctx.tdma, 0x3000)
}
