//! Virtual ISA program for test_compile_llama3_1_mlperf_latest_8pe_4chip_w8a16kv16_prefill_first_block_b1_s1024

#![expect(clippy::type_complexity)]

use furiosa_visa_std::prelude::*;

axes![V = 128256, S = 1024, E = 8192, X = 2, Y = 2];

type Chip = m![1];

#[device(chip = 1)]
pub fn compile_llama3_1_mlperf_latest_8pe_4chip_w8a16kv16_prefill_first_block_b1_s1024(
    ctx: &mut Context,
    input_ids_hbm: &HbmTensor<i32, Chip, m![S]>,
    embedding_table_hbm: &HbmTensor<bf16, Chip, m![V, E]>,
    norm_weight_hbm: &HbmTensor<f32, Chip, m![E]>,
) {
    let input_ids_sram_1 =
        input_ids_hbm.to_dm::<m![S / 512], m![S / 16 % 32, 1 # 8], m![S % 16]>(&mut ctx.tdma, 0x00010000);
    let input_ids_sram_1 = ctx
        .main
        .begin(input_ids_sram_1.view())
        .fetch::<i32, m![S / 2 % 8], m![S % 2]>()
        .collect::<m![1 # 8], m![S % 2 # 8]>()
        .commit::<m![1 # 8, S % 2]>(0x00010100);

    let input_ids_sram: DmTensor<i32, Chip, m![S / 512], m![S / 16 % 32, 1 # 8], m![S % 2]> = ctx
        .main
        .begin(input_ids_sram_1.view())
        .fetch::<i32, m![1], m![S]>()
        .commit(0x00010000);

    let input_ids_scaled_sram = ctx
        .main
        .begin(input_ids_sram.view())
        .fetch::<i32, m![1], m![S % 2]>()
        .collect::<m![1], m![S % 2 = 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, 16384) // 16384 = 8192(E) * 2(bf16 bytes). Computes byte offset for token lookup(byte offset of ith vocab inside V * E table).
        .vector_final()
        .commit::<m![S % 2]>(0x00010000);

    let input_ids_scaled_dram: HbmTensor<i32, Chip, m![S]> = ctx
        .main
        .begin(input_ids_scaled_sram.view())
        .fetch::<i32, m![1], m![S % 2]>()
        .collect::<m![S / 16 % 4, S / 2 % 8], m![S % 2 = 8]>()
        .commit::<m![S % 64]>(0x00010100)
        .to_hbm(&mut ctx.tdma, 0x936ac000);

    let embeddings_sram: DmTensor<bf16, Chip, m![S / 512], m![S / 128 % 4, E / 128], m![E % 128]> =
        embedding_table_hbm.dma_gather(&input_ids_scaled_dram, 0x00018000);

    let norm_weight_sram_0: DmTensor<f32, Chip, m![X], m![E / 128, 1 # 2], m![E % 64]> =
        norm_weight_hbm.to_dm(&mut ctx.tdma, 0x00010200);
    let norm_weight_sram_1: DmTensor<f32, Chip, m![X], m![E / 128, 1 # 2], m![1 # 2, E % 32]> = ctx
        .main
        .begin(norm_weight_sram_0.view())
        .fetch::<f32, m![E / 32 % 2], m![E % 32]>()
        .collect::<m![1 # 2], m![E % 32]>()
        .commit(0x00010000); // 0x00010000
    let _norm_weight_sram_2: DmTensor<f32, Chip, m![X], m![E / 128, 1 # 2], m![E % 32]> = ctx
        .main
        .begin(norm_weight_sram_1.view())
        .fetch::<f32, m![1], m![E % 32]>()
        .collect::<m![1], m![E % 32]>()
        .commit(0x001200);

    let embeddings_hbm: HbmTensor<bf16, m![1], m![S, E]> = embeddings_sram.to_hbm(&mut ctx.tdma, 0x936ad000);

    for i in 0..2 {
        let embedding_view = embeddings_hbm.view().tile::<m![S / 512], 1, m![1 # 2, S % 512, E]>(i);

        let embedding_view_sram_0: DmTensor<bf16, Chip, m![Y], m![E / 128, S / 128 % 4], m![S % 128, E % 128]> =
            embedding_view.to_dm(&mut ctx.tdma, 0);

        let _embedding_view_sram_1: DmTensor<
            bf16,
            Chip,
            m![Y],
            m![E / 128, S / 128 % 4],
            m![S / 128 % 4, S % 128, E % 32],
        > = ctx
            .main
            .begin(embedding_view_sram_0.view())
            .fetch::<bf16, m![S % 128, E / 32 % 4], m![E % 32]>()
            .collect::<m![S % 128, S / 128 % 4], m![E % 32]>()
            .commit(0);

        // TODO: Complete the function definition.
    }
}
