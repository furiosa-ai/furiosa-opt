use furiosa_visa_examples::attention::compile_llama3_1_mlperf_latest_8pe_4chip_w8a16kv16_prefill_first_block_b1_s1024;
use furiosa_visa_std::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

#[tokio::test]
#[ignore = "incomplete kernel with todo!()"]
async fn test_compile_llama3_1_mlperf_latest_8pe_4chip_w8a16kv16_prefill_first_block_b1_s1024() {
    use furiosa_visa_examples::attention::{E, S, V};

    type Chip = m![1];

    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let input_ids = HostTensor::<i32, m![S]>::rand(&mut rng)
        .to_hbm::<Chip, m![S]>(&mut ctx.pdma, 0)
        .await;
    let embedding_table = HostTensor::<bf16, m![V, E]>::uninit()
        .to_hbm::<Chip, m![V, E]>(&mut ctx.pdma, 0x92487000)
        .await;
    let norm_weight = HostTensor::<f32, m![E]>::rand(&mut rng)
        .to_hbm::<Chip, m![E]>(&mut ctx.pdma, 0x92487000)
        .await;

    launch(
        compile_llama3_1_mlperf_latest_8pe_4chip_w8a16kv16_prefill_first_block_b1_s1024,
        (&mut *ctx, &input_ids, &embedding_table, &norm_weight),
    )
    .await;
}
