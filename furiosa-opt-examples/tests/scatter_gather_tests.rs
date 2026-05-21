//! Tests for scatter/gather DMA operations.
//!
//! Equivalent PyTorch (K = Scatter_key, D = Payload, C = Cache_len):
//! ```python
//! cache = torch.zeros(C, D, dtype=torch.bfloat16)  # [612, 128]
//! data = torch.randn(K, D, dtype=torch.bfloat16)   # [512, 128]
//! index = torch.arange(K)                          # [512]
//! cache[index] = data                              # index_put along dim 0
//! assert (cache[:K] != 0).all()
//! assert (cache[K:] == 0).all()
//! ```

use furiosa_opt_examples::scatter_gather::scatter_minimal;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_scatter_minimal() {
    use furiosa_opt_examples::scatter_gather::{C, D, K};

    let _ = env_logger::try_init();
    let mut ctx = Context::acquire();

    let data: HbmTensor<bf16, m![1], m![K, D]> = HostTensor::<bf16, m![K, D]>::from_buf(
        (1..=<m![K, D]>::SIZE)
            .map(|i| bf16::from_f32(i as f32))
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut ctx.pdma, 0x0)
    .await;

    // Convert row indices to byte offsets (scaled=true).
    let entry_bytes = (<m![D]>::SIZE * std::mem::size_of::<bf16>()) as i32;
    let index: HbmTensor<i32, m![1], m![K]> =
        HostTensor::<i32, m![K]>::from_buf((0..<m![K]>::SIZE as i32).map(|i| i * entry_bytes).collect::<Vec<_>>())
            .to_hbm(&mut ctx.pdma, 0x1000_0000)
            .await;

    let mut output: HbmTensor<bf16, m![1], m![C, D]> = HostTensor::<bf16, m![C, D]>::zero()
        .to_hbm(&mut ctx.pdma, 0x2000_0000)
        .await;

    launch(scatter_minimal, (&mut *ctx, &data, &index, &mut output)).await;

    let actual = output.to_host::<m![C, D]>(&mut ctx.pdma).await.to_buf();
    let expected = Tensor::<bf16, m![C, D]>::from_buf(
        (0..<m![C, D]>::SIZE)
            .map(|i| bf16::from_f32(if i < <m![K, D]>::SIZE { (i + 1) as f32 } else { 0.0 }))
            .collect::<Vec<_>>(),
    )
    .to_buf();

    assert_eq!(actual, expected);
}
