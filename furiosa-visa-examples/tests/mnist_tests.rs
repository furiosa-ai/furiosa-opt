use furiosa_visa_examples::mnist::{C, H, X, forward};
use furiosa_visa_std::prelude::*;
use safetensors::SafeTensors;

const MNIST: &[u8] = include_bytes!("../data/mnist/mnist.safetensors");

#[tokio::test]
#[cfg_attr(
    not(furiosa_opt),
    ignore = "fc1_bias_prepared's reshape-around-padding trips CPU-sim verify_transpose; run via `cargo furiosa-opt test`"
)]
async fn test_mnist() {
    let model = SafeTensors::deserialize(MNIST).unwrap();
    let mut ctx = Context::acquire();

    let w1 = HostTensor::<bf16, m![H, X]>::from_safetensors(&model.tensor("hw.fc1.weight").unwrap())
        .unwrap()
        .to_hbm(&mut ctx.pdma, 0x0000_0000)
        .await;
    let b1 = HostTensor::<bf16, m![H]>::from_safetensors(&model.tensor("fc1.bias").unwrap())
        .unwrap()
        .to_hbm(&mut ctx.pdma, 0x1000_0000)
        .await;
    let w2 = HostTensor::<bf16, m![C, H]>::from_safetensors(&model.tensor("hw.fc2.weight").unwrap())
        .unwrap()
        .to_hbm(&mut ctx.pdma, 0x2000_0000)
        .await;
    let b2 = HostTensor::<bf16, m![C]>::from_safetensors(&model.tensor("hw.fc2.bias").unwrap())
        .unwrap()
        .to_hbm(&mut ctx.pdma, 0x3000_0000)
        .await;

    for i in 0..10 {
        let img = HostTensor::<bf16, m![X]>::from_safetensors(&model.tensor(&format!("hw.image_{i}")).unwrap())
            .unwrap()
            .to_hbm(&mut ctx.pdma, 0x4000_0000)
            .await;

        let logits = launch(forward, (&mut *ctx, &img, &w1, &b1, &w2, &b2))
            .await
            .to_host::<m![C]>(&mut ctx.pdma)
            .await;

        let buf = logits.to_buf();
        let predicted = buf
            .iter()
            .take(10)
            .map(|opt| f32::from(opt.unwrap()))
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;

        let label_data = model.tensor(&format!("label_{i}")).unwrap().data();
        let expected = i32::from_le_bytes(label_data[..4].try_into().unwrap()) as usize;
        assert_eq!(predicted, expected, "image_{i}: expected {expected}, got {predicted}");
    }
}
