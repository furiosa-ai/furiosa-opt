use furiosa_opt_examples::mnist::{C, H, X, forward};
use furiosa_opt_std::prelude::*;
use safetensors::SafeTensors;

const MNIST: &[u8] = include_bytes!("../data/mnist/mnist.safetensors");

#[tokio::test]
async fn test_mnist() -> eyre::Result<()> {
    let model = SafeTensors::deserialize(MNIST)?;
    let mut device = Device::new(forward.topology())?;

    let w1 = HostTensor::<bf16, m![H, X]>::from_safetensors(&model.tensor("hw.fc1.weight")?)?
        .to_hbm(&mut device.pdma)
        .await?;
    let b1 = HostTensor::<bf16, m![H]>::from_safetensors(&model.tensor("fc1.bias")?)?
        .to_hbm(&mut device.pdma)
        .await?;
    let w2 = HostTensor::<bf16, m![C, H]>::from_safetensors(&model.tensor("hw.fc2.weight")?)?
        .to_hbm(&mut device.pdma)
        .await?;
    let b2 = HostTensor::<bf16, m![C]>::from_safetensors(&model.tensor("hw.fc2.bias")?)?
        .to_hbm(&mut device.pdma)
        .await?;

    for i in 0..10 {
        let img = HostTensor::<bf16, m![X]>::from_safetensors(&model.tensor(&format!("hw.image_{i}"))?)?
            .to_hbm(&mut device.pdma)
            .await?;

        let logits = launch(forward, (&mut device, &img, &w1, &b1, &w2, &b2))
            .await?
            .to_host::<m![C]>(&mut device.pdma)
            .await?;

        let buf = logits.into_vec();
        let predicted = buf
            .iter()
            .take(10)
            .map(|x| f32::from(*x))
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or_else(|| eyre::eyre!("MNIST model produced no logits"))?
            .0;

        let label_data = model.tensor(&format!("label_{i}"))?.data();
        let expected = i32::from_le_bytes(label_data[..4].try_into()?) as usize;
        assert_eq!(predicted, expected, "image_{i}: expected {expected}, got {predicted}");
    }

    Ok(())
}
