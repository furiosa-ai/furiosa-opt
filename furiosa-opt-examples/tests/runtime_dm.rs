use furiosa_opt_examples::runtime_dm::{E, S, conditional_write, increment, loop_read, readback, stage};
use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Dm = DmTensor<i32, Chip, m![1], m![S], m![E]>;

#[tokio::test]
async fn residents_survive_calls() {
    let mut device = Device::new(stage.topology()).unwrap();
    let values: Vec<i32> = (0..512).collect();
    let input = HostTensor::<i32, m![S, E]>::from_vec(values.clone())
        .to_hbm::<Chip, m![S, E]>(&mut device.pdma)
        .await
        .unwrap();
    let mut first = Dm::alloc(&device).unwrap();
    let mut second = Dm::alloc(&device).unwrap();
    device.launch(stage, (&input, &mut first)).await.unwrap();
    device.launch(conditional_write, (&input, &mut second)).await.unwrap();
    let output = device.launch(readback, (&second,)).await.unwrap();
    assert_eq!(
        output.to_host::<m![S, E]>(&mut device.pdma).await.unwrap().into_vec(),
        values
    );
    device.launch(increment, (&first, &mut second)).await.unwrap();
    let mut output = device.launch(readback, (&second,)).await.unwrap();
    let expected: Vec<_> = values.iter().map(|value| value + 1).collect();
    assert_eq!(
        output.to_host::<m![S, E]>(&mut device.pdma).await.unwrap().into_vec(),
        expected
    );
    device.launch(loop_read, (&first, &mut output)).await.unwrap();
    assert_eq!(
        output.to_host::<m![S, E]>(&mut device.pdma).await.unwrap().into_vec(),
        values
    );
}
