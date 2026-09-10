use furiosa_opt_examples::param::{
    A, B, Inputs, Layer, Layers, Model, Weight, array_passthrough, borrowed_model_layer_loop,
    borrowed_model_passthrough, explicit_unit, local_struct_passthrough, struct_passthrough, tuple_passthrough,
};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn tuple_param_passes() {
    let mut device = Device::new(tuple_passthrough.topology()).unwrap();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    let out = launch(tuple_passthrough, (&mut device, (&input,))).await.unwrap();

    assert_eq!(
        out.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn tuple_param_reuses_output() {
    let mut device = Device::new(tuple_passthrough.topology()).unwrap();
    let first = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let second = first.iter().map(|x| x.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first)
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let mut output = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    launch(tuple_passthrough, (&mut device, (&first_input,)))
        .output(&mut output)
        .await
        .unwrap();
    launch(tuple_passthrough, (&mut device, (&second_input,)))
        .output(&mut output)
        .await
        .unwrap();

    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second)
    );
}

#[tokio::test]
async fn struct_param_passes() {
    let mut device = Device::new(struct_passthrough.topology()).unwrap();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    let out = launch(struct_passthrough, (&mut device, Inputs { x: &input }))
        .await
        .unwrap();

    assert_eq!(
        out.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn array_param_passes() {
    let mut device = Device::new(array_passthrough.topology()).unwrap();
    let first = (0..<m![A, B]>::SIZE).map(|index| index as i8).collect::<Vec<_>>();
    let second = first.iter().map(|value| value.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    let (first_out, second_out) = launch(
        array_passthrough,
        (&mut device, [Inputs { x: &first_input }, Inputs { x: &second_input }]),
    )
    .await
    .unwrap();

    assert_eq!(
        first_out
            .to_host::<m![A, B]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(first),
    );
    assert_eq!(
        second_out
            .to_host::<m![A, B]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second),
    );
}

#[tokio::test]
async fn borrowed_model_param_passes() {
    let mut device = Device::new(borrowed_model_passthrough.topology()).unwrap();
    let first = (0..<m![A, B]>::SIZE).map(|index| index as i8).collect::<Vec<_>>();
    let second = first.iter().map(|value| value.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let model = Model {
        layers: [
            Layer {
                weight: Weight { x: &first_input },
            },
            Layer {
                weight: Weight { x: &second_input },
            },
        ],
    };

    let (first_out, second_out) = launch(borrowed_model_passthrough, (&mut device, &model)).await.unwrap();

    assert_eq!(
        first_out
            .to_host::<m![A, B]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(first),
    );
    assert_eq!(
        second_out
            .to_host::<m![A, B]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second),
    );
}

#[tokio::test]
async fn borrowed_model_layer_loop_selects_each_layer() {
    let mut device = Device::new(borrowed_model_layer_loop.topology()).unwrap();
    let first = vec![1; <m![A, B]>::SIZE];
    let second = vec![2; <m![A, B]>::SIZE];
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let model = Model {
        layers: [&first_input, &second_input].map(|x| Layer { weight: Weight { x } }),
    };
    let mut output = HostTensor::<i8, m![Layers, A, B]>::zero()
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();

    launch(borrowed_model_layer_loop, (&mut device, &model, &mut output))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![Layers, A, B]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![Layers, A, B], CurrentBackend>::from_vec([first, second].concat()),
    );
}

#[tokio::test]
async fn local_struct_passes() {
    let mut device = Device::new(local_struct_passthrough.topology()).unwrap();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    let out = launch(local_struct_passthrough, (&mut device, &input)).await.unwrap();

    assert_eq!(
        out.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn explicit_unit_passes() {
    let mut device = Device::new(explicit_unit.topology()).unwrap();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();
    let mut output = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    launch(explicit_unit, (&mut device, &input, &mut output)).await.unwrap();

    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data),
    );
}
