use furiosa_opt_examples::param::{A, B, Inputs, struct_passthrough, tuple_passthrough};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn tuple_param_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_buf(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

    let out = launch(tuple_passthrough, (&mut *ctx, (&input,))).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_raw(),
        <CurrentBackend as Backend>::RawTensor::from_buf::<m![A, B]>(data)
    );
}

#[tokio::test]
async fn struct_param_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_buf(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

    let out = launch(struct_passthrough, (&mut *ctx, Inputs { x: &input })).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_raw(),
        <CurrentBackend as Backend>::RawTensor::from_buf::<m![A, B]>(data)
    );
}
