use furiosa_opt_examples::param::{A, B, Inputs, struct_passthrough, tuple_passthrough};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn tuple_param_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let out = launch(tuple_passthrough, (&mut *ctx, (&input,))).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn tuple_param_reuses_output() {
    let mut ctx = Context::acquire();
    let first = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let second = first.iter().map(|x| x.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first)
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let mut output = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    launch(tuple_passthrough, (&mut *ctx, (&first_input,)))
        .output(&mut output)
        .await;
    launch(tuple_passthrough, (&mut *ctx, (&second_input,)))
        .output(&mut output)
        .await;

    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second)
    );
}

#[tokio::test]
async fn struct_param_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let out = launch(struct_passthrough, (&mut *ctx, Inputs { x: &input })).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}
