use furiosa_visa_std::prelude::*;

axes![A = 512, B = 4];

#[tokio::test]
async fn test_view_broadcast() {
    let mut ctx = Context::acquire(); // TODO: 여기에서 할 수 있는게 맞는지?

    // Create input tensor with shape (A=512)(B=4).
    let input = HostTensor::<i32, m![A]>::from_buf((0..512).map(Opt::Init).collect::<Vec<_>>());
    let hbm1 = input.to_hbm::<m![1], m![A]>(&mut ctx.pdma, 0).await;
    let hbm2 = hbm1.to_hbm::<{ Dma::Tensor }, m![A, B]>(&mut ctx.tdma, 0);
    let output = hbm2.to_host::<m![A, B]>(&mut ctx.pdma).await;

    assert_eq!(
        output.to_buf(),
        (0..<m![A]>::SIZE as i32)
            .flat_map(|x| [x; <m![B]>::SIZE])
            .map(Opt::Init)
            .collect::<Vec<_>>()
    );
}
