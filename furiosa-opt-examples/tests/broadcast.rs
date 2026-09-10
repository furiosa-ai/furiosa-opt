use furiosa_opt_std::prelude::*;

axes![A = 512, B = 4];

#[tokio::test]
async fn test_view_broadcast() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap(); // TODO: 여기에서 할 수 있는게 맞는지?

    // Create input tensor with shape (A=512)(B=4).
    let input = HostTensor::<i32, m![A]>::from_vec((0..512).collect::<Vec<_>>());
    let hbm1 = input.to_hbm::<m![1], m![A]>(&mut device.pdma).await.unwrap();
    let hbm2 = hbm1.to_hbm::<{ Dma::Tensor }, m![A, B]>(&mut device.tdma);
    let output = hbm2.to_host::<m![A, B]>(&mut device.pdma).await.unwrap();

    assert_eq!(
        output.into_vec(),
        Tensor::<i32, m![A, B]>::from_vec(
            (0..<m![A]>::SIZE as i32)
                .flat_map(|x| [x; <m![B]>::SIZE])
                .collect::<Vec<_>>(),
        )
        .into_vec()
    );
}
