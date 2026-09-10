use furiosa_opt_examples::view::simpl::view_simpl;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_view_simpl() {
    use furiosa_opt_examples::view::simpl::{A, B};

    let mut device = Device::new(view_simpl.topology()).unwrap();

    // Create input tensor with shape (A=512)(B=8).
    let input = HostTensor::<i32, m![A, B]>::from_vec((0..4096).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    // Call the device function.
    let output = launch(view_simpl, (&mut device, &input)).await.unwrap();

    // Verify the output tensor content: [[6,7,0,1,2,3,4,5],[14,15,8,9,10,11,12,13],...].
    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0..512)
                .flat_map(|x| [6, 7, 0, 1, 2, 3, 4, 5].into_iter().map(move |i| 8 * x + i))
                .collect::<Vec<_>>(),
        )
    );
}
