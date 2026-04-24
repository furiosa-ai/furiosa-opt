use furiosa_visa_examples::view::{padding::view_padding, simpl::view_simpl};
use furiosa_visa_std::prelude::*;

#[tokio::test]
async fn test_view_simpl() {
    use furiosa_visa_examples::view::simpl::{A, B};

    let mut ctx = Context::acquire();

    // Create input tensor with shape (A=512)(B=4).
    let input = HostTensor::<i32, m![A, B]>::from_buf((0..2048).map(Opt::Init).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

    // Call the device function.
    let output = launch(view_simpl, (&mut *ctx, &input)).await;

    // Verify the output tensor content: [[3,0,1,2],[7,4,5,6],...].
    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.to_buf(),
        (0..512)
            .flat_map(|x| [4 * x + 3, 4 * x, 4 * x + 1, 4 * x + 2])
            .map(Opt::Init)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_view_padding() {
    use furiosa_visa_examples::view::padding::{A, B};

    let mut ctx = Context::acquire();

    // Create input tensor with shape (A=9)(B=7)=64, with last element uninitialized.
    let input = HostTensor::<i32, m![[A, B] # 64]>::from_buf((0..63).map(Opt::Init).collect::<Vec<_>>())
        .to_hbm::<m![1], m![[A, B] # 64]>(&mut ctx.pdma, 0x1000)
        .await;

    // Call the device function.
    let output = launch(view_padding, (&mut *ctx, &input)).await;

    // Verify the output tensor content: [[3,0,1,2,7,4,5],[6,11,8,9,10,15,11],...], with 63 replaced with uninitialized value.
    assert_eq!(
        output.to_host::<m![[A, B] # 64]>(&mut ctx.pdma).await.to_buf(),
        (0..16)
            .flat_map(|x| [4 * x + 3, 4 * x, 4 * x + 1, 4 * x + 2])
            .take(63)
            .map(|x| { if x < 63 { Opt::Init(x) } else { Opt::Uninit } })
            .collect::<Vec<_>>(),
    );
}
