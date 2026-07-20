use furiosa_opt_examples::view::simpl::view_simpl;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_view_simpl() {
    use furiosa_opt_examples::view::simpl::{A, B};

    let mut ctx = Context::acquire();

    // Create input tensor with shape (A=512)(B=8).
    let input = HostTensor::<i32, m![A, B]>::from_vec((0..4096).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    // Call the device function.
    let output = launch(view_simpl, (&mut *ctx, &input)).await;

    // Verify the output tensor content: [[6,7,0,1,2,3,4,5],[14,15,8,9,10,11,12,13],...].
    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0..512)
                .flat_map(|x| [6, 7, 0, 1, 2, 3, 4, 5].into_iter().map(move |i| 8 * x + i))
                .collect::<Vec<_>>(),
        )
    );
}

/// `m![[A, B] # 1024]` extends `(A=9)(B=7)=63` real logical values up to the padded size `1024`
/// (a valid 64-slice layout); indices `63..1024` are padding. Padding reads back as `0` regardless
/// of what a host buffer writes there (`Backend::uninit`'s zero-filled default). A padding
/// *destination* slot also reads back `0` even when its rotation source is a real (`< 63`) value:
/// the commit only ever writes the real `0..63` destination range, so a real value rotated into a
/// padding destination is dropped, not carried through. [`view_padding`] rotates every group of 4
/// as `[3, 0, 1, 2]` across the whole tensor; the expected buffer below applies that same two-sided
/// clamp (real destination AND real source, else `0`).
#[tokio::test]
async fn test_view_padding() {
    use furiosa_opt_examples::view::padding::{A, B, view_padding};

    let mut ctx = Context::acquire();

    let input = HostTensor::<i32, m![[A, B] # 1024]>::from_vec(
        (0..1024).map(|i| if i < 63 { i } else { 0 }).collect::<Vec<_>>(),
    )
    .to_hbm::<m![1], m![[A, B] # 1024]>(&mut ctx.pdma)
    .await;

    let output = launch(view_padding, (&mut *ctx, &input)).await;

    assert_eq!(
        output.to_host::<m![[A, B] # 1024]>(&mut ctx.pdma).await.into_vec(),
        (0..256)
            .flat_map(|x| [4 * x + 3, 4 * x, 4 * x + 1, 4 * x + 2])
            .enumerate()
            .map(|(dest, src)| if dest < 63 && src < 63 { src } else { 0 })
            .collect::<Vec<_>>(),
    );
}
