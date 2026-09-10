//! Report §3.4's question, whether a padding-notation axis loses the tile index. It does not.

use furiosa_opt_examples::padded_axis_tile::{
    Chip, Chunk, Chunked, ChunkedDivisible, ChunkedPadded, ChunkedPlain, ChunkedWide, Divisible, H, Live, Padded,
    Plain, Wide, divisible_chunk_read, padded_axis_chunk_read, padded_symbol_chunk_copy, plain_chunk_read,
};
use furiosa_opt_std::prelude::*;

const CHUNK: usize = 8;
const WANT: usize = 1;
/// The chunks `ChunkedDivisible` spans. Live rows reach only the first three.
const PADDED_CHUNKS: usize = 4;

/// Row `r` holds the value of the chunk it belongs to, so one element names the chunk read.
fn rows(count: usize) -> Vec<bf16> {
    (0..count)
        .flat_map(|r| std::iter::repeat_n(bf16::from_f32((r / CHUNK) as f32), H::SIZE))
        .collect()
}

/// The tutorial's own shape, whose live rows do not divide into its three chunks.
#[tokio::test]
async fn test_tutorial_shape_reads_the_requested_chunk() {
    let mut device = Device::new(padded_axis_chunk_read.topology()).unwrap();
    let input = HostTensor::<bf16, Chunked>::from_vec(rows(24))
        .to_hbm::<Chip, Chunked>(&mut device.pdma)
        .await
        .unwrap();

    let output = launch(padded_axis_chunk_read, (&mut device, &input)).await.unwrap();

    let actual = output.to_host::<Chunk>(&mut device.pdma).await.unwrap().into_vec();
    assert_eq!(actual[0].to_f32(), WANT as f32, "asked for chunk {WANT}");
}

#[tokio::test]
async fn test_padded_symbol_chunk_copy_keeps_each_chunk() {
    let mut device = Device::new(padded_symbol_chunk_copy.topology()).unwrap();
    let input = HostTensor::<bf16, ChunkedPadded>::from_vec(rows(24))
        .to_hbm::<Chip, ChunkedPadded>(&mut device.pdma)
        .await
        .unwrap();

    let output = launch(padded_symbol_chunk_copy, (&mut device, &input)).await.unwrap();

    let actual = output
        .to_host::<ChunkedPadded>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (r, cell) in actual.iter().step_by(H::SIZE).enumerate().take(Padded::SIZE) {
        assert_eq!(
            cell.to_f32(),
            (r / CHUNK) as f32,
            "row {r} after the chunk-by-chunk copy"
        );
    }
}

/// `Live # 24 / 8` is not `Padded / 8`: a projection carries one term per symbol, at its live size.
#[test]
fn test_inline_padding_leaves_the_axis_at_its_live_size() {
    let axis_size = |mapping: &furiosa_opt_std::prelude::Mapping, name: &str| {
        mapping
            .axes()
            .iter()
            .find_map(|a| (a.symbol.to_string() == name).then_some(a.size))
    };

    let inline = <Chunked>::to_value();
    let symbol = <ChunkedPadded>::to_value();

    assert_eq!(inline.size(), symbol.size(), "both spell the same element count");
    assert_eq!(axis_size(&inline, "Live"), Some(Live::SIZE));
    assert_eq!(axis_size(&symbol, "Padded"), Some(Padded::SIZE));
    assert_ne!(Live::SIZE, Padded::SIZE, "the live extent is what the two disagree on");
}

/// The wide declaration's trailing chunk is pure padding, measured on the mapping.
#[test]
fn test_wide_padding_has_a_chunk_with_no_live_rows() {
    let chunks = <ChunkedWide>::to_value().size() / (CHUNK * H::SIZE);
    let live_chunks = Wide::SIZE.div_ceil(CHUNK);

    assert_eq!(chunks, 4, "the declaration spans four chunks");
    assert_eq!(live_chunks, 3, "only three of them hold live rows");
}

/// The lowering side: 4 chunks, 3 with live rows, and the read must return the chunk it asked for.
#[tokio::test]
async fn test_divisible_padding_reads_the_requested_chunk() {
    let mut device = Device::new(divisible_chunk_read.topology()).unwrap();
    let input = HostTensor::<bf16, ChunkedDivisible>::from_vec(rows(PADDED_CHUNKS * CHUNK))
        .to_hbm::<Chip, ChunkedDivisible>(&mut device.pdma)
        .await
        .unwrap();

    let output = launch(divisible_chunk_read, (&mut device, &input)).await.unwrap();

    let actual = output
        .to_host::<m![Divisible # 32 % 8, H]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    assert_eq!(actual[0].to_f32(), WANT as f32, "asked for chunk {WANT}");
}

#[tokio::test]
async fn test_plain_chunk_read_reads_the_requested_chunk() {
    let mut device = Device::new(plain_chunk_read.topology()).unwrap();
    let input = HostTensor::<bf16, ChunkedPlain>::from_vec(rows(Plain::SIZE))
        .to_hbm::<Chip, ChunkedPlain>(&mut device.pdma)
        .await
        .unwrap();

    let output = launch(plain_chunk_read, (&mut device, &input)).await.unwrap();

    let actual = output
        .to_host::<m![Plain % 8, H]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    assert_eq!(actual[0].to_f32(), WANT as f32, "asked for chunk {WANT}");
}
