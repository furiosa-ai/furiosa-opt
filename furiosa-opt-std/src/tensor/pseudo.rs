//! Pseudo tensor types used by the docs to explain the common buffer-stream pattern shared by Fetch, Commit, and DMA engines.
//!
//! These types do not correspond to any concrete hardware tensor.
//! They exist so the sequencer documentation can use real source-anchored types instead of duplicating a hidden `# struct BufTensor` boilerplate in every code block.
//!
//! The bodies of `read` / `write` move data by transposing the underlying [`Tensor`] between `Buf` and `(Time, Packet)`.
//! `read` produces a `StreamTensor` whose values are the buffer's values reordered into `(Time, Packet)` iteration order (broadcast allowed, matching Fetch / Switch / DMA-read).
//! `write` reverses the operation (broadcast rejected, matching Commit).

#![allow(dead_code)]

use std::marker::PhantomData;

use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::tensor::Tensor;

// ANCHOR: buf_tensor_def
/// A generic buffer-backed tensor.
/// Anything that holds data in a memory `Buf` and can be streamed in or out.
#[derive(Debug)]
pub struct BufTensor<D: Scalar, Buf: M> {
    inner: Tensor<D, Buf>,
}
// ANCHOR_END: buf_tensor_def

// ANCHOR: stream_tensor_def
/// A streaming view of a tensor in flight.
/// `Packet` is the per-cycle shape and `Time` is the multi-cycle shape.
#[derive(Debug)]
pub struct StreamTensor<'l, D: Scalar, Time: M, Packet: M> {
    inner: Tensor<D, Pair<Time, Packet>>,
    _marker: PhantomData<&'l ()>,
}
// ANCHOR_END: stream_tensor_def

// ANCHOR: buf_tensor_read_write
impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
    /// Reads a stream from this buffer with the supplied `Time` and `Packet` mapping.
    /// `(Time, Packet)` may be a broadcast of `Buf` (matches Fetch / Switch / DMA-read behavior).
    pub fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> {
        StreamTensor {
            inner: self.inner.transpose(true),
            _marker: PhantomData,
        }
    }

    /// Writes a stream back into this buffer.
    /// Broadcast is rejected: each `Buf` slot must have exactly one source position in `(Time, Packet)` (matches Commit behavior).
    pub fn write<'l, Time: M, Packet: M>(&mut self, stream: StreamTensor<'l, D, Time, Packet>) {
        self.inner = stream.inner.transpose(false);
    }
}
// ANCHOR_END: buf_tensor_read_write

impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
    /// Constructs a `BufTensor` from a flat buffer in `Buf`-order.
    /// Mirrors [`Tensor::from_buf`].
    pub fn from_buf(data: impl IntoIterator<Item = D>) -> Self {
        Self {
            inner: Tensor::from_buf(data),
        }
    }

    /// Returns the underlying buffer as a flat `Vec<D>` in `Buf`-order.
    /// Mirrors [`Tensor::to_buf`].
    pub fn to_buf(&self) -> Vec<D> {
        self.inner.to_buf()
    }
}

impl<'l, D: Scalar, Time: M, Packet: M> StreamTensor<'l, D, Time, Packet> {
    /// Returns the stream contents as a flat `Vec<D>` in `(Time, Packet)`-order.
    /// Mirrors [`Tensor::to_buf`].
    pub fn to_buf(&self) -> Vec<D> {
        self.inner.to_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity read: when `Time, Packet` equals `Buf`, `read` returns the buffer unchanged.
    #[test]
    fn read_identity_preserves_order() {
        axes![A = 2, B = 3];

        let buf = BufTensor::<i32, m![A, B]>::from_buf(vec![10, 11, 12, 20, 21, 22]);
        let stream = buf.read::<m![A], m![B]>();

        assert_eq!(
            stream.to_buf(),
            Tensor::<i32, m![A, B]>::from_buf(vec![10, 11, 12, 20, 21, 22]).to_buf()
        );
    }

    /// Axis reordering read: visiting `m![B, A]` over a `m![A, B]` buffer transposes the matrix.
    #[test]
    fn read_reorders_axes() {
        axes![A = 2, B = 3];

        // buf in [A, B]-order: A=0 row, then A=1 row.
        let buf = BufTensor::<i32, m![A, B]>::from_buf(vec![10, 11, 12, 20, 21, 22]);

        // stream in [B, A]-order: B=0 column, then B=1 column, then B=2 column.
        let stream = buf.read::<m![B], m![A]>();

        assert_eq!(
            stream.to_buf(),
            Tensor::<i32, m![A, B]>::from_buf(vec![10, 20, 11, 21, 12, 22]).to_buf()
        );
    }

    /// Write reverses read: round-tripping through a transposed stream restores the buffer.
    #[test]
    fn write_inverts_read() {
        axes![A = 2, B = 3];

        let original = vec![10, 11, 12, 20, 21, 22];
        let buf = BufTensor::<i32, m![A, B]>::from_buf(original.clone());

        let stream = buf.read::<m![B], m![A]>();

        let mut sink = BufTensor::<i32, m![A, B]>::from_buf(vec![0; 6]);
        sink.write(stream);

        assert_eq!(sink.to_buf(), Tensor::<i32, m![A, B]>::from_buf(original).to_buf());
    }

    /// Axis split read: `m![A % 2, A / 2]` reads `A`'s low bit then high bit.
    #[test]
    fn read_splits_axis() {
        axes![A = 4];

        // A=[0, 1, 2, 3]. A % 2 outer, A / 2 inner visits: (0,0)=0, (0,1)=2, (1,0)=1, (1,1)=3.
        let buf = BufTensor::<i32, m![A]>::from_buf(vec![0, 1, 2, 3]);
        let stream = buf.read::<m![A % 2], m![A / 2]>();

        assert_eq!(
            stream.to_buf(),
            Tensor::<i32, m![A]>::from_buf(vec![0, 2, 1, 3]).to_buf()
        );
    }
}
