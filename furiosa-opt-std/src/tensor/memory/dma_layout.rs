//! Byte-alignment and address-stride validation shared by every DMA-producing tensor kind.

use furiosa_mapping::*;

use crate::scalar::Scalar;

/// Checks the DMA packet and stream strides against the destination access width.
pub(crate) fn assert_dma_layout<D: Scalar, Src: M, SrcElement: M, Dst: M, DstElement: M>(min_align: usize) {
    assert!(min_align > 0, "min_align must be positive");

    let packet_end = check_dma_tail::<D>(&SrcElement::to_value(), &DstElement::to_value(), min_align);
    check_dma_address_stride::<D>(&[Src::to_value()], &[Dst::to_value()], min_align, packet_end);
}

pub(crate) fn assert_dm_dma_layout<
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    Chip2: M,
    Cluster2: M,
    Slice2: M,
    Element2: M,
>(
    min_align: usize,
) {
    assert!(min_align > 0, "min_align must be positive");

    let packet_end = check_dma_tail::<D>(&Element::to_value(), &Element2::to_value(), min_align);
    check_dma_address_stride::<D>(
        &[
            Chip::to_value(),
            Cluster::to_value(),
            Slice::to_value(),
            Element::to_value(),
        ],
        &[
            Chip2::to_value(),
            Cluster2::to_value(),
            Slice2::to_value(),
            Element2::to_value(),
        ],
        min_align,
        packet_end,
    );
}

fn check_dma_tail<D: Scalar>(src_element: &Mapping, dst_element: &Mapping, min_align: usize) -> usize {
    let reachable_end = reachable_end(src_element, dst_element);
    let reachable_end_bytes = D::size_in_bytes_from_length(reachable_end);

    assert!(
        reachable_end_bytes.is_multiple_of(min_align),
        "DMA tail alignment violation: reachable destination tail \
         end is not aligned to {min_align} bytes.\n  \
         reachable destination tail end (elements) = {reachable_end}\n  \
         reachable destination tail end (bytes) = {reachable_end_bytes}\n  \
         src element mapping = {src_element:?}\n  \
         dst element mapping = {dst_element:?}",
    );

    reachable_end
}

fn reachable_end(src_element: &Mapping, dst_element: &Mapping) -> usize {
    let (_src_packet, dst_packet, _valid) = src_element.dma_tails(dst_element);
    dst_packet
}

fn check_dma_address_stride<D: Scalar>(src: &[Mapping], dst: &[Mapping], min_align: usize, packet_end: usize) {
    let src = src.iter().collect::<Vec<_>>();
    let dst = dst.iter().collect::<Vec<_>>();
    let configs = sequence(&src, &dst, SequencerMode::Carve)
        .expect("dma layout: destination stream must be covered by the source");
    for config in &configs {
        for (&stream_stride, _entry) in config.0.iter() {
            if stream_stride < packet_end {
                continue;
            }
            let stride_bytes = D::size_in_bytes_from_length(stream_stride);
            assert!(
                stride_bytes.is_multiple_of(min_align),
                "DMA address stride alignment violation: destination stream stride {stream_stride} \
                 (at or past the burst packet) is {stride_bytes} bytes, not aligned to {min_align}-byte \
                 granularity.\n  \
                 reachable packet end (elements) = {packet_end}\n  \
                 src mappings = {src:?}\n  \
                 dst mappings = {dst:?}",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_lower::DM_WRITE_ALIGN_BYTES;

    use super::*;
    use crate::scalar::Scalar;

    #[test]
    fn unittest_extents_reachable_end_with_dst_padding_absorb() {
        axes![A = 8, B = 3];
        // matched B (directcast, [1,3)) + divisor padding [3,8) extend the tail.
        // matched A is non-directcast (divisor_stride=8 ≠ dividend_stride=3),
        // so the walk stops at 8 — A's iteration breaks src-side contiguity.
        assert_eq!(reachable_end(&<m![A, B]>::to_value(), &<m![A, B # 8]>::to_value()), 8);
    }

    #[test]
    fn unittest_extents_reachable_end_invariant_under_outer_cluster_slice() {
        axes![Cl = 2, Sl = 4, A = 3];
        // The tail check looks only at divisor-side spans, so adding outer
        // cluster/slice axes to the source must produce the same answer.
        assert_eq!(
            reachable_end(&<m![A]>::to_value(), &<m![A # 16]>::to_value()),
            reachable_end(&<m![Cl, Sl, A]>::to_value(), &<m![A # 16]>::to_value()),
        );
    }

    #[test]
    fn unittest_extents_reachable_end_single_element_underflows_alignment() {
        axes![A = 1];
        // A single i32 tail is narrower than one DM write.
        let end = reachable_end(&<m![A]>::to_value(), &<m![A]>::to_value());
        assert_eq!(end, 1);
        assert_eq!(<i32 as Scalar>::size_in_bytes_from_length(end), 4);
        assert!(!<i32 as Scalar>::size_in_bytes_from_length(end).is_multiple_of(DM_WRITE_ALIGN_BYTES));
    }

    #[test]
    fn unittest_assert_dma_layout_canonical_cluster_slice_passes() {
        // End-to-end wrapper test on a realistic DM-tier shape:
        // outer Cluster/Slice partitioning, inner element data.
        axes![Cl = 2, Sl = 4, A = 8, B = 4];
        assert_dma_layout::<i32, m![Cl, Sl, A, B], m![A, B], m![Cl, Sl, A, B], m![A, B]>(DM_WRITE_ALIGN_BYTES);
    }

    #[test]
    fn unittest_assert_dma_layout_dst_padding_absorbed() {
        axes![A = 8, B = 3];
        assert_dma_layout::<i32, m![A, B], m![A, B], m![A, B # 8], m![A, B # 8]>(DM_WRITE_ALIGN_BYTES);
    }

    #[test]
    fn unittest_assert_dma_layout_min_align_one_is_noop() {
        // DM→HBM / HBM→HBM use min_align = 1, where both the tail-end check
        // and the stride-alignment check trivially pass. This pins that
        // contract so future refactors of either check cannot regress the
        // DRAM-write path.
        axes![A = 1];
        assert_dma_layout::<i32, m![A], m![A], m![A], m![A]>(1);

        axes![Cl = 2, Sl = 4, B = 3];
        assert_dma_layout::<i32, m![Cl, Sl, B], m![B], m![Cl, Sl, B # 7], m![B # 7]>(1);
    }

    #[test]
    fn unittest_assert_dma_layout_decomposed_padded_axis() {
        // Destination splits a padded axis: `A` (live 3) padded to 4, then `(A # 4) / 2, (A # 4) % 2`.
        // The factor-algebra division does not surface the `/ 2` outer stride, so it never checked it;
        // sequencing enumerates every stream stride. For i32 (4 B) the `% 2` packet is 8 B and the
        // `/ 2` stride is 8 B, both aligned, so the layout passes.
        axes![Cl = 2, Sl = 4, A = 3];
        assert_dma_layout::<i32, m![Cl, Sl, A], m![A], m![Cl, Sl, A # 4 / 2, A # 4 % 2], m![A # 4 / 2, A # 4 % 2]>(
            DM_WRITE_ALIGN_BYTES,
        );
    }

    /// A packed sub-byte load whose flat source element (`m![A, B]`, 32768 elements) feeds one
    /// 128-element period of a modulo-decomposed DM tile. `dma_tails` compares their semantic prefix,
    /// skipping the sixteen affine B rows despite the different factorization, so `reachable_end` is
    /// the full 128-element period (64 bytes), which is `min_align(8)`-aligned.
    #[test]
    fn unittest_assert_dma_layout_packed_subbyte_sliced_load() {
        use crate::scalar::f4e2m1;
        // A packed sub-byte load whose innermost axis is a fraction of `min_align` bytes (`B = 8`
        // `f4e2m1` = 4 bytes), feeding a sliced, modulo-decomposed DM tile.
        axes![A = 4096, B = 8];
        assert_dma_layout::<
            f4e2m1,
            m![1, A, B],
            m![A, B],
            m![1, 1 # 2, A / 16, A / 8 % 2, A % 8, B],
            m![A / 8 % 2, A % 8, B],
        >(DM_WRITE_ALIGN_BYTES);
    }
}
