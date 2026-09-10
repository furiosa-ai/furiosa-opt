//! Type-level wrappers over the shared `furiosa_opt_lower::config_vector_*` verifications.

use furiosa_mapping::*;
use furiosa_opt_lower::{
    SliceRequest, VectorIntraSliceUnzipInput, VectorNarrowSplitInput, VectorNarrowTrimInput, VectorWidenConcatInput,
    VectorWidenPadInput,
};

pub(crate) fn verify_vector_intra_slice_unzip<I: AxisName, Time: M, SplitTime: M, Packet: M>() {
    furiosa_opt_lower::config_vector_intra_slice_unzip(VectorIntraSliceUnzipInput {
        group_axis: I::NAME,
        in_time: Time::to_value(),
        in_packet: Packet::to_value(),
    })
    .unwrap_or_else(|message| panic!("{message}"));
    furiosa_opt_lower::config_slice(SliceRequest::Layout {
        axes: vec![Symbol::<I>::to_value()],
        element: Time::to_value(),
        output: SplitTime::to_value(),
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_narrow_split<Time: M, Packet: M, Time2: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_narrow_split(VectorNarrowSplitInput {
        in_time: Time::to_value(),
        in_packet: Packet::to_value(),
        out_time: Time2::to_value(),
        out_packet: Packet2::to_value(),
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_widen_concat<Time: M, Packet: M, Time2: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_widen_concat(VectorWidenConcatInput {
        in_time: Time::to_value(),
        in_packet: Packet::to_value(),
        out_time: Time2::to_value(),
        out_packet: Packet2::to_value(),
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_narrow_trim<Packet: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_narrow_trim(VectorNarrowTrimInput {
        in_packet: Packet::to_value(),
        out_packet: Packet2::to_value(),
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

pub(crate) fn verify_vector_widen_pad<Packet: M, Packet2: M>() {
    furiosa_opt_lower::config_vector_widen_pad(VectorWidenPadInput {
        in_packet: Packet::to_value(),
        out_packet: Packet2::to_value(),
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![I = 2, A = 16];

    #[test]
    fn accepts_unzip_output_with_group_axis_removed() {
        verify_vector_intra_slice_unzip::<I, m![I, A / 8], m![A / 8], m![A % 8]>();
    }
}
