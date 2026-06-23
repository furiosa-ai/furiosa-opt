//! Commit Engine: Tensor Unit stream to DM.
//!
//! Drains a post-Commit-Adapter `TuTensor` (from `commit_trim` onwards) into a
//! [`DmTensor`] (or an existing mutable view). `verify_commit` synthesizes the
//! write sequencer with `SMapping::sequence` (Write), the dual of the fetch read.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use furiosa_opt_lower::config_commit;

use crate::context::*;
use crate::engine::CanApplyCommit;
use crate::runtime::Backend;
use crate::scalar::*;
use crate::tensor::memory::{Address, DmTensor, DmTensorViewMut};
use crate::tensor::tu::TuTensor;

// ANCHOR: commit_impl
impl<'l, const T: Tu, P: CanApplyCommit, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Commits to data memory at `address`.
    #[primitive(TuTensor::commit)]
    pub fn commit<Element: M>(self, address: Address) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        verify_commit::<D, Time, Packet, Element>();
        DmTensor::new(self.inner.transpose(false), address)
    }

    /// Commits to a mutable tensor view in data memory.
    #[primitive(TuTensor::commit_view)]
    pub fn commit_view<Element: M>(self, mut dst: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>) {
        verify_commit::<D, Time, Packet, Element>();
        dst.inner.write_transpose(self.inner.view(), false);
    }
}
// ANCHOR_END: commit_impl

/// Validates the Commit engine and synthesizes the write config via [`config_commit`], panicking
/// with the commit error on failure. The resolved descriptors are not consumed yet, so they're
/// explicitly discarded. The dual of `verify_fetch`: it writes the time-ordered `(InTime, InPacket)`
/// stream into the DM `Element` layout.
pub(crate) fn verify_commit<D: Scalar, InTime: M, InPacket: M, Element: M>() {
    let _ = config_commit(
        &InTime::to_value(),
        &InPacket::to_value(),
        &Element::to_value(),
        D::BITS,
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each accepts a commit across an orthogonal phenomenon; the matcher's write synthesis is
    /// exhaustively tested in `furiosa-mapping-impl`.
    mod commit_valid {
        use super::*;

        axes![N = 8, A = 4, B = 3, C = 4];

        /// Full trim, then a clean multi-axis commit.
        #[test]
        fn full_trim_then_commit() {
            verify_commit::<i8, m![A, B, C], m![N], m![A, B, C, N]>();
        }

        /// Packet keeps trailing flit padding (partial trim).
        #[test]
        fn partial_trim_then_commit() {
            verify_commit::<i8, m![A], m![N # 16], m![A, N # 16]>();
        }

        /// Time axes are transposed across the commit.
        #[test]
        fn time_transpose() {
            verify_commit::<i8, m![A # 32, B], m![N], m![B, A # 32, N]>();
        }

        /// The stream emits the live block four times (`1#2 ⋈ 1#2`); overlapping
        /// dummy writes collapse onto the DM's smaller dummy region — realizable
        /// even though an injective placement could not fit.
        #[test]
        fn interleaved_time_padding_overlaps_into_dm_padding() {
            verify_commit::<i8, m![1 # 2, A, 1 # 2], m![N], m![A, 1 # 3, N]>();
        }
    }
}
