//! Rejection fixtures for the vector engine: an ALU the chain asks for twice, an interleaved group
//! that outgrows the VE register-file cache, a VRF operand the indexer cannot address, and the two
//! ways an operand can be under the wrong partition. The legal operand shapes are
//! [`vrf_operand`](crate::vrf_operand).
//!
//! The partition fixtures are doctests rather than kernels, for the reason
//! [`negative`](crate::negative) gives: rustc refuses each of them. The first is an operand that
//! simply is not the stream's, which the operand impls reject by naming one shared set of `Chip` /
//! `Cluster` / `Slice` for both sides.
//!
//! ```compile_fail,E0277
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![W = 256, B = 32];
//!
//! // The operand is replicated over the 256 slices under an anonymous broadcast, the stream is
//! // partitioned by `W`. Same 256 slices, but the two mappings are not the same type.
//! fn add<'l, const T: Tu>(
//!     stream: BeginTensor<'l, T, i32, m![1], m![1 # 2], m![W], m![1], m![B]>,
//!     operand: VrfTensor<i32, m![1], m![1 # 2], m![256], m![B]>,
//! ) -> DmTensor<i32, m![1], m![1 # 2], m![W], m![B]> {
//!     stream
//!         .fetch::<m![B / 8], m![B % 8]>()
//!         .collect::<m![B / 8], m![B % 8]>()
//!         .vector_init()
//!         .vector_intra_slice_tag(TagMode::Zero)
//!         .vector_fxp(FxpBinaryOp::AddFxp, &operand)
//!         .vector_final()
//!         .commit_trim::<m![B % 8]>()
//!         .commit()
//! }
//! ```
//!
//! Restating the operand under the stream's slice mapping with [`VrfTensor::reshape`] is what makes
//! the same read legal. The two examples differ in that one line, so the one above fails for the
//! partition and nothing else.
//!
//! ```
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![W = 256, B = 32];
//!
//! fn add<'l, const T: Tu>(
//!     stream: BeginTensor<'l, T, i32, m![1], m![1 # 2], m![W], m![1], m![B]>,
//!     operand: VrfTensor<i32, m![1], m![1 # 2], m![256], m![B]>,
//! ) -> DmTensor<i32, m![1], m![1 # 2], m![W], m![B]> {
//!     let operand: VrfTensor<i32, m![1], m![1 # 2], m![W], m![B]> = unsafe { operand.reshape() };
//!     stream
//!         .fetch::<m![B / 8], m![B % 8]>()
//!         .collect::<m![B / 8], m![B % 8]>()
//!         .vector_init()
//!         .vector_intra_slice_tag(TagMode::Zero)
//!         .vector_fxp(FxpBinaryOp::AddFxp, &operand)
//!         .vector_final()
//!         .commit_trim::<m![B % 8]>()
//!         .commit()
//! }
//! ```
//!
//! `reshape` is not a way around the rule, which the last fixture is about: its guard is per-level
//! `SIZE` equality, so an operand spanning a different number of slices is refused before it can
//! reach an op. The guard is a `const` block, so the example calls what it declares.
//!
//! ```compile_fail
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![W = 256, B = 32];
//!
//! fn store<'l>(
//!     input: BeginTensor<'l, { Tu::Sub }, i32, m![1], m![1 # 2], m![1 # 128], m![1], m![B]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 128], m![B]> {
//!     input.fetch::<m![1], m![B]>().collect::<m![B / 8], m![B % 8]>().to_vrf()
//! }
//!
//! // 128 slices restated as 256: `Slice::SIZE` differs, so `assert_slice_preserved` fires.
//! fn relabel(
//!     operand: VrfTensor<i32, m![1], m![1 # 2], m![1 # 128], m![B]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![W], m![B]> {
//!     unsafe { operand.reshape() }
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 128], m![1], m![B]> =
//! #     BeginTensor::new(&mut device.sub, Tensor::zero());
//! # let _ = relabel(store(input));
//! ```

use furiosa_opt_std::prelude::*;

pub use crate::vector_engine::{A, I};

axes![G = 8, D = 8];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// Three fxp ops on one chain, two of which want the same ALU: `AddFxp` takes FxpAdd, `MulInt` takes
/// FxpMul, then `SubFxp` asks for FxpAdd again.
///
/// Expected: `6 is not available for op Binary(SubFxp)`, pinned by the snapshot and by the
/// answer-key test that catches the Cpu panic.
#[device(chip = 1)]
pub fn ve_elementwise_fxp_chain(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut device.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 10)
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_fxp(FxpBinaryOp::SubFxp, 5)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// A VRF operand whose packet lanes are `D` elements apart: the stream's packet is `G`, which the
/// operand stores a row apart, while the indexer can only read one broadcast address or a contiguous
/// run per access.
///
/// Expected: `cannot feed the 8-element packet`, pinned by the snapshot and by the answer-key test
/// that catches the Cpu panic. Storing the operand as `[D, G]` (`G` innermost) is the fix.
#[device(chip = 1)]
pub fn ve_vrf_strided_packet(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![D, G]>,
    operand: &HbmTensor<i32, Chip, m![G, D]>,
) -> HbmTensor<i32, Chip, m![D, G]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![D, G]>(&mut device.tdma);
    let operand_dm = operand.to_dm::<Cluster, Slice, m![G, D]>(&mut device.tdma);

    let operand_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![G, D]> = device
        .sub
        .begin(operand_dm.view())
        .fetch::<m![G, D / 8], m![D % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .to_vrf();

    let result: DmTensor<i32, Chip, Cluster, Slice, m![D, G]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![D], m![G]>()
        .fetch_cast::<i32>()
        .collect::<m![D], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, &operand_vrf)
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// The group axis outermost in the FETCH time, so one interleaved group is the whole 512-element row,
/// over what the VE register-file cache holds (1 KiB = 256 `i32`).
///
/// Expected: translation reports `512 elements sit inside the group axis I`, which the snapshot pins.
/// The fix is [`ve_group_pair_add_group_axis_inner`](crate::vector_engine::ve_group_pair_add_group_axis_inner).
#[device(chip = 1)]
pub fn ve_group_pair_over_cache(
    device: &mut Device,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);

    let result: DmTensor<i32, Chip, Cluster, Slice, m![A]> = device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I, A / 8], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![I, A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![A / 8]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}
