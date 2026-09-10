//! Assertions for `to_vrf`: the per-slice vector register file capacity.
//!
//! The bound holds at both positions the store can take, off the `collect` and off the vector
//! engine's write port, so each has a kernel below at exactly capacity.
//!
//! The file holds 8 KiB per slice, so one slice's `Element` is what has to fit. The kernel below
//! stores a vector that every slice keeps a full copy of, sized to fill the file exactly.
//!
//! Doubling that vector overruns the file, and `to_vrf` rejects it at compile time. The two examples
//! differ only in the axis size, so the second one fails for the capacity and nothing else.
//!
//! ```
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 2048];
//!
//! // 2048 x i32 = 8192 bytes per slice, the whole file.
//! fn store<'l>(
//!     input: BeginTensor<'l, { Tu::Sub }, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![A]> {
//!     input.fetch::<m![1], m![A]>().collect::<m![A / 8], m![A % 8]>().to_vrf()
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]> =
//! #     BeginTensor::new(&mut device.sub, Tensor::zero());
//! # let _vrf = store(input);
//! ```
//!
//! The same bound holds for a main-context store off the vector engine's write port, which takes
//! `device.sub` as well:
//!
//! ```
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 2048];
//!
//! fn store<'l>(
//!     input: BeginTensor<'l, { Tu::Main }, f32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]>,
//!     sub: &mut TuContext<{ Tu::Sub }>,
//! ) -> VrfTensor<f32, m![1], m![1 # 2], m![1 # 256], m![A]> {
//!     input
//!         .fetch::<m![1], m![A]>()
//!         .collect::<m![A / 8], m![A % 8]>()
//!         .vector_init()
//!         .vector_intra_slice_tag(TagMode::Zero)
//!         .vector_narrow_split::<m![A / 8, A % 8 / 4 % 2], m![A % 4]>()
//!         .vector_fp_unary(FpUnaryOp::Sqrt)
//!         .vector_widen_concat::<m![A / 8], m![A % 8]>()
//!         .vector_final()
//!         .to_vrf(sub)
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, { Tu::Main }, f32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]> =
//! #     BeginTensor::new(&mut device.main, Tensor::zero());
//! # let _vrf = store(input, &mut device.sub);
//! ```
//!
//! Doubling it overruns the file there too:
//!
//! ```compile_fail
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 4096];
//!
//! fn store<'l>(
//!     input: BeginTensor<'l, { Tu::Main }, f32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]>,
//!     sub: &mut TuContext<{ Tu::Sub }>,
//! ) -> VrfTensor<f32, m![1], m![1 # 2], m![1 # 256], m![A]> {
//!     input
//!         .fetch::<m![1], m![A]>()
//!         .collect::<m![A / 8], m![A % 8]>()
//!         .vector_init()
//!         .vector_intra_slice_tag(TagMode::Zero)
//!         .vector_narrow_split::<m![A / 8, A % 8 / 4 % 2], m![A % 4]>()
//!         .vector_fp_unary(FpUnaryOp::Sqrt)
//!         .vector_widen_concat::<m![A / 8], m![A % 8]>()
//!         .vector_final()
//!         .to_vrf(sub)
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, { Tu::Main }, f32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]> =
//! #     BeginTensor::new(&mut device.main, Tensor::zero());
//! # let _vrf = store(input, &mut device.sub);
//! ```
//!
//! ```compile_fail
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 4096];
//!
//! // 4096 x i32 = 16384 bytes per slice, twice the file.
//! fn store<'l>(
//!     input: BeginTensor<'l, { Tu::Sub }, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![A]> {
//!     input.fetch::<m![1], m![A]>().collect::<m![A / 8], m![A % 8]>().to_vrf()
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]> =
//! #     BeginTensor::new(&mut device.sub, Tensor::zero());
//! # let _vrf = store(input);
//! ```

use furiosa_opt_std::prelude::*;

axes![A = 2048];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// `A` `i32` values are 8 KiB per slice, filling the file exactly. This is the largest legal store.
#[device(chip = 1)]
pub fn to_vrf_fills_file(device: &mut Device, input: &HbmTensor<i32, Chip, m![A]>) {
    let dm = input.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);

    let _vrf: VrfTensor<i32, Chip, Cluster, Slice, m![A]> = device
        .sub
        .begin(dm.view())
        .fetch::<m![1], m![A]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 8], m![A % 8]>()
        .to_vrf();
}

/// The same capacity through the vector engine's write port: `A` `f32` values are 8 KiB per slice,
/// so the pass output fills the file exactly.
#[device(chip = 1)]
pub fn ve_to_vrf_fills_file(device: &mut Device, input: &HbmTensor<f32, Chip, m![A]>) {
    let dm = input.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);

    let _vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A]> = device
        .main
        .begin(dm.view())
        .fetch::<m![1], m![A]>()
        .fetch_cast::<f32>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![A / 8, A % 8 / 4 % 2], m![A % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![A / 8], m![A % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
}
