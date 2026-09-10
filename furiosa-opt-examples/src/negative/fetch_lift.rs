//! Rejection fixtures for fetch lifts.
//! Compile-fail cases cover typestate and size constraints; kernels cover lowering checks.
//!
//! Lifting onto the same dimension twice:
//!
//! ```compile_fail,E0599
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 128, H = 2, V = 16];
//!
//! fn twice<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, bf16, m![1], m![1 # 2], m![A, 2], m![1], m![H, V]>,
//! ) -> DmTensor<bf16, m![1], m![1 # 2], m![A, H], m![V]> {
//!     input
//!         .fetch::<m![H], m![V]>()
//!         .fetch_slice_lift::<m![A, H], m![1]>()
//!         .fetch_slice_lift::<m![A, H], m![1]>()
//!         .collect::<m![1], m![V]>()
//!         .commit_trim::<m![V]>()
//!         .commit()
//! }
//! ```
//!
//! Resizing `Cluster`:
//!
//! ```compile_fail,E0080
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 128, H = 2, V = 16];
//!
//! fn resized<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, bf16, m![1], m![1 # 2], m![A], m![1], m![H, V]>,
//! ) -> DmTensor<bf16, m![1], m![1], m![A], m![V]> {
//!     input
//!         .fetch::<m![H], m![V]>()
//!         .fetch_cluster_lift::<m![1], m![H]>()
//!         .collect::<m![1], m![V]>()
//!         .commit_trim::<m![V]>()
//!         .commit()
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, bf16, m![1], m![1 # 2], m![A], m![1], m![H, V]> =
//! #     BeginTensor::new(&mut device.main, Tensor::zero());
//! # let _dm = resized(input);
//! ```
//!
//! Resizing `Slice` from 256 to the next supported width, 128:
//!
//! ```compile_fail,E0080
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 128, H = 2, V = 16];
//!
//! fn resized_slice<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, bf16, m![1], m![1 # 2], m![A, 2], m![1], m![H, V]>,
//! ) -> DmTensor<bf16, m![1], m![1 # 2], m![A], m![V]> {
//!     input
//!         .fetch::<m![H], m![V]>()
//!         .fetch_slice_lift::<m![A], m![1]>()
//!         .collect::<m![1], m![V]>()
//!         .commit_trim::<m![V]>()
//!         .commit()
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, bf16, m![1], m![1 # 2], m![A, 2], m![1], m![H, V]> =
//! #     BeginTensor::new(&mut device.main, Tensor::zero());
//! # let _dm = resized_slice(input);
//! ```
//!
//! A lift output time that does not divide its input time:
//!
//! ```compile_fail,E0080
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 128, H = 2, Six = 6, Four = 4, V = 16];
//!
//! fn nondivisible<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, bf16, m![1], m![2], m![A], m![1], m![Six, V]>,
//! ) {
//!     let _ = input
//!         .fetch::<m![Six], m![V]>()
//!         .fetch_cluster_lift::<m![H], m![Four]>();
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, bf16, m![1], m![2], m![A], m![1], m![Six, V]> =
//! #     BeginTensor::new(&mut device.main, Tensor::zero());
//! # nondivisible(input);
//! ```
//!
//! Taking more out of the stream than the dimension can hold, which its size alone rules out:
//!
//! ```compile_fail,E0080
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![A = 128, H = 2, G = 2, Q = 4, V = 16];
//!
//! fn overshoots<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, bf16, m![1], m![2], m![A, 2], m![1], m![H, G, Q, V]>,
//! ) -> DmTensor<bf16, m![1], m![H], m![A, 2], m![V]> {
//!     input
//!         .fetch::<m![H, G, Q], m![V]>()
//!         .fetch_cluster_lift::<m![H], m![1]>()
//!         .collect::<m![1], m![V]>()
//!         .commit_trim::<m![V]>()
//!         .commit()
//! }
//! #
//! # let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
//! # let input: BeginTensor<'_, _, bf16, m![1], m![2], m![A, 2], m![1], m![H, G, Q, V]> =
//! #     BeginTensor::new(&mut device.main, Tensor::zero());
//! # let _dm = overshoots(input);
//! ```
//!

use furiosa_opt_std::prelude::*;

// The axes come from the public twin, so a fixture and the kernel it contrasts with are stated over
// the very same axis types.
pub use crate::fetch_lift::{A, G, H, Oct, V};

// `Step` is the axis a lift turns into bases, and `Four` the four data bytes under a packet's padding.
axes![Step = 2, Four = 4];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Rejects replacing the live slice dimension `G` with the lifted dimension `H`.
#[device(chip = 1)]
pub fn live_placement_renamed(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![A, G, H, V]>,
    output: &mut HbmTensor<bf16, Chip, m![A, H, V]>,
) {
    let dm: DmTensor<bf16, Chip, Cluster, m![A, G], m![H, V]> =
        input.to_dm::<Cluster, m![A, G], m![H, V]>(&mut device.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Rejects a lifted axis whose 4-byte step cannot be represented by an 8-byte-aligned fetch base.
#[device(chip = 1)]
pub fn unaligned_base_step(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![A, Step, Four]>,
    output: &mut HbmTensor<i8, Chip, m![A, Step, Four # 8]>,
) {
    let dm: DmTensor<i8, Chip, Cluster, m![A, 2], m![Step, Four]> =
        input.to_dm::<Cluster, m![A, 2], m![Step, Four]>(&mut device.tdma);

    let result: DmTensor<i8, Chip, Cluster, m![A, Step], m![Four # 8]> = device
        .main
        .begin(dm.view())
        .fetch::<m![Step], m![Four # 8]>()
        .fetch_slice_lift::<m![A, Step], m![1]>()
        .collect::<m![1], m![Four # 32]>()
        .commit_trim::<m![Four # 8]>()
        .commit();

    result.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
