//! Assertions for `to_vrf`: the per-slice vector register file capacity.
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
//! fn store<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![A]> {
//!     input.fetch::<m![1], m![A]>().collect::<m![A / 8], m![A % 8]>().to_vrf()
//! }
//! #
//! # let mut ctx = Context::acquire();
//! # let input: BeginTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]> =
//! #     BeginTensor::new(&mut ctx.sub, Tensor::zero());
//! # let _vrf = store(input);
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
//! fn store<'l, const T: Tu>(
//!     input: BeginTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![A]> {
//!     input.fetch::<m![1], m![A]>().collect::<m![A / 8], m![A % 8]>().to_vrf()
//! }
//! #
//! # let mut ctx = Context::acquire();
//! # let input: BeginTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A]> =
//! #     BeginTensor::new(&mut ctx.sub, Tensor::zero());
//! # let _vrf = store(input);
//! ```

use furiosa_opt_std::prelude::*;

axes![A = 2048];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// `A` `i32` values are 8 KiB per slice, filling the file exactly. This is the largest legal store.
#[device(chip = 1)]
pub fn to_vrf_at_capacity(ctx: &mut Context, input: &HbmTensor<i32, Chip, m![A]>) {
    let dm = input.to_dm::<Cluster, Slice, m![A]>(&mut ctx.tdma);

    let _vrf: VrfTensor<i32, Chip, Cluster, Slice, m![A]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![1], m![A]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 8], m![A % 8]>()
        .to_vrf();
}
