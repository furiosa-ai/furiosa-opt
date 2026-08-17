//! Rejection fixture for the generic-`#[device]`-entrypoint check in MIR->vISA
//! translation (npu-visa-translate). A `#[device]` fn generic over an axis
//! (`A: M`) is not monomorphized by the time it reaches translation, so
//! `Instance::mono` would ICE (`... has type parameters`). MIR->vISA must reject
//! every such entrypoint up front with a clean diagnostic pointing at the
//! concrete (monomorphized) wrapper pattern.
//!
//! Two generic fns share the leaf name `gen_copy` (in sibling modules) so a
//! single `--device-function gen_copy` filter matches BOTH, exercising the
//! "report EVERY generic entrypoint, not just the first" behaviour. See
//! furiosa-ai#18538.

use furiosa_opt_std::prelude::*;

axes![B = 256];

pub mod a {
    use super::*;

    /// Generic over axis `A`: an HBM->DM->HBM copy whose shape is not concrete
    /// until monomorphized. MIR->vISA must reject this generic entrypoint.
    #[device(chip = 1)]
    pub fn gen_copy<A: AxisName>(
        ctx: &mut Context,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
        dm.to_hbm(&mut ctx.tdma)
    }
}

pub mod b {
    use super::*;

    /// A second generic entrypoint with the same leaf name, in a sibling module,
    /// so the same `--device-function` filter selects both.
    #[device(chip = 1)]
    pub fn gen_copy<A: AxisName>(
        ctx: &mut Context,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
        dm.to_hbm(&mut ctx.tdma)
    }
}
