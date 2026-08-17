//! Rejection fixtures for the `switch` configuration assertions: each kernel states a switch
//! (`Broadcast1`, `Broadcast01`, `Transpose`, `InterTranspose`) or collect mapping that the
//! compiler must refuse, mirroring a `valid_*` twin in the public
//! [`switch_assertions`](crate::switch_assertions) module.
//!
//! The kernels still emulate (the assertions are device-translation checks), so the answer-key
//! tests run them; only compilation must fail. See [`super`] for the snapshot contract.

use furiosa_opt_std::prelude::*;

// The axes come from the public twin, so a fixture and the `valid_*` kernel it contrasts with are
// stated over the very same axis types (and produce the same axis labels in a diagnostic).
pub use crate::switch_assertions::{A, B, C, D, E, G};

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

pub mod packet {
    use super::*;

    #[device(chip = 1)]
    pub fn packet_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![A # 32]>()
            .commit_trim::<m![A # 32]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn collect_time_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<bf16, Chip, m![A, B]>,
        output: &mut HbmTensor<bf16, Chip, m![A, B % 16]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);

        // bf16 B=32 elements = 64 bytes = 2 flits.
        // Correct Time2 would be m![A, B / 16] (absorbing outer flit into time).
        // Here we provide m![A] which is wrong → "Collect time mismatch".
        let result: DmTensor<bf16, Chip, Cluster, Slice, m![A, B % 16]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<bf16>()
            .collect::<m![A], m![B % 16]>()
            .commit_trim::<m![B % 16]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod broadcast1 {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_slice1_zero(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C % 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C % 4, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, D % 4], m![A, C % 4]>(SwitchConfig::Broadcast1 { slice1: 0, slice0: 64 })
            .collect::<m![A, C % 4], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C % 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C % 4, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, D % 4], m![A, C % 4]>(SwitchConfig::Broadcast1 { slice1: 3, slice0: 64 })
            .collect::<m![A, C % 4], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![D % 4, C % 64, A, C / 64, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![D % 4, C % 64], m![A, C / 64, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![D % 4, C % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![A, C / 64], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_broadcast_axes_not_new(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, C % 4, A, C % 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 64], m![A, C / 64, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 64, C % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, C / 64], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice0_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 4, D % 64], m![A, C / 64, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![1 # 4, D % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, C / 64], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_out_time(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, E % 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 4, C % 64], m![A, E % 4, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![1 # 4, C % 64], m![A, E % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, E % 4], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod broadcast01 {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_slice0_zero(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C / 2 % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C / 2 % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, D % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 0,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C / 2 % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C / 2 % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, D % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 3,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C / 2 % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C / 2 % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, D % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 8, D % 8, A / 2, C / 2 % 2, A % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 8, D % 8], m![A / 2, C / 2 % 2, A % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 8, D % 8], m![A / 2, C / 2 % 2, A % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A / 2, C / 2 % 2, A % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice_axes_in_broadcast(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, C % 4, A, C / 2 % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, C % 4], m![A, C / 2 % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, C % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_axes_in_broadcast(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, A % 4, A, C / 2 % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, A % 4], m![A, C / 2 % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, A % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_out_time(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, E % 4, A / 2, C / 2, A % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, E % 4], m![A / 2, C / 2, A % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, E % 4], m![A / 2, C / 2, A % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A / 2, C / 2, A % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod transpose {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_time_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 64, C % 2, C / 2 % 32, D, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 2, C / 2 % 32], m![D, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 64, C % 2, C / 2 % 32], m![D]>(SwitchConfig::Transpose { slice1: 32, slice0: 2 })
            .collect::<m![D], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_mapping(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 64, C % 2, C / 2 % 32, E % 8, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 2, C / 2 % 32], m![E % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 64, C % 2, C / 2 % 32], m![E % 8]>(SwitchConfig::Transpose { slice1: 32, slice0: 2 })
            .collect::<m![E % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_transpose_placement(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C / 8 % 16, C % 8], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C % 8, C / 128, C / 8 % 16], m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C % 8, C / 128, C / 8 % 16], m![A]>(SwitchConfig::Transpose { slice1: 16, slice0: 8 })
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod inter_transpose {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_time0(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![A, C % 32], m![C / 32 % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![A, C % 32], m![C / 32 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 32,
                time0: 0,
            })
            .collect::<m![C / 32 % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_dims(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<Slice, m![A]>(SwitchConfig::InterTranspose {
                slice1: 3,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time0_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![A, C % 32], m![C / 32 % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![A, C % 32], m![C / 32 % 8]>(SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 64,
                time0: 3,
            })
            .collect::<m![C / 32 % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C % 128], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![D % 2, A, C % 16], m![B / 16, E % 2, C / 16 % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A, B / 8], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![D % 2, A, C % 16], m![B / 16, E % 2, C / 16 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![B / 16, E % 2, C / 16 % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice0_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![B]>,
        output: &mut HbmTensor<i8, Chip, m![B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![D % 2, A, C % 16], m![B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![D % 2, A, B / 2], m![B / 8, B / 4 % 2, B % 2, C / 16 % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![B], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![D % 2, A, B / 2], m![B / 8, B / 4 % 2, B % 2, C / 16 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![B / 8, B / 4 % 2, B % 2, C / 16 % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time1_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C % 128], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, D % 2, C % 16], m![A / 2, G / 16, A % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 32, D % 2, C % 16], m![A / 2, G / 16, A % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 2, G / 16, A % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 2, A % 2, C / 16 % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 2, A % 2, C / 16 % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 2, A % 2, C / 16 % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time0_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 4, A / 2 % 2, D % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A / 2 % 2, D % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 4, A / 2 % 2, D % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice1_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, D % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, D % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 4, A % 2, D % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}
