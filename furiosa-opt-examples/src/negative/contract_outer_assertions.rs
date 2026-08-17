//! Rejection fixtures for the `contract_outer` operand assertions: each kernel states a lane,
//! packet, time-broadcast or TRF mapping that the compiler must refuse, mirroring a `valid_*` twin
//! in the public [`contract_outer_assertions`](crate::contract_outer_assertions) module.
//!
//! The kernels still emulate (the assertions are device-translation checks), so the answer-key
//! tests run them; only compilation must fail. See [`super`] for the snapshot contract.

use furiosa_opt_std::prelude::*;

// The axes come from the public twin, so a fixture and the `valid_*` kernel it contrasts with are
// stated over the very same axis types (and produce the same axis labels in a diagnostic).
pub use crate::contract_outer_assertions::{A, B, C, D, E, F, R, T, V};

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];
type Lane = m![R];

pub mod lane_size {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_size_3(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![1 # 3], m![R, B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .contract_outer::<m![A], m![B], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_size_16(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![1 # 16], m![R, B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .contract_outer::<m![A], m![B], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod cpacket_mapping {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_mapping(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .contract_outer::<m![A], m![D], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_one_collect_flit_no_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .contract_outer::<m![A], m![A / 4, B % 32], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_one_collect_flit_no_padding_reversed(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .contract_outer::<m![A], m![B % 32, A / 4], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod time_broadcast {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_time_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, E]>,
        output: &mut HbmTensor<i32, Chip, m![F, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, E]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![F, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![F], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![F]>()
            .contract_lane::<m![F], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_swapped_time_axes(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, V, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, V, E]>,
        output: &mut HbmTensor<i32, Chip, m![V, A, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, V, E]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, V, E]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![T, V, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R, T, V], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R, T, V], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![V, A, T, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A, V], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A, V], m![E]>()
            .contract_outer::<m![V, A, T], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![V, A, T]>()
            .contract_lane::<m![V, A, T], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_tiling_not_innermost(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, E]>,
        output: &mut HbmTensor<i32, Chip, m![T, A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, E]>(&mut ctx.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![T, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R, T], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R, T], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![T, A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![T, A], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![T, A]>()
            .contract_lane::<m![T, A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod trf_mapping {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_lane_mapping(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![E / 4], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf();
    }

    #[device(chip = 1)]
    pub fn invalid_mapping(
        ctx: &mut Context,
        _input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        _output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![A, C]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();
    }

    #[device(chip = 1)]
    pub fn invalid_lane_not_divisible_by_time(ctx: &mut Context, input_trf: &HbmTensor<i8, Chip, m![F, E]>) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![F, E]>(&mut ctx.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![R / 2], m![F, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![F], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![F], m![E]>()
            .to_trf();
    }
}

pub mod trf_lane_time {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_lane_exceeds_time(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A / 4, E]>,
        _output: &mut HbmTensor<i8, Chip, m![A / 4, E]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A / 4, E]>(&mut ctx.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![R / 2], m![A / 4, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![A / 4], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A / 4], m![E]>()
            .to_trf();
    }
}

pub mod trf_size {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_to_trf_full(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![A, B, C]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf();
    }
}
