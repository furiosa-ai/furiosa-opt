use furiosa_visa_std::prelude::*;

axes![A = 8, B = 32, C = 256, D = 64, E = 1024, F = 1024, G = 32];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

pub mod alignment {
    use super::*;

    #[device(chip = 1)]
    pub fn unaligned_fetch_packet_i4(
        ctx: &mut Context,
        input: &HbmTensor<i4, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i4, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 64]>(&mut ctx.tdma, 0);

        let result: DmTensor<i4, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i4, m![A], m![B # 32]>()
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn aligned_fetch_packet_i4(
        ctx: &mut Context,
        input: &HbmTensor<i4, Chip, m![A, B # 64]>,
        output: &mut HbmTensor<i4, Chip, m![A, B # 64]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 64]>(&mut ctx.tdma, 0);

        let result: DmTensor<i4, Chip, Cluster, Slice, m![A, B # 64]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i4, m![A], m![B # 64]>()
            .collect::<m![A], m![B # 64]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn unaligned_fetch_packet_i8(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B / 2]>,
        output: &mut HbmTensor<i8, Chip, m![A, B / 2]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B / 2]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B / 2]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B / 2]>()
            .collect::<m![A], m![B / 2]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn aligned_fetch_packet_i8(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn unaligned_fetch_packet_bf16(
        ctx: &mut Context,
        input: &HbmTensor<bf16, Chip, m![A, B / 4]>,
        output: &mut HbmTensor<bf16, Chip, m![A, B / 4]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B / 4]>(&mut ctx.tdma, 0);

        let result: DmTensor<bf16, Chip, Cluster, Slice, m![A, B / 4]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<bf16, m![A], m![B / 4]>()
            .collect::<m![A], m![B / 4]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn aligned_fetch_packet_bf16(
        ctx: &mut Context,
        input: &HbmTensor<bf16, Chip, m![A, B]>,
        output: &mut HbmTensor<bf16, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<bf16, Chip, Cluster, Slice, m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<bf16, m![A], m![B]>()
            .collect::<m![A, B / 16], m![B % 16]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod packet {
    use super::*;

    #[device(chip = 1)]
    pub fn packet_padding_unchanged(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_padding_added_in_switch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_nested_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .collect::<m![A], m![B # 32 # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_restructuring(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, C]>,
        output: &mut HbmTensor<i8, Chip, m![A, C / 16, C % 16]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, C]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, C / 16, C % 16]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![C]>()
            .collect::<m![A, C / 32], m![C % 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .collect::<m![A], m![A # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn collect_time_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<bf16, Chip, m![A, B]>,
        output: &mut HbmTensor<bf16, Chip, m![A, B % 16]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        // bf16 B=32 elements = 64 bytes = 2 flits.
        // Correct Time2 would be m![A, B / 16] (absorbing outer flit into time).
        // Here we provide m![A] which is wrong → "Collect time mismatch".
        let result: DmTensor<bf16, Chip, Cluster, Slice, m![A, B % 16]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<bf16, m![A], m![B]>()
            .collect::<m![A], m![B % 16]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_excessive_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 64]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 64]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A], m![B # 64]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod slice {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_matching_slice_sizes(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod broadcast1 {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_basic(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 16, 1 # 4, C % 4], m![A, C / 4 % 4, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 16, 1 # 4, C % 4], m![A, C / 4 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 4 })
            .collect::<m![A, C / 4 % 4], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_degenerate(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, 1 # 4, A, C % 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 4, C % 64], m![A, C / 64, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![1 # 4, C % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, C / 64], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice1_zero(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C % 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C % 4, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, D % 4], m![A, C % 4]>(SwitchConfig::Broadcast1 { slice1: 0, slice0: 64 })
            .collect::<m![A, C % 4], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C % 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C % 4, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, D % 4], m![A, C % 4]>(SwitchConfig::Broadcast1 { slice1: 3, slice0: 64 })
            .collect::<m![A, C % 4], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![D % 4, C % 64, A, C / 64, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![D % 4, C % 64], m![A, C / 64, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![D % 4, C % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![A, C / 64], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_broadcast_axes_not_new(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, C % 4, A, C % 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 64], m![A, C / 64, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 64, C % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, C / 64], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice0_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 4, D % 64], m![A, C / 64, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![1 # 4, D % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, C / 64], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_out_time(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, E % 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 4, C % 64], m![A, E % 4, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![1 # 4, C % 64], m![A, E % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, E % 4], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod broadcast01 {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_only_slice1(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![F / 4, E / 4, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![E / 4], m![1, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![F / 4], m![E / 4, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![1], m![B # 32]>()
            .switch::<m![F / 4], m![E / 4]>(SwitchConfig::Broadcast01 {
                slice1: 256,
                slice0: 1,
                time0: 1,
            })
            .collect::<m![E / 4], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_with_time0(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A / 2, C / 2 % 2, A % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A / 2, C / 2 % 2, A % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, D % 4], m![A / 2, C / 2 % 2, A % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A / 2, C / 2 % 2, A % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_broadcast_with_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, 1 # 4], m![A, C / 2 % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, 1 # 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice0_zero(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C / 2 % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C / 2 % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, D % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 0,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C / 2 % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C / 2 % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, D % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 3,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A, C / 2 % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A, C / 2 % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, D % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 8, D % 8, A / 2, C / 2 % 2, A % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 8, D % 8], m![A / 2, C / 2 % 2, A % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 8, D % 8], m![A / 2, C / 2 % 2, A % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A / 2, C / 2 % 2, A % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice_axes_in_broadcast(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, C % 4, A, C / 2 % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, C % 4], m![A, C / 2 % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, C % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_axes_in_broadcast(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, A % 4, A, C / 2 % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, A % 4], m![A, C / 2 % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, A % 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_out_time(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, E % 4, A / 2, C / 2, A % 2, C % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, E % 4], m![A / 2, C / 2, A % 2, C % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 4, E % 4], m![A / 2, C / 2, A % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 2,
            })
            .collect::<m![A / 2, C / 2, A % 2, C % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod transpose {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_single_axis(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 64, C % 2, C / 2 % 32, A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 2, C / 2 % 32], m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 64, C % 2, C / 2 % 32], m![A]>(SwitchConfig::Transpose { slice1: 32, slice0: 2 })
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_three_axes(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 128, C % 8, C / 8 % 16, A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C / 8 % 16, C % 8], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 128, C % 8, C / 8 % 16], m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 128, C % 8, C / 8 % 16], m![A]>(SwitchConfig::Transpose { slice1: 16, slice0: 8 })
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_split_inner(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 16, C % 4, C / 4 % 4, A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 16, C % 16], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 16, C % 4, C / 4 % 4], m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 16, C % 4, C / 4 % 4], m![A]>(SwitchConfig::Transpose { slice1: 4, slice0: 4 })
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 64, C % 2, C / 2 % 32, D, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 2, C / 2 % 32], m![D, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 64, C % 2, C / 2 % 32], m![D]>(SwitchConfig::Transpose { slice1: 32, slice0: 2 })
            .collect::<m![D], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_mapping(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 64, C % 2, C / 2 % 32, E % 8, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 2, C / 2 % 32], m![E % 8, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 64, C % 2, C / 2 % 32], m![E % 8]>(SwitchConfig::Transpose { slice1: 32, slice0: 2 })
            .collect::<m![E % 8], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_transpose_placement(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C / 8 % 16, C % 8], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C % 8, C / 128, C / 8 % 16], m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C % 8, C / 128, C / 8 % 16], m![A]>(SwitchConfig::Transpose { slice1: 16, slice0: 8 })
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod inter_transpose {
    use super::*;

    #[device(chip = 1)]
    pub fn valid(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![C / 32, A / 2 % 2, C % 16, A / 4, A % 2, C / 16 % 2, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, C / 16 % 2, B # 32]> =
            ctx.main
                .begin(input_dm.view())
                .fetch::<i8, m![A], m![B # 32]>()
                .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, C / 16 % 2]>(SwitchConfig::InterTranspose {
                    slice1: 2,
                    slice0: 16,
                    time0: 2,
                })
                .collect::<m![A / 4, A % 2, C / 16 % 2], m![B # 32]>()
                .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_degenerate(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, C % 32, C / 32 % 8, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![A, C % 32], m![C / 32 % 8, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![A, C % 32], m![C / 32 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 32,
                time0: 1,
            })
            .collect::<m![C / 32 % 8], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time0(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![A, C % 32], m![C / 32 % 8, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![A, C % 32], m![C / 32 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 32,
                time0: 0,
            })
            .collect::<m![C / 32 % 8], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_dims(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<Slice, m![A]>(SwitchConfig::InterTranspose {
                slice1: 3,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time0_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![A, C % 32], m![C / 32 % 8, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![A, C % 32], m![C / 32 % 8]>(SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 64,
                time0: 3,
            })
            .collect::<m![C / 32 % 8], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C % 128], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![D % 2, A, C % 16], m![B / 16, E % 2, C / 16 % 8, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A, B / 8], m![B # 32]>()
            .switch::<m![D % 2, A, C % 16], m![B / 16, E % 2, C / 16 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![B / 16, E % 2, C / 16 % 8], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice0_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![D % 2, A, C % 16], m![B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![D % 2, A, B / 2], m![B / 8, B / 4 % 2, B % 2, C / 16 % 8, B # 32]> =
            ctx.main
                .begin(input_dm.view())
                .fetch::<i8, m![B], m![B # 32]>()
                .switch::<m![D % 2, A, B / 2], m![B / 8, B / 4 % 2, B % 2, C / 16 % 8]>(SwitchConfig::InterTranspose {
                    slice1: 8,
                    slice0: 16,
                    time0: 2,
                })
                .collect::<m![B / 8, B / 4 % 2, B % 2, C / 16 % 8], m![B # 32]>()
                .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time1_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C % 128], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, D % 2, C % 16], m![A / 2, G / 16, A % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 32, D % 2, C % 16], m![A / 2, G / 16, A % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 2, G / 16, A % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time2_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 2, A % 2, C / 16 % 2, B # 32]> =
            ctx.main
                .begin(input_dm.view())
                .fetch::<i8, m![A], m![B # 32]>()
                .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 2, A % 2, C / 16 % 2]>(SwitchConfig::InterTranspose {
                    slice1: 2,
                    slice0: 16,
                    time0: 2,
                })
                .collect::<m![A / 2, A % 2, C / 16 % 2], m![B # 32]>()
                .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time0_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 4, A / 2 % 2, D % 2, B # 32]> =
            ctx.main
                .begin(input_dm.view())
                .fetch::<i8, m![A], m![B # 32]>()
                .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A / 2 % 2, D % 2]>(SwitchConfig::InterTranspose {
                    slice1: 2,
                    slice0: 16,
                    time0: 2,
                })
                .collect::<m![A / 4, A / 2 % 2, D % 2], m![B # 32]>()
                .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_slice1_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B # 32]>,
        output: &mut HbmTensor<i8, Chip, m![A, B # 32]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B # 32]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, D % 2, B # 32]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B # 32]>()
            .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, D % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 4, A % 2, D % 2], m![B # 32]>()
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}
