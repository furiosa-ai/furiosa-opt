use furiosa_opt_std::prelude::*;

axes![A = 8, B = 32, C = 256, D = 64, E = 1024, F = 1024, G = 32];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

pub mod alignment {
    use super::*;

    #[device(chip = 1)]
    pub fn aligned_fetch_packet_i8(
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
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn aligned_fetch_packet_bf16(
        ctx: &mut Context,
        input: &HbmTensor<bf16, Chip, m![A, B]>,
        output: &mut HbmTensor<bf16, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<bf16, Chip, Cluster, Slice, m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<bf16>()
            .collect::<m![A, B / 16], m![B % 16]>()
            .commit_trim::<m![B % 16]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod packet {
    use super::*;

    #[device(chip = 1)]
    pub fn packet_padding_unchanged(
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
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_padding_added_in_switch(
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
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_nested_padding(
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
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn packet_restructuring(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, C]>,
        output: &mut HbmTensor<i8, Chip, m![A, C / 16, C % 16]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, C]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, C / 16, C % 16]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![C]>()
            .fetch_cast::<i8>()
            .collect::<m![A, C / 32], m![C % 32]>()
            .commit_trim::<m![C % 32]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_padding(
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
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod slice {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_matching_slice_sizes(
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
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod broadcast1 {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_basic(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 16, 1 # 4, C % 4, A, C / 4 % 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 16, 1 # 4, C % 4], m![A, C / 4 % 4, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 16, 1 # 4, C % 4], m![A, C / 4 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 4 })
            .collect::<m![A, C / 4 % 4], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_degenerate(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, 1 # 4, A, C % 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 4, C % 64], m![A, C / 64, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![1 # 4, C % 64], m![A, C / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 64 })
            .collect::<m![A, C / 64], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod broadcast01 {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_only_slice1(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![B]>,
        output: &mut HbmTensor<i8, Chip, m![F / 4, E / 4, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![E / 4], m![1, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![F / 4], m![E / 4, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![1], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![F / 4], m![E / 4]>(SwitchConfig::Broadcast01 {
                slice1: 256,
                slice0: 1,
                time0: 1,
            })
            .collect::<m![E / 4], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_with_time0(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, D % 4, A / 2, C / 2 % 2, A % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, D % 4], m![A / 2, C / 2 % 2, A % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, D % 4], m![A / 2, C / 2 % 2, A % 2, C % 2]>(SwitchConfig::Broadcast01 {
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
    pub fn valid_broadcast_with_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 4, 1 # 4, A, C / 2 % 2, C % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 4, 1 # 4], m![A, C / 2 % 2, C % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 4, 1 # 4], m![A, C / 2 % 2, C % 2]>(SwitchConfig::Broadcast01 {
                slice1: 2,
                slice0: 2,
                time0: 1,
            })
            .collect::<m![A, C / 2 % 2, C % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod transpose {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_single_axis(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 64, C % 2, C / 2 % 32, A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 64, C % 2, C / 2 % 32], m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 64, C % 2, C / 2 % 32], m![A]>(SwitchConfig::Transpose { slice1: 32, slice0: 2 })
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_three_axes(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 128, C % 8, C / 8 % 16, A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 128, C / 8 % 16, C % 8], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 128, C % 8, C / 8 % 16], m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 128, C % 8, C / 8 % 16], m![A]>(SwitchConfig::Transpose { slice1: 16, slice0: 8 })
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_split_inner(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 16, C % 4, C / 4 % 4, A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C / 16, C % 16], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 16, C % 4, C / 4 % 4], m![A, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 16, C % 4, C / 4 % 4], m![A]>(SwitchConfig::Transpose { slice1: 4, slice0: 4 })
            .collect::<m![A], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod inter_transpose {
    use super::*;

    #[device(chip = 1)]
    pub fn valid(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![C / 32, A / 2 % 2, C % 16, A / 4, A % 2, C / 16 % 2, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![C], m![A, B]>(&mut ctx.tdma);

        let result: DmTensor<i8, Chip, Cluster, m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, C / 16 % 2, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, C / 16 % 2]>(SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 16,
                time0: 2,
            })
            .collect::<m![A / 4, A % 2, C / 16 % 2], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_degenerate(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, C % 32, C / 32 % 8, B]>,
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
                time0: 1,
            })
            .collect::<m![C / 32 % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit();

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}
