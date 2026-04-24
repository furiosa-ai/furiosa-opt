use furiosa_visa_std::prelude::*;

axes![
    A = 8,
    B = 64,
    C = 128,
    D = 64,
    E = 32,
    F = 6,
    R = 8,
    T = 5,
    U = 3,
    V = 2
];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];
type Row = m![R];

pub mod row_size {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_size_1(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![B]>,
        output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![1], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![1], m![B]>()
            .collect::<m![B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, 1 # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![1 # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_size_2(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R / 4, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R / 4 # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R / 4, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![R / 4], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R / 4], m![B]>()
            .collect::<m![R / 4, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R / 4 # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R / 4 # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_size_4(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R / 2, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R / 2 # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R / 2, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![R / 2], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R / 2], m![B]>()
            .collect::<m![R / 2, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R / 2 # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R / 2 # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_size_8(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![R], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_size_3(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![1 # 3], m![R, B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_size_16(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![1 # 16], m![R, B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod cpacket_size {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_size_64(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_size_32(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B / 2]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B / 2]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B / 2]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B / 2]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B / 2]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B / 2]>()
            .collect::<m![R], m![B / 2]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B / 2]>()
            .collect::<m![A], m![B / 2]>()
            .align::<m![A], m![B / 2], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_size_128(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, C]>,
        input_trf: &HbmTensor<i8, Chip, m![R, C]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, C]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, C]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![C]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![C]>()
            .collect::<m![R, C / 32], m![C % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![C]>()
            .collect::<m![A, C / 32], m![C % 32]>()
            .align::<m![A], m![C], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod cpacket_mapping {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_one_collect_flit(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![E]>()
            .collect::<m![R], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![A], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_two_collect_flits(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_mapping(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![D], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_one_collect_flit_no_padding(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![A / 4, B % 32], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_one_collect_flit_no_padding_reversed(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .align::<m![A], m![B % 32, A / 4], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod time_broadcast {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_single_tiling(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![T, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R, T], m![E]>()
            .collect::<m![R, T], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, T, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![A, T], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A, T], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_double_tiling(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, U, T, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, U, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, U, T, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![U, T, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R, U, T], m![E]>()
            .collect::<m![R, U, T], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, U, T, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![A, U, T], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A, U, T], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_tiling_not_in_trf(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![E]>()
            .collect::<m![R], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, T, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![A, T], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A, T], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_transposed_tiling(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, V, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, V, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, V, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![T, V, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R, T, V], m![E]>()
            .collect::<m![R, T, V], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, V, T, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![A, V, T], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![A, V, T], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_time_mismatch(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, E]>,
        output: &mut HbmTensor<i32, Chip, m![F, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![E]>()
            .collect::<m![R], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![F, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![F], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![F], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_swapped_time_axes(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, V, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, V, E]>,
        output: &mut HbmTensor<i32, Chip, m![V, A, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, V, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, V, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![T, V, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R, T, V], m![E]>()
            .collect::<m![R, T, V], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![V, A, T, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A, V], m![E]>()
            .collect::<m![A, V], m![E]>()
            .align::<m![V, A, T], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![V, A, T], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn invalid_tiling_not_innermost(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, E]>,
        output: &mut HbmTensor<i32, Chip, m![T, A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut ctx.tdma, 0);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, E]>(&mut ctx.tdma, 0);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![T, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R, T], m![E]>()
            .collect::<m![R, T], m![E]>()
            .to_trf(TrfAddress::Full);

        let result: DmTensor<i32, Chip, Cluster, Slice, m![T, A, R # 8]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<i8, m![A], m![E]>()
            .collect::<m![A], m![E]>()
            .align::<m![T, A], m![E # 64], _, _>(&trf)
            .contract::<m![1]>()
            .accumulate::<m![T, A], m![R # 8]>(AccumulationKind::Interleaved)
            .commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod trf_mapping {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_mapping(
        ctx: &mut Context,
        _input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        _output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);
    }

    #[device(chip = 1)]
    pub fn valid_unit_time_row(ctx: &mut Context, input_trf: &HbmTensor<i8, Chip, m![E]>) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![E]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![1], m![E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![1], m![E]>()
            .collect::<m![1], m![E]>()
            .to_trf(TrfAddress::Full);
    }

    #[device(chip = 1)]
    pub fn invalid_row_mapping(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![E / 4], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);
    }

    #[device(chip = 1)]
    pub fn invalid_mapping(
        ctx: &mut Context,
        _input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        _output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, Row, m![A, C]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![R], m![B]>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);
    }

    #[device(chip = 1)]
    pub fn invalid_row_not_divisible_by_time(ctx: &mut Context, input_trf: &HbmTensor<i8, Chip, m![F, E]>) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![F, E]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![R / 2], m![F, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![F], m![E]>()
            .collect::<m![F], m![E]>()
            .to_trf(TrfAddress::Full);
    }
}

pub mod trf_row_time {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_row_exceeds_time(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A / 4, E]>,
        _output: &mut HbmTensor<i8, Chip, m![A / 4, E]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A / 4, E]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![R / 2], m![A / 4, E]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![A / 4], m![E]>()
            .collect::<m![A / 4], m![E]>()
            .to_trf(TrfAddress::Full);
    }
}

pub mod trf_size {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_to_trf_full(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 128 * 1024);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);
    }

    #[device(chip = 1)]
    pub fn valid_to_trf_half(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 128 * 1024);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::FirstHalf);
    }

    #[device(chip = 1)]
    pub fn invalid_to_trf_full(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 128 * 1024);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![A, B, C]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::Full);
    }

    #[device(chip = 1)]
    pub fn invalid_to_trf_half(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 128 * 1024);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![B, C]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<i8, m![A], m![B]>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf(TrfAddress::FirstHalf);
    }
}
