use furiosa_visa_std::prelude::*;

axes![A = 8, B = 32];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

pub mod cluster_size {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_cluster_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<m![1 # 4], Slice, m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, m![1 # 4], Slice, m![A, B]> =
            ctx.main.begin(input_dm.view()).fetch::<i8, m![A], m![B]>().commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_cluster_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B]> =
            ctx.main.begin(input_dm.view()).fetch::<i8, m![A], m![B]>().commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

pub mod slice_size {
    use super::*;

    #[device(chip = 1)]
    pub fn invalid_slice_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, m![1 # 512], m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, m![1 # 512], m![A, B]> =
            ctx.main.begin(input_dm.view()).fetch::<i8, m![A], m![B]>().commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_slice_size(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let result: DmTensor<i8, Chip, Cluster, Slice, m![A, B]> =
            ctx.main.begin(input_dm.view()).fetch::<i8, m![A], m![B]>().commit(0);

        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}
