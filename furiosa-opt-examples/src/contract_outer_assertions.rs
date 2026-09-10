use furiosa_opt_std::prelude::*;

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
type Lane = m![R];

pub mod lane_size {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_size_8(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, m![R], m![B]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = device
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

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }
}

pub mod cpacket_size {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_size_64(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = device
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

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_size_32(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, B / 2]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B / 2]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B / 2]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B / 2]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B / 2]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B / 2]>()
            .fetch_cast::<i8>()
            .collect::<m![R], m![B / 2]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = device
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![B / 2]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![B / 2]>()
            .contract_outer::<m![A], m![B / 2], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }
}

pub mod cpacket_mapping {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_one_collect_flit(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, E]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![E]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = device
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![A], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_two_collect_flits(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        output: &mut HbmTensor<i32, Chip, m![A, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R # 8]> = device
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

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }
}

pub mod time_broadcast {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_single_tiling(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, E]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![T, E]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R, T], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R, T], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, T, R # 8]> = device
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![A, T], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A, T]>()
            .contract_lane::<m![A, T], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_double_tiling(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, U, T, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, U, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, U, T, E]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![U, T, E]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R, U, T], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R, U, T], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, U, T, R # 8]> = device
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![A, U, T], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A, U, T]>()
            .contract_lane::<m![A, U, T], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_tiling_not_in_trf(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, E]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![E]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, T, R # 8]> = device
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![A, T], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A, T]>()
            .contract_lane::<m![A, T], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn valid_transposed_tiling(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, E]>,
        input_trf: &HbmTensor<i8, Chip, m![R, T, V, E]>,
        output: &mut HbmTensor<i32, Chip, m![A, V, T, R # 8]>,
    ) {
        let input_dm = input.to_dm::<Cluster, Slice, m![A, E]>(&mut device.tdma);
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, T, V, E]>(&mut device.tdma);

        let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![T, V, E]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R, T, V], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![R, T, V], m![E]>()
            .to_trf();

        let result: DmTensor<i32, Chip, Cluster, Slice, m![A, V, T, R # 8]> = device
            .main
            .begin(input_dm.view())
            .fetch::<m![A], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![E]>()
            .contract_outer::<m![A, V, T], m![E], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A, V, T]>()
            .contract_lane::<m![A, V, T], m![R # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![R # 8]>()
            .commit();

        result.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }
}

pub mod trf_mapping {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_mapping(
        device: &mut Device,
        _input: &HbmTensor<i8, Chip, m![A, B]>,
        input_trf: &HbmTensor<i8, Chip, m![R, B]>,
        _output: &mut HbmTensor<i32, Chip, m![A, 1 # 8]>,
    ) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, B]>(&mut device.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![B]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![R], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![R, B / 32], m![B % 32]>()
            .to_trf();
    }

    #[device(chip = 1)]
    pub fn valid_unit_time_lane(device: &mut Device, input_trf: &HbmTensor<i8, Chip, m![E]>) {
        let trf_dm = input_trf.to_dm::<Cluster, Slice, m![E]>(&mut device.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![1], m![E]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![1], m![E]>()
            .fetch_cast::<i8>()
            .collect::<m![1], m![E]>()
            .to_trf();
    }
}

pub mod trf_size {
    use super::*;

    #[device(chip = 1)]
    pub fn valid_to_trf_full(
        device: &mut Device,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm::<Cluster, Slice, m![A, B]>(&mut device.tdma);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![B]> = device
            .sub
            .begin(trf_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf();
    }
}
