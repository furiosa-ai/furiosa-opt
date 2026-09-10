//! Axis lifting on a contraction weight stored in TRF.

use furiosa_opt_std::prelude::*;

axes![Row = 128, Grp = 2, Act = 8, Dot = 32, Col = 8];

type Chip = m![1];
type Cluster = m![1 # 2];
type Replicated = m![Row, 2];
type Lifted = m![Row, Grp];
type Lane = m![Col];

/// Lifts `Grp` onto `Slice` while loading a contraction weight into TRF.
#[device(chip = 1)]
pub fn lift_to_trf_contract(
    device: &mut Device,
    act: &HbmTensor<i8, Chip, m![Row, Grp, Act, Dot]>,
    weight: &HbmTensor<i8, Chip, m![Row, Grp, Col, Dot]>,
) -> HbmTensor<i32, Chip, m![Row, Grp, Act, Col]> {
    let act_dm = act.to_dm::<Cluster, Lifted, m![Act, Dot]>(&mut device.tdma);
    let weight_dm = weight.to_dm::<Cluster, Replicated, m![Grp, Col, Dot]>(&mut device.tdma);

    let trf: TrfTensor<i8, Chip, Cluster, Lifted, Lane, m![Dot]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Grp, Col], m![Dot]>()
        .fetch_slice_lift::<Lifted, m![Col]>()
        .collect::<m![Col], m![Dot]>()
        .to_trf();

    let result: DmTensor<i32, Chip, Cluster, Lifted, m![Act, Col]> = device
        .main
        .begin(act_dm.view())
        .fetch::<m![Act], m![Dot]>()
        .collect::<m![Act], m![Dot]>()
        .contract_outer::<m![Act], m![Dot], _, _, i8>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Act]>()
        .contract_lane::<m![Act], m![Col]>(LaneMode::Interleaved)
        .commit_trim::<m![Col]>()
        .commit();

    result.to_hbm::<m![Row, Grp, Act, Col]>(&mut device.tdma)
}
