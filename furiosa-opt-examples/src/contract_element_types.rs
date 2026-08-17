//! Contraction element-type coverage.
//!
//! One minimal single-flit `contract_outer` per DPE input type. Each kernel
//! pins the logical->DPE promotion: `i8 -> i9` and `i4 -> i5` widen implicitly
//! at the contraction boundary, while `f8`/`bf16` pass through. The reduction
//! (packet) dim is one 32-byte flit per type, so these also exercise the
//! sub-mac `CollectFlits::Flit1` path for each element size: i8/f8 = 32 elems,
//! bf16 = 16 elems, i4 = 64 elems.
//!
//! i8/f8e4m3/f8e5m2/bf16 reach EDF. i4 reaches LIR: its vISA translation is
//! complete, but LIR->EDF has incomplete RawInt5/i5 backend support (COM-63).
//! `i16`, `i32` and `f32` are not DPE inputs at all, so no kernel here covers
//! them: none has a `ContractionCast` impl, and a `contract_outer` on any of
//! them does not compile.

use furiosa_opt_std::prelude::*;

axes![A = 8, K8 = 32, K16 = 16, K4 = 64, R = 8];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];
type Lane = m![R];

/// `i8` activation x `i8` weight, promoted to `i9` x `i9` at the DPE.
#[device(chip = 1)]
pub fn i8_contract(
    ctx: &mut Context,
    input: &HbmTensor<i8, Chip, m![A, K8]>,
    input_trf: &HbmTensor<i8, Chip, m![R, K8]>,
) -> HbmTensor<i32, Chip, m![A, R]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A, K8]>(&mut ctx.tdma);
    let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, K8]>(&mut ctx.tdma);

    let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![K8]> = ctx
        .sub
        .begin(trf_dm.view())
        .fetch::<m![R], m![K8]>()
        .fetch_cast::<i8>()
        .collect::<m![R], m![K8]>()
        .to_trf();

    let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![K8]>()
        .fetch_cast::<i8>()
        .collect::<m![A], m![K8]>()
        .contract_outer::<m![A], m![K8], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A]>()
        .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// `f8e4m3` contraction; passes through to the DPE unchanged, accumulates in f32.
#[device(chip = 1)]
pub fn f8e4m3_contract(
    ctx: &mut Context,
    input: &HbmTensor<f8e4m3, Chip, m![A, K8]>,
    input_trf: &HbmTensor<f8e4m3, Chip, m![R, K8]>,
) -> HbmTensor<f32, Chip, m![A, R]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A, K8]>(&mut ctx.tdma);
    let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, K8]>(&mut ctx.tdma);

    let trf: TrfTensor<f8e4m3, Chip, Cluster, Slice, Lane, m![K8]> = ctx
        .sub
        .begin(trf_dm.view())
        .fetch::<m![R], m![K8]>()
        .fetch_cast::<f8e4m3>()
        .collect::<m![R], m![K8]>()
        .to_trf();

    let result: DmTensor<f32, Chip, Cluster, Slice, m![A, R]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![K8]>()
        .fetch_cast::<f8e4m3>()
        .collect::<m![A], m![K8]>()
        .contract_outer::<m![A], m![K8], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A]>()
        .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// Zero-point-subtracted contraction: the activation is fetched as `i8` and
/// widened to `i9` by `fetch_zero_point_sub` (subtracting a per-tensor zero
/// point), then contracted against an `i8` weight via the mixed i8/i9 family.
/// The i9 stream can only reach `contract_outer` (it is not committable).
#[device(chip = 1)]
pub fn zero_point_sub_contract(
    ctx: &mut Context,
    input: &HbmTensor<i8, Chip, m![A, K8]>,
    input_trf: &HbmTensor<i8, Chip, m![R, K8]>,
) -> HbmTensor<i32, Chip, m![A, R]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A, K8]>(&mut ctx.tdma);
    let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, K8]>(&mut ctx.tdma);

    let trf: TrfTensor<i8, Chip, Cluster, Slice, Lane, m![K8]> = ctx
        .sub
        .begin(trf_dm.view())
        .fetch::<m![R], m![K8]>()
        .fetch_cast::<i8>()
        .collect::<m![R], m![K8]>()
        .to_trf();

    let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![K8]>()
        .fetch_zero_point_sub::<i9>(3)
        .collect::<m![A], m![K8]>()
        .contract_outer::<m![A], m![K8], _, _, i8>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A]>()
        .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// `i4` activation x `i4` weight, promoted to `i5` x `i5` at the contraction engine.
/// One flit is 64 i4 elements (half a byte each), sub-mac padded to the i4 mac.
#[device(chip = 1)]
pub fn i4_contract(
    ctx: &mut Context,
    input: &HbmTensor<i4, Chip, m![A, K4]>,
    input_trf: &HbmTensor<i4, Chip, m![R, K4]>,
) -> HbmTensor<i32, Chip, m![A, R]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A, K4]>(&mut ctx.tdma);
    let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, K4]>(&mut ctx.tdma);

    let trf: TrfTensor<i4, Chip, Cluster, Slice, Lane, m![K4]> = ctx
        .sub
        .begin(trf_dm.view())
        .fetch::<m![R], m![K4]>()
        .fetch_cast::<i4>()
        .collect::<m![R], m![K4]>()
        .to_trf();

    let result: DmTensor<i32, Chip, Cluster, Slice, m![A, R]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![K4]>()
        .fetch_cast::<i4>()
        .collect::<m![A], m![K4]>()
        .contract_outer::<m![A], m![K4], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A]>()
        .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// `f8e5m2` contraction; passes through to the DPE unchanged, accumulates in f32.
#[device(chip = 1)]
pub fn f8e5m2_contract(
    ctx: &mut Context,
    input: &HbmTensor<f8e5m2, Chip, m![A, K8]>,
    input_trf: &HbmTensor<f8e5m2, Chip, m![R, K8]>,
) -> HbmTensor<f32, Chip, m![A, R]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A, K8]>(&mut ctx.tdma);
    let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, K8]>(&mut ctx.tdma);

    let trf: TrfTensor<f8e5m2, Chip, Cluster, Slice, Lane, m![K8]> = ctx
        .sub
        .begin(trf_dm.view())
        .fetch::<m![R], m![K8]>()
        .fetch_cast::<f8e5m2>()
        .collect::<m![R], m![K8]>()
        .to_trf();

    let result: DmTensor<f32, Chip, Cluster, Slice, m![A, R]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![K8]>()
        .fetch_cast::<f8e5m2>()
        .collect::<m![A], m![K8]>()
        .contract_outer::<m![A], m![K8], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A]>()
        .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}

/// `bf16` contraction; passes through to the DPE unchanged, accumulates in f32.
/// One flit is 16 bf16 elements (2 bytes each).
#[device(chip = 1)]
pub fn bf16_contract(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, K16]>,
    input_trf: &HbmTensor<bf16, Chip, m![R, K16]>,
) -> HbmTensor<f32, Chip, m![A, R]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A, K16]>(&mut ctx.tdma);
    let trf_dm = input_trf.to_dm::<Cluster, Slice, m![R, K16]>(&mut ctx.tdma);

    let trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K16]> = ctx
        .sub
        .begin(trf_dm.view())
        .fetch::<m![R], m![K16]>()
        .fetch_cast::<bf16>()
        .collect::<m![R], m![K16]>()
        .to_trf();

    let result: DmTensor<f32, Chip, Cluster, Slice, m![A, R]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![K16]>()
        .fetch_cast::<bf16>()
        .collect::<m![A], m![K16]>()
        .contract_outer::<m![A], m![K16], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A]>()
        .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
