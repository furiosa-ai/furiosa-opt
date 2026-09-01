//! Legal VRF operand shapes: an operand is partitioned like the stream it feeds, and
//! [`VrfTensor::reshape`] restates one naming of the same slices as another. The shapes that are
//! refused are [`negative::vector_engine`](crate::negative::vector_engine).

use furiosa_opt_std::prelude::*;

axes![W = 256, X = 16, Y = 16, V = 64, B = 32, N = 128];

type Chip = m![1];
type Cluster = m![1 # 2];
/// The single-cluster peer of [`Cluster`], for the 64-slice kernel below.
type Cluster1 = m![1];
/// The operand's own partition: it does not vary with the slice, so its 256 copies carry no axis.
/// `256` is `W`'s size, which [`vrf_slice_reshape`] relabels it to.
type Broadcast256 = m![256];
/// The stream's partition: one `W` row per slice.
type Slice = m![W];
/// The same 256 slices, grouped in pairs: `(w / 2) * 2 + (w % 2) == w`, so it is a regroup.
type SliceGrouped = m![W / 2, W % 2];
/// The same 256 slices under two other axes: `x * 16 + y` is the slice `W = x * 16 + y` is, so it is
/// a renaming rather than a decomposition of `W`.
type SliceRenamed = m![X, Y];
/// [`Slice`] at the 64-slice topology, where one cluster covers the whole partition.
type Slice64 = m![V];
/// [`Broadcast256`] at the 64-slice topology.
type Broadcast64 = m![64];

/// Loads a broadcast operand into the VRF: its source carries no slice axis, so the DMA replicated it
/// and every slice holds the same `B` vector. The kernel that reads it names those copies with
/// [`VrfTensor::reshape`].
fn broadcast_operand<Cluster: M, Slice: M>(
    ctx: &mut Context,
    operand_dm: &DmTensor<i32, Chip, Cluster, Slice, m![B]>,
) -> VrfTensor<i32, Chip, Cluster, Slice, m![B]> {
    ctx.sub
        .begin(operand_dm.view())
        .fetch::<m![1], m![B]>()
        .fetch_cast::<i32>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf()
}

/// Loads a per-slice operand into the VRF: every slice holds a different `B` row.
fn per_slice_operand<Cluster: M, Slice: M>(
    ctx: &mut Context,
    operand_dm: &DmTensor<i32, Chip, Cluster, Slice, m![B]>,
) -> VrfTensor<i32, Chip, Cluster, Slice, m![B]> {
    ctx.sub
        .begin(operand_dm.view())
        .fetch::<m![B / 8], m![B % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf()
}

/// Adds the operand to every row of the stream. The four kernels below share this chain exactly and
/// reach it with the operand already at the stream's `Slice`, so how it got there is the one thing
/// each of them shows.
fn add_operand<Cluster: M, Slice: M>(
    ctx: &mut Context,
    input_dm: &DmTensor<i32, Chip, Cluster, Slice, m![B]>,
    operand_vrf: &VrfTensor<i32, Chip, Cluster, Slice, m![B]>,
) -> DmTensor<i32, Chip, Cluster, Slice, m![B]> {
    ctx.main
        .begin(input_dm.view())
        .fetch::<m![B / 8], m![B % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, operand_vrf)
        .vector_final()
        .commit_trim::<m![B % 8]>()
        .commit()
}

/// Adds a slice-invariant operand to every row, reshaping the operand onto the stream's partition.
#[device(chip = 1)]
pub fn vrf_slice_reshape(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![W, B]>,
    operand: &HbmTensor<i32, Chip, m![B]>,
) -> HbmTensor<i32, Chip, m![W, B]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![B]>(&mut ctx.tdma);
    let operand_dm = operand.to_dm::<Cluster, Broadcast256, m![B]>(&mut ctx.tdma);
    let operand_vrf = broadcast_operand(ctx, &operand_dm);

    // Name the copy each slice already holds, so the operand feeds a `W`-partitioned stream.
    let operand_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![B]> = unsafe { operand_vrf.reshape() };

    add_operand(ctx, &input_dm, &operand_vrf).to_hbm(&mut ctx.tdma)
}

/// Adds a per-slice operand, regrouping the stream's own axis into a pair.
///
/// `SliceGrouped` decomposes the very axis `Slice` names, so slice `w` holds row `w` either way. What
/// this pins is that a composite `Slice` still meets the partition rule; [`vrf_slice_rename`] is the
/// case where the reshape has something to get wrong.
#[device(chip = 1)]
pub fn vrf_slice_regroup(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![W, B]>,
    operand: &HbmTensor<i32, Chip, m![W, B]>,
) -> HbmTensor<i32, Chip, m![W, B]> {
    let input_dm = input.to_dm::<Cluster, SliceGrouped, m![B]>(&mut ctx.tdma);
    let operand_dm = operand.to_dm::<Cluster, Slice, m![B]>(&mut ctx.tdma);
    let operand_vrf = per_slice_operand(ctx, &operand_dm);

    // Same slices, grouped in pairs: the regroup keeps slice `w` holding row `w`.
    let operand_vrf: VrfTensor<i32, Chip, Cluster, SliceGrouped, m![B]> = unsafe { operand_vrf.reshape() };

    add_operand(ctx, &input_dm, &operand_vrf).to_hbm(&mut ctx.tdma)
}

/// [`vrf_slice_reshape`] at the 64-slice topology, which is the one a default NPU config has.
///
/// The kernels above need 512 slices, so their `compare_edf` tests skip unless the config supplies
/// them; this one is the same relabel where the comparison actually runs.
///
/// One PE, since 64 slices is what one covers: a DM allocation spans the whole device, so the
/// single-cluster mapping below only adds up under `pe = 1`.
#[device(chip = 1, pe = 1)]
pub fn vrf_slice_reshape_64(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![V, B]>,
    operand: &HbmTensor<i32, Chip, m![B]>,
) -> HbmTensor<i32, Chip, m![V, B]> {
    let input_dm = input.to_dm::<Cluster1, Slice64, m![B]>(&mut ctx.tdma);
    let operand_dm = operand.to_dm::<Cluster1, Broadcast64, m![B]>(&mut ctx.tdma);
    let operand_vrf = broadcast_operand(ctx, &operand_dm);

    let operand_vrf: VrfTensor<i32, Chip, Cluster1, Slice64, m![B]> = unsafe { operand_vrf.reshape() };

    add_operand(ctx, &input_dm, &operand_vrf).to_hbm(&mut ctx.tdma)
}

/// Adds a per-slice operand written under `X` / `Y`, renamed onto the stream's `W`.
///
/// The slice-sensitive case: the reshape claims that slice `x * 16 + y` is slice `w = x * 16 + y`,
/// and every slice holds a DIFFERENT operand row, so a relabel that swapped the two axes or shifted
/// the slice identity reads another slice's row and the answer key says so.
#[device(chip = 1)]
pub fn vrf_slice_rename(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![W, B]>,
    operand: &HbmTensor<i32, Chip, m![X, Y, B]>,
) -> HbmTensor<i32, Chip, m![W, B]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![B]>(&mut ctx.tdma);
    let operand_dm = operand.to_dm::<Cluster, SliceRenamed, m![B]>(&mut ctx.tdma);
    let operand_vrf = per_slice_operand(ctx, &operand_dm);

    // The same 256 slices under the stream's own axis: slice `x * 16 + y` keeps the row it holds.
    let operand_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![B]> = unsafe { operand_vrf.reshape() };

    add_operand(ctx, &input_dm, &operand_vrf).to_hbm(&mut ctx.tdma)
}

/// Divides every element by the scale its group of 16 shares.
///
/// The operand is COARSER than the stream: `m![N / 16]` steps once per 16 stream elements, so the
/// four lanes of one `vector_narrow_split` access all land on the same operand cell and the indexer
/// reads one address. Nothing is reshaped here, unlike the kernels above; the operand names the
/// stream's partition already and it is the `Element` axis that differs.
///
/// One PE at the 64-slice topology, like [`vrf_slice_reshape_64`], so the EDF comparison runs in a
/// default config.
#[device(chip = 1, pe = 1)]
pub fn vrf_group_scale(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![V, N]>,
    scale: &HbmTensor<f32, Chip, m![V, N / 16]>,
) -> HbmTensor<f32, Chip, m![V, N]> {
    let input_dm = input.to_dm::<Cluster1, Slice64, m![N]>(&mut ctx.tdma);
    let scale_dm = scale.to_dm::<Cluster1, Slice64, m![N / 16]>(&mut ctx.tdma);

    let scale_vrf: VrfTensor<f32, Chip, Cluster1, Slice64, m![N / 16]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![N / 128], m![N / 16 % 8]>()
        .collect::<m![N / 128], m![N / 16 % 8]>()
        .to_vrf();

    ctx.main
        .begin(input_dm.view())
        .fetch::<m![N / 8], m![N % 8]>()
        .collect::<m![N / 8], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![N / 4], m![N % 4]>()
        .vector_fp_div(&scale_vrf)
        .vector_widen_concat::<m![N / 8], m![N % 8]>()
        .vector_final()
        .commit_trim::<m![N % 8]>()
        .commit::<m![N]>()
        .to_hbm(&mut ctx.tdma)
}
