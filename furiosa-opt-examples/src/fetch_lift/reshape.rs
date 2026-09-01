//! Axis lifts require real replication even when `reshape` names a placement slot as a broadcast.
//!
//! The first two kernels preserve replication. The last kernel shows the CPU and hardware mismatch
//! caused by relabelling live data as a broadcast.

use furiosa_opt_std::prelude::*;

use super::{A, Chip, Cluster, G, H, Slice, V};

axes![HV = 32, Rep = 2];

/// Splits a replicated `HV` element into the `H` dimension lifted onto `Slice`.
#[device(chip = 1)]
pub fn fetch_slice_lift_reshaped_broadcast(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, HV]>,
) -> HbmTensor<bf16, Chip, m![A, H, V]> {
    let flat: DmTensor<bf16, Chip, Cluster, Slice, m![HV]> = input.to_dm::<Cluster, Slice, m![HV]>(&mut ctx.tdma);

    // SAFETY: `[HV]` and `[H, V]` enumerate the same elements in the same wire order.
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, V]> = unsafe { flat.reshape() };

    let result: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, H, V]>(&mut ctx.tdma)
}

/// Relabels a replicated `Rep` dimension as the broadcast filled by the lift.
#[device(chip = 1)]
pub fn fetch_slice_lift_reshaped_named_broadcast(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, H, V]>,
) -> HbmTensor<bf16, Chip, m![A, H, V]> {
    let replicated: DmTensor<bf16, Chip, Cluster, m![A, Rep], m![H, V]> =
        input.to_dm::<Cluster, m![A, Rep], m![H, V]>(&mut ctx.tdma);

    // SAFETY: `[A, Rep]` and `[A, 2]` enumerate the same slices in the same wire order.
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, V]> = unsafe { replicated.reshape() };

    let result: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, H, V]>(&mut ctx.tdma)
}

/// Demonstrates the backend mismatch caused by relabelling live `G` data as a broadcast.
#[device(chip = 1)]
pub fn fetch_slice_lift_reshaped_without_broadcast(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, G, H, V]>,
) -> HbmTensor<bf16, Chip, m![A, H, V]> {
    let live: DmTensor<bf16, Chip, Cluster, m![A, G], m![H, V]> =
        input.to_dm::<Cluster, m![A, G], m![H, V]>(&mut ctx.tdma);

    // SAFETY: `[A, G]` and `[A, 2]` enumerate the same slices in the same wire order. The reshape is
    // legal, but the following lift is invalid because those slices do not contain replicated data.
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, V]> = unsafe { live.reshape() };

    let result: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, H, V]>(&mut ctx.tdma)
}
