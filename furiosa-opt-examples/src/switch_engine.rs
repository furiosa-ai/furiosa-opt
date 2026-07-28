//! Switch Engine examples: each `#[device]` fn lowers a `SwitchConfig::CustomBroadcast`
//! and is checked bit-exact (VISA vs LIR) in `npu-visa-test`.
//!
//! Axis scoping: `Chip`/`Cluster` and the `(B, V)` time/packet axes below are shared by
//! every example. Every other axis is declared in its own `axes!` block directly above
//! the single example that uses it, so each block's domain is exactly that example.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
// Time (`B`) and packet (`V`) axes, common to every example below.
axes![B = 8, V = 16];

// ── custom_broadcast (ring 4) ───────────────────────────────────────────────────────
// The padding slot `1#4` becomes a fresh broadcast tile `Y`; slice volume 64*4 = 256.
axes![A = 64, Y = 4];
type Slice = m![A, 1 # 4];
type OutSlice = m![A, Y];

#[device(chip = 1)]
pub fn custom_broadcast(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, B, V]>,
) -> HbmTensor<bf16, Chip, m![A, Y, B, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![B, V]> = input.to_dm::<Cluster, Slice, m![B, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, OutSlice, m![B, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<OutSlice, m![B]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![B], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, Y, B, V]>(&mut ctx.tdma)
}

// ── custom_broadcast_ring2 ──────────────────────────────────────────────────────────
// Same padding-slot → tile pattern at ring 2; slice volume 128*2 = 256.
axes![A2 = 128, Y2 = 2];

#[device(chip = 1)]
pub fn custom_broadcast_ring2(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A2, B, V]>,
) -> HbmTensor<bf16, Chip, m![A2, Y2, B, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, m![A2, 1 # 2], m![B, V]> =
        input.to_dm::<Cluster, m![A2, 1 # 2], m![B, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![A2, Y2], m![B, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<m![A2, Y2], m![B]>(SwitchConfig::CustomBroadcast { ring_size: 2 })
        .collect::<m![B], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A2, Y2, B, V]>(&mut ctx.tdma)
}

// ── custom_broadcast_ring8 ──────────────────────────────────────────────────────────
// Same padding-slot → tile pattern at ring 8; slice volume 32*8 = 256.
axes![A8 = 32, Y8 = 8];

#[device(chip = 1)]
pub fn custom_broadcast_ring8(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A8, B, V]>,
) -> HbmTensor<bf16, Chip, m![A8, Y8, B, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, m![A8, 1 # 8], m![B, V]> =
        input.to_dm::<Cluster, m![A8, 1 # 8], m![B, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![A8, Y8], m![B, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<m![A8, Y8], m![B]>(SwitchConfig::CustomBroadcast { ring_size: 8 })
        .collect::<m![B], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A8, Y8, B, V]>(&mut ctx.tdma)
}

// ── custom_broadcast_multi (ring 16) ────────────────────────────────────────────────
// Two padding slots become two fresh broadcast tiles `Ym`/`Zm` (cf. the multi-axis
// broadcast sequencing answer key).
axes![Am = 16, Ym = 4, Zm = 4];

#[device(chip = 1)]
pub fn custom_broadcast_multi(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![Am, B, V]>,
) -> HbmTensor<bf16, Chip, m![Am, Ym, Zm, B, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, m![Am, 1 # 4, 1 # 4], m![B, V]> =
        input.to_dm::<Cluster, m![Am, 1 # 4, 1 # 4], m![B, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![Am, Ym, Zm], m![B, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<m![Am, Ym, Zm], m![B]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![B], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![Am, Ym, Zm, B, V]>(&mut ctx.tdma)
}

// ── custom_broadcast_moved_axis (multi-bit bitmap) ──────────────────────────────────
// Unlike the broadcasts above (a padding slot becomes a tile, nothing moves slice to time, so
// every output snoops a single input lane), here a live slice axis `Q` moves to output time and
// a fresh tile `Yt` fills its slot. Each output slice then reads, over the `Q` time steps, the
// whole `Q`-block of input lanes: a multi-bit snoop bitmap. The compare_lir test checks the
// resulting data permutation, while the bitmap bits are pinned by the sequencer-vs-enumeration
// cross-check in `custom_snoop_bitmap` (the LIR executor computes from shapes, not the bitmap).
axes![P = 64, Q = 4, Yt = 4];

#[device(chip = 1)]
pub fn custom_broadcast_moved_axis(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![P, Q, B, V]>,
) -> HbmTensor<bf16, Chip, m![P, Yt, B, Q, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, m![P, Q], m![B, V]> =
        input.to_dm::<Cluster, m![P, Q], m![B, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![P, Yt], m![B, Q, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<m![P, Yt], m![B, Q]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![B, Q], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![P, Yt, B, Q, V]>(&mut ctx.tdma)
}

// ── custom_broadcast_transpose (relocation) ─────────────────────────────────────────
// A pure slice permutation (`[Pt, Qt]` to `[Qt, Pt]`, nothing broadcast or moved to time): each
// output snoops a single input lane, but at a relocated position. The compare_lir test checks the
// permuted data, while the relocated base index in the bitmap is pinned by the internal
// sequencer-vs-enumeration cross-check.
axes![Pt = 32, Qt = 8];

#[device(chip = 1)]
pub fn custom_broadcast_transpose(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![Pt, Qt, B, V]>,
) -> HbmTensor<bf16, Chip, m![Qt, Pt, B, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, m![Pt, Qt], m![B, V]> =
        input.to_dm::<Cluster, m![Pt, Qt], m![B, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![Qt, Pt], m![B, V]> = ctx
        .main
        .begin(dm.view())
        .fetch::<m![B], m![V]>()
        .switch::<m![Qt, Pt], m![B]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![B], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![Qt, Pt, B, V]>(&mut ctx.tdma)
}
