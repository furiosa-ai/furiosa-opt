//! `f4e2m1` (NVFP4 / MXFP4) weight GEMM via decode-then-stage.
//!
//! Table lookup is a main-context fetch stage, so a packed `f4e2m1` weight is decoded to `f8e4m3` in
//! a main-context fetch (`fetch -> fetch_table_lookup::<f8e4m3>() -> collect -> commit`) that writes
//! the decoded weight to DM, and a plain convert-only StoTrf then stages that `f8e4m3` weight into
//! the contraction TRF. Decoding cannot happen inside the StoTrf fetch itself: the sub-context Fetch
//! Unit has no lookup-table register, so table lookup is offered only on the main context (see
//! `computing-tensors/fetch-adapter.md`).
//!
//! The kernel is a tokens-on-slices GEMM: an `f8e4m3` activation streams the contraction while the
//! decoded `f8e4m3` weight is the weight-stationary filter, the f8 MAC (`Red % 64`) reduces, and the
//! VE tail applies the per-output-channel dequant scale and rounds to `bf16`. Only the packed weight
//! read plus its decode pass (¼ the weight DMA volume) differs from a native f8 weight GEMM.

use furiosa_opt_std::prelude::*;

axes![Tok = 16, Red = 64, Out = 16];

type Chip = m![1];
// Tokens on the slices: `Tok / 8 % 2` cluster, `Tok % 8` slice padded to 256.
type GemmCl = m![Tok / 8 % 2];
type TokSlice = m![Tok % 8 # 256];

/// `[Tok, Red] x [Out, Red]^T -> [Tok, Out]` tokens-on-slices GEMM with the `[Out, Red]` weight
/// PACKED `f4e2m1`, decoded to `f8e4m3` by a main-context table-lookup fetch that commits to DM,
/// then staged into the TRF by a plain convert-only StoTrf and contracted against the `f8e4m3`
/// `[Tok, Red]` activation. Each output is `scale[o] * sum_red act[tok, red] * decode(weight[o, red])`,
/// accumulated in `f32` and rounded to `bf16`.
#[device(chip = 1)]
pub fn stotrf_table_lookup_gemm(
    ctx: &mut Context,
    act: &HbmTensor<f8e4m3, Chip, m![Tok, Red]>,
    weight: &HbmTensor<f4e2m1, Chip, m![Out, Red]>,
    w_scale: &HbmTensor<bf16, Chip, m![Out]>,
) -> HbmTensor<bf16, Chip, m![Tok, Out]> {
    const {
        assert!(
            Red::SIZE % 64 == 0,
            "Red must be a multiple of the 64-element f8 MAC width"
        );
        assert!(
            Out::SIZE % 8 == 0,
            "output width must be a multiple of the 8-lane output group"
        );
        assert!(Tok::SIZE <= 16, "Tok must map onto 2 clusters x 8 slices (<= 16)");
    }

    // Stream operand: the e4m3 activation, tokens on the slices, `Red` in the element.
    let act_dm: DmTensor<f8e4m3, Chip, GemmCl, TokSlice, m![Red]> = act.to_dm(&mut ctx.tdma);

    // Resident token-major output DM (tokens on slices, `Out` contiguous in the element).
    const OUT_DM_ADDR: Address = 0x0040_0000;
    let mut out_dm: DmTensor<bf16, Chip, GemmCl, TokSlice, m![Out]> = DmTensor::new();

    // The full `[Out]` per-output-channel dequant scale, broadcast across the token-slices.
    let scale_all_dm: DmTensor<bf16, Chip, GemmCl, TokSlice, m![Out]> = w_scale.to_dm(&mut ctx.tdma);

    for g in 0..(Out::SIZE / 8) {
        // This group's 8 packed f4e2m1 weight rows, broadcast across the token-slices.
        let weight_group = weight
            .view()
            .tile::<m![Out / 8], 1, m![1 # { Out::SIZE / 8 }, Out % 8, Red]>(g);
        let weight_dm: DmTensor<f4e2m1, Chip, GemmCl, TokSlice, m![Out % 8, Red]> = weight_group.to_dm(&mut ctx.tdma);

        // Table lookup is a main-context fetch stage, so the packed f4e2m1 weight is decoded to
        // f8e4m3 in a main-context fetch that commits the decoded stream to a resident DM. The fetch
        // reads the packed f4 (2 codes/byte) and `fetch_table_lookup::<f8e4m3>()` decodes each 4-bit
        // code to its e4m3 value. The decoded weight lands in a resident DM (fixed address, not a
        // transient `commit()` scratch) so the sub-context StoTrf below can read it back.
        const WEIGHT_F8_DM_ADDR: Address = 0x0060_0000;
        let mut weight_f8_dm: DmTensor<f8e4m3, Chip, GemmCl, TokSlice, m![Out % 8, Red]> = DmTensor::new();
        ctx.main
            .begin(weight_dm.view())
            .fetch::<m![Out % 8], m![Red]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![Out % 8, Red / 32], m![Red % 32]>()
            .commit_trim::<m![Red % 32]>()
            .commit_view(weight_f8_dm.view_mut());

        // Stage the decoded f8e4m3 weight into the TRF as the weight-stationary filter (a plain
        // convert-only StoTrf, `f8e4m3 -> f8e4m3`, no table lookup on the sub-context fetch).
        let weight_trf: TrfTensor<f8e4m3, Chip, GemmCl, TokSlice, m![Out % 8], m![Red]> = ctx
            .sub
            .begin(weight_f8_dm.view())
            .fetch::<m![Out % 8], m![Red]>()
            .collect::<m![Out % 8, Red / 32], m![Red % 32]>()
            .to_trf();

        // The group's 8 per-output-channel dequant scales, broadcast across the token-slices.
        let scale_group = scale_all_dm
            .view()
            .tile::<m![Out / 8], 1, m![1 # { Out::SIZE / 8 }, Out % 8]>(g);
        let scale_vrf: VrfTensor<f32, Chip, GemmCl, TokSlice, m![Out % 8]> = ctx
            .sub
            .begin(scale_group)
            .fetch::<m![1], m![Out % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![1], m![Out % 8]>()
            .to_vrf();

        // Contract the streamed per-token e4m3 activation against the decoded 8-row f8 weight TRF
        // (`mac_width(f8) = 64`, so the contraction OutPacket is `Red % 64`), fold the 8 output
        // features Interleaved into the `OutPacket`, then dequant in the VE tail and round to bf16.
        ctx.main
            .begin(act_dm.view())
            .fetch::<m![Red / 32], m![Red % 32]>()
            .collect::<m![Red / 32], m![Red % 32]>()
            .contract_outer::<m![Red / 64], m![Red % 64], _, _, _>(&weight_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![1]>()
            .contract_lane::<m![1], m![Out % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Out / 4 % 2], m![Out % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_concat::<m![1], m![Out % 8]>()
            .vector_final()
            .cast::<bf16, m![Out % 8 # 16]>()
            .commit_trim::<m![Out % 8]>()
            .commit_view(
                out_dm
                    .view_mut()
                    .tile::<m![Out / 8], 1, m![1 #{!} { Out::SIZE / 8 }, Out % 8]>(g),
            );
    }

    let mut out = unsafe { HbmTensor::<bf16, Chip, m![Tok, Out]>::from_addr(0x0080_0000) };
    out_dm.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
    out
}
