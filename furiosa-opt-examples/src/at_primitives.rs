//! Codegen coverage for the address-taking `*_at` primitives.
//!
//! This PR strips explicit addresses from the on-chip transfers, which removes the only
//! callers of the `*_at` variants (`HbmTensor[View]::to_dm_at`, `DmTensor::to_dm_at`,
//! `dma_gather_scaled_at`, `to_vrf_at`, `to_trf_at`, `commit_at`, `to_dm_pcopy_at`).
//! These kernels keep each one exercised so the primitive still compiles and codegens.
//! They mirror the address-free examples, swapping the on-chip call for its `*_at` form
//! with an explicit address. (`to_hbm` is not split: an HBM tensor always takes an
//! address, so every `to_hbm` call already exercises that path.)

#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

type Chip = m![1];

/// `HbmTensor::to_dm_at` and `commit_at` (mirrors `fetch_commit_simple`).
pub mod dma_commit {
    use super::*;

    type Cluster = m![1 # 2];
    axes![A = 4096, B = 8];

    #[device(chip = 1)]
    pub fn fetch_commit_at(
        ctx: &mut Context,
        input: &HbmTensor<i8, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![B, A]> {
        let input_dm = input.to_dm_at::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma, 0);

        let fetch_and_commit_tensor: DmTensor<i32, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A / 8 % 2], m![A % 8, B]>()
            .fetch_cast::<i32>()
            .collect::<m![A / 8 % 2, A % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit_at(0);

        fetch_and_commit_tensor.to_hbm(&mut ctx.tdma)
    }
}

/// `HbmTensorView::to_dm_at` and `to_vrf_at` (mirrors `ve_elementwise_multi_vrf`, which passes on cpu).
pub mod vrf {
    use super::*;

    type Cluster = m![1 # 2];
    axes![A = 512, B = 256];

    #[device(chip = 1)]
    pub fn multi_vrf_at(
        ctx: &mut Context,
        input: &HbmTensor<i32, Chip, m![A, B]>,
        vrf_data1: &HbmTensor<i32, Chip, m![B]>,
        vrf_data2: &HbmTensor<i32, Chip, m![B]>,
    ) -> HbmTensor<i32, Chip, m![A, B]> {
        let input_intermediate: HbmTensor<i32, Chip, m![B, A]> = input.to_hbm(&mut ctx.tdma);
        let input_dm = input_intermediate.to_dm::<Cluster, m![A / 2], m![B, A % 2]>(&mut ctx.tdma);
        let vrf_dm1 = vrf_data1.view().to_dm_at::<Cluster, m![A / 2], m![B]>(&mut ctx.tdma, 0);
        let vrf_dm2 = vrf_data2.to_dm::<Cluster, m![A / 2], m![B]>(&mut ctx.tdma);

        let vrf1: VrfTensor<i32, Chip, Cluster, m![A / 2], m![B]> = ctx
            .sub
            .begin(vrf_dm1.view())
            .fetch::<m![1], m![B]>()
            .fetch_cast::<i32>()
            .collect::<m![B / 8], m![B % 8]>()
            .to_vrf_at(0);

        let vrf2: VrfTensor<i32, Chip, Cluster, m![A / 2], m![B]> = ctx
            .sub
            .begin(vrf_dm2.view())
            .fetch::<m![1], m![B]>()
            .fetch_cast::<i32>()
            .collect::<m![B / 8], m![B % 8]>()
            .to_vrf();

        let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![B, A % 2]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![B], m![A % 2]>()
            .fetch_cast::<i32>()
            .collect::<m![B], m![A % 2 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &vrf1)
            .vector_fxp(FxpBinaryOp::MulInt, &vrf2)
            .vector_clip(ClipBinaryOpI32::AddFxp, &vrf1)
            .vector_final()
            .commit_trim::<m![A % 2]>()
            .commit();

        result.to_hbm(&mut ctx.tdma)
    }
}

/// `to_trf_at` (mirrors `contract_outer_assertions::trf_size::valid_to_trf_full`).
pub mod trf {
    use super::*;

    type Cluster = m![1 # 2];
    type Slice = m![1 # 256];
    axes![A = 8, B = 64];

    #[device(chip = 1)]
    pub fn to_trf_at(
        ctx: &mut Context,
        input: &HbmTensor<i8, Chip, m![A, B]>,
        _output: &mut HbmTensor<i8, Chip, m![A, B]>,
    ) {
        let trf_dm = input.to_dm_at::<Cluster, Slice, m![A, B]>(&mut ctx.tdma, 0);

        let _trf: TrfTensor<i8, Chip, Cluster, Slice, m![A], m![B]> = ctx
            .sub
            .begin(trf_dm.view())
            .fetch::<m![A], m![B]>()
            .fetch_cast::<i8>()
            .collect::<m![A, B / 32], m![B % 32]>()
            .to_trf_at(TrfAddress::Full);
    }
}

/// `DmTensor::to_dm_pcopy_at` (mirrors `memory_op::dm_pcopy`, which uses the address-free form).
///
/// `to_dm_pcopy` reaches MIR only (no VISA/LIR translation yet), so this twin is exercised at the
/// emulation level in `at_primitives_tests`, not in the VISA-vs-LIR suite.
pub mod pcopy {
    use super::*;

    axes![A = 256, B = 4096];

    #[device(chip = 1)]
    pub fn dm_pcopy_at(ctx: &mut Context, hbm: &HbmTensor<i32, m![1], m![A, B]>) -> HbmTensor<i32, m![1], m![A, B]> {
        let src: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
        let dst: DmTensor<i32, m![1], m![1], m![A], m![B]> = src.to_dm_pcopy_at(&mut ctx.sub, 0x10_0000);
        dst.to_hbm(&mut ctx.tdma)
    }
}

/// `DmTensor::to_dm_at` (mirrors `memory_op::dm_relayout`). The DM → DM `to_dm` reaches MIR only,
/// so this twin is exercised at the emulation level in `at_primitives_tests`.
pub mod relayout {
    use super::*;

    axes![A = 256, B = 4096];

    #[device(chip = 1)]
    pub fn dm_relayout_at(
        ctx: &mut Context,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A / 2, A % 2, B]> {
        let dm: DmTensor<i32, m![1], m![1 # 2], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
        let relaid: DmTensor<i32, m![1], m![1 # 2], m![1 # 2, A / 2], m![A % 2, B]> =
            dm.to_dm_at(&mut ctx.tdma, 0x10_0000);
        relaid.to_hbm(&mut ctx.tdma)
    }
}

/// `HbmTensor::dma_gather_scaled_at` (mirrors `scatter_gather::gather_minimal`, which uses the address-free
/// form). Like that example it reaches the VISA stage only, so this is codegen coverage.
pub mod gather {
    use super::*;

    type Cluster = m![1];
    axes![K = 512, D = 128, C = 612];

    // Runs at a 2-PE device (1 cluster x 128 slices), which matches the gather output's
    // `Cluster = m![1]` x slice `m![D]` (D = 128).
    #[device(chip = 1, pe = 2)]
    pub fn gather_at(
        ctx: &mut Context,
        table: &HbmTensor<bf16, Chip, m![K, D]>,
        index: &HbmTensor<i32, Chip, m![C]>,
    ) -> HbmTensor<bf16, Chip, m![C, D]> {
        let values_dm: DmTensor<bf16, Chip, Cluster, m![D], m![C]> = table.dma_gather_scaled_at(index, 0x0);
        values_dm.to_hbm(&mut ctx.tdma)
    }
}
