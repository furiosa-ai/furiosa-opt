//! Device-function scalar-type coverage: one fixture per LIR-supported device scalar.
//!
//! Every scalar that `furiosa-opt-std` implements `Scalar` for and that has a matching LIR
//! `ElementType` gets exactly one fixture here, so the per-type roster is visible at a glance and
//! each type's mir -> vISA -> IR translation (`translate_ty_to_scalar` / `eval_scalar_type`) is
//! pinned by a sibling test in `npu-visa-test/tests/scalar.rs`. Each fixture names its scalar in a
//! tensor parameter, a DM staging, and the fetch cast, so it must translate through those two
//! functions rather than be rejected as an "unsupported scalar primitive".
//!
//! Fixture shape per type:
//! - **at-width identity relayout** for `f32` / `i32` / `i16` / `i8` / `u8`: the element threads
//!   through the parameter, the DM staging, the identity fetch cast, and the transposed output.
//! - **widen** for the narrowing floats: `bf16 -> f32`, `f8e4m3 -> f32`.
//!
//! `i4` is a supported LIR scalar (`eval_scalar_type` maps it), but has no fixture here: it is
//! sub-byte (`RawInt4` packs two per byte), so a host-buffer execute-compare is infeasible and the
//! `--device-function` build probe is the only available check. Omitted rather than carry a
//! build-only outlier.
//!
//! The collect stage requires a packet of exactly one 32-byte flit in the post-cast element type,
//! so each module sizes its innermost axis `B` to make `B * size_of::<dst>() == 32` (8 x 4B,
//! 16 x 2B, or 32 x 1B). Each fixture lives in its own module (for a per-type `axes!`) but carries
//! a unique leaf fn name, because the `--device-function` filter matches any path ending in
//! `::<name>`.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];

/// Emits one `src -> dst` fetch-cast fixture in its own module: the `src` element threads through
/// the parameter and DM staging, the fetch cast maps `src -> dst`, and the `dst` result is stored
/// transposed. Passing the same type for `$src` and `$dst` is an at-width identity relayout, a
/// narrower `$src` a widen, a wider `$src` a narrowing. The innermost `B` is both the DMA tail
/// (>= min_align = 8 bytes) and one collect flit (= 32 bytes in `$dst`), so the caller picks `$b`
/// to satisfy `$b * size_of::<$dst>() == 32`.
macro_rules! scalar_fixture {
    ($module:ident, $fn:ident, $src:ty, $dst:ty, $b:literal) => {
        pub mod $module {
            use super::*;

            axes![A = 4096, B = $b];

            #[device(chip = 1)]
            pub fn $fn(
                ctx: &mut Context,
                input: &HbmTensor<$src, m![1], m![A, B]>,
            ) -> HbmTensor<$dst, m![1], m![B, A]> {
                let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma);

                let result: DmTensor<$dst, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = ctx
                    .main
                    .begin(input_dm.view())
                    .fetch::<m![A / 8 % 2], m![A % 8, B]>()
                    .fetch_cast::<$dst>()
                    .collect::<m![A / 8 % 2, A % 8], m![B]>()
                    .commit_trim::<m![B]>()
                    .commit();

                result.to_hbm(&mut ctx.tdma)
            }
        }
    };
}

// At-width identity relayout. f32 / i32: 8 x 4 bytes = 32-byte flit. i16: 16 x 2 bytes. i8 / u8: 32 x 1 byte.
scalar_fixture!(f32_relayout, relayout_f32, f32, f32, 8);
scalar_fixture!(i32_relayout, relayout_i32, i32, i32, 8);
scalar_fixture!(i16_relayout, relayout_i16, i16, i16, 16);
scalar_fixture!(i8_relayout, relayout_i8, i8, i8, 32);
scalar_fixture!(u8_relayout, relayout_u8, u8, u8, 32);

// Widen to a 4-byte target: 8 x 4 bytes = 32-byte flit (the narrow source's 8-element tail is
// >= 8 bytes).
scalar_fixture!(bf16_widen, widen_bf16, bf16, f32, 8);
scalar_fixture!(f8e4m3_widen, widen_f8e4m3, f8e4m3, f32, 8);
scalar_fixture!(f8e5m2_widen, widen_f8e5m2, f8e5m2, f32, 8);
scalar_fixture!(i16_widen, widen_i16, i16, i32, 8);

// The one narrowing fetch conversion; 16 x 2-byte `bf16` is one flit.
scalar_fixture!(f32_narrow, narrow_f32, f32, bf16, 16);

/// The same `f32 -> bf16` narrowing folded into the commit path instead of the fetch, so the Cast
/// Engine stays free for sub-context work. `commit_trim` runs first and its packet is the commit
/// unit's input width, which a converting commit needs to be a multiple of 16 B: 8 `f32` is 32 B.
pub mod f32_commit_narrow {
    use super::*;

    axes![A = 4096, B = 8];

    #[device(chip = 1)]
    pub fn commit_narrow_f32(
        ctx: &mut Context,
        input: &HbmTensor<f32, m![1], m![A, B]>,
    ) -> HbmTensor<bf16, m![1], m![B, A]> {
        let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma);

        let result: DmTensor<bf16, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A / 8 % 2], m![A % 8, B]>()
            .fetch_cast::<f32>()
            .collect::<m![A / 8 % 2, A % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit_cast::<bf16>()
            .commit();

        result.to_hbm(&mut ctx.tdma)
    }

    /// The ReLU-fused conversion, a separate hardware conversion rather than a mode of the one
    /// above.
    #[device(chip = 1)]
    pub fn commit_narrow_relu_f32(
        ctx: &mut Context,
        input: &HbmTensor<f32, m![1], m![A, B]>,
    ) -> HbmTensor<bf16, m![1], m![B, A]> {
        let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma);

        let result: DmTensor<bf16, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = ctx
            .main
            .begin(input_dm.view())
            .fetch::<m![A / 8 % 2], m![A % 8, B]>()
            .fetch_cast::<f32>()
            .collect::<m![A / 8 % 2, A % 8], m![B]>()
            .commit_trim::<m![B]>()
            .commit_cast_relu::<bf16>()
            .commit();

        result.to_hbm(&mut ctx.tdma)
    }
}

// `i4 -> i32` has no fixture: `i4` is sub-byte, so a host-buffer execute-compare is infeasible (see
// the module doc). `i4` translation is still covered by the `eval_scalar_type` match arm.
