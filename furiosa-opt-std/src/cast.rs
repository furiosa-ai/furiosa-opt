use std::ops::RangeInclusive;

use crate::scalar::{bf16, f4e2m1, f8e4m3, f8e5m2, i4, i5, i9};

use super::scalar::{MaterializableScalar, Scalar};

/// Host-side value conversion between scalar types: what a conversion computes, not a claim that
/// hardware performs it. Some impls match no hardware conversion at all (`i32 -> u8`, `i32 -> i5`),
/// existing only for the contraction fold's round trip. Each converting stage names its own legal
/// subset in a subtrait its primitive binds instead ([`FetchCast`], [`CastEngineCast`],
/// [`CommitCast`]); only `contract_outer` still binds `Cast`, gated by [`ContractionCast`].
pub trait Cast<D: Scalar> {
    /// Casts self to target type D.
    fn cast(self) -> D;
}

/// Element-type pairs the Fetch Adapter's type-casting stage converts, one impl per RNGD
/// conversion. `i4 -> i5` and `i8 -> i9` belong to [`FetchZeroPointSub`] instead. See
/// `computing-tensors/fetch-adapter.md`.
pub trait FetchCast<D: Scalar>: Cast<D> {}

// Identity: the stage is programmed `TypeConversion::None`.
impl<D> FetchCast<D> for D where D: Scalar {}

// Integer widenings to the i32 compute width.
impl FetchCast<i32> for i4 {}
impl FetchCast<i32> for i8 {}
impl FetchCast<i32> for i16 {}
// Float widenings to the f32 compute width.
impl FetchCast<f32> for f8e4m3 {}
impl FetchCast<f32> for f8e5m2 {}
impl FetchCast<f32> for bf16 {}
impl FetchCast<bf16> for f32 {}
// No 8-bit float to `bf16`; use `fetch_table_lookup`. `TypeConversion`'s four Renegade-S variants
// stay out: the trait has no chip-generation axis to hold them.

/// Input scalar types the Fetch Adapter's table-lookup stage can decode to `OutD`.
///
/// One impl per hardware table: a paired 4b->8b decode of `f4e2m1` into either 8-bit float, and a
/// non-paired 8-bit decode of either into `bf16`, which is the only route from an 8-bit float to
/// `bf16` since no [`FetchCast`] does it. Selecting the table by type is what lets
/// `fetch_table_lookup` take no runtime table argument. See `computing-tensors/fetch-adapter.md`.
pub trait TableLookup<D: Scalar> {
    /// Functional model of the hardware decode table.
    fn lookup(self) -> D;
}

impl TableLookup<f8e4m3> for f4e2m1 {
    fn lookup(self) -> f8e4m3 {
        self.to_f8e4m3()
    }
}

impl TableLookup<f8e5m2> for f4e2m1 {
    fn lookup(self) -> f8e5m2 {
        f8e5m2::from_f32(self.to_f32())
    }
}

impl TableLookup<bf16> for f8e5m2 {
    /// Exact for the same reason as the `f8e4m3` decode: 2 mantissa bits leave the `f32`'s low 16
    /// bits zero, so the hardware table's truncation equals `from_f32`.
    fn lookup(self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }
}

impl TableLookup<bf16> for f8e4m3 {
    /// `f8e4m3 -> f32` is exact and `f8e4m3` has only 3 mantissa bits, so the low 16 bits of the
    /// `f32` are zero and the hardware table's `bf16` truncation equals `from_f32` exactly.
    fn lookup(self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }
}

/// Valid zero-point-subtraction widenings for `fetch_zero_point_sub`.
///
/// Subtracting the zero point widens an integer to its contraction-engine staging type, which
/// holds the `(value - zero_point)` range: `i4 -> i5` and `i8 -> i9` (see [`i5`]/[`i9`]
/// for why the extra bit is needed). Only these pairs have an impl, so an invalid
/// widening (e.g. `i8 -> i5`) is a compile error. This is the only way to
/// produce an [`i5`]/[`i9`]; `fetch_cast` cannot.
pub trait FetchZeroPointSub<Out: Scalar>: Scalar {
    /// Valid zero-point range: the source integer type's own range (`i4`:
    /// `[-8, 7]`, `i8`: `[-128, 127]`). A zero point inside it keeps every
    /// `value - zero_point` residual within `Out`, so the caller checks
    /// `zero_point` once (data-independent) instead of each widened element.
    const ZERO_POINT_RANGE: RangeInclusive<i32>;

    /// Subtracts `zero_point` (already range-checked against
    /// [`ZERO_POINT_RANGE`](Self::ZERO_POINT_RANGE)) and widens `self` to `Out`.
    fn zero_point_sub(self, zero_point: i32) -> Out;
}

impl FetchZeroPointSub<i5> for i4 {
    const ZERO_POINT_RANGE: RangeInclusive<i32> = -8..=7;

    fn zero_point_sub(self, zero_point: i32) -> i5 {
        i5::from_i32(i32::from(self) - zero_point)
    }
}

impl FetchZeroPointSub<i9> for i8 {
    const ZERO_POINT_RANGE: RangeInclusive<i32> = -128..=127;

    fn zero_point_sub(self, zero_point: i32) -> i9 {
        i9::from_i32(i32::from(self) - zero_point)
    }
}

/// Output element types the Cast Engine narrows an `i32` / `f32` packet to, one impl per RNGD
/// cast-compaction conversion. No identity impl: a width-preserving cast is not a compaction and
/// the stage rejects it. See `computing-tensors/cast-engine.md`.
#[diagnostic::on_unimplemented(
    message = "the Cast Engine cannot cast `{Self}` to `{D}`",
    label = "not a cast-compaction conversion",
    note = "widenings belong to the Fetch Adapter: `.fetch_cast::<{D}>()`"
)]
pub trait CastEngineCast<D: Scalar>: Cast<D> {}

impl CastEngineCast<i4> for i32 {}
impl CastEngineCast<i8> for i32 {}
impl CastEngineCast<i16> for i32 {}
impl CastEngineCast<f8e4m3> for f32 {}
impl CastEngineCast<f8e5m2> for f32 {}
impl CastEngineCast<bf16> for f32 {}

/// The one cast the Commit Adapter folds in on the way to DM: `f32 -> bf16`
/// (`CommitConversion::CommitF32ToBf16`, ReLU optionally fused). Anything else narrows in
/// [`CastEngineCast`]; a plain `commit()` is already `NoCommitConversion`. See
/// `computing-tensors/commit-adapter.md`.
#[diagnostic::on_unimplemented(
    message = "the Commit Adapter cannot cast `{Self}` to `{D}`",
    label = "commit_cast converts only `f32` to `bf16`",
    note = "narrow to anything else in the Cast Engine: `.cast::<{D}, OutPacket>()`"
)]
pub trait CommitCast<D: Scalar>: Cast<D> {
    /// The fused-ReLU form of the same conversion: negatives clamp to zero on the way through.
    /// ReLU has no standalone hardware stage, so it lives on the cast it fuses into.
    fn cast_relu(self) -> D;
}

impl CommitCast<bf16> for f32 {
    /// NaN is tested before the clamp, so a NaN canonicalizes even with its sign bit set. The clamp
    /// then keys on the sign bit rather than on `< 0.0`, so `-0.0` and every negative subnormal land
    /// on `+0.0` instead of keeping their sign.
    fn cast_relu(self) -> bf16 {
        if !self.is_nan() && self.is_sign_negative() {
            return Cast::cast(0.0);
        }
        Cast::cast(self)
    }
}

// `#[inline]` because each `cast` runs once per MAC in the contraction fold; the integer narrow legs
// (`i32 -> i8/i16`) wrap via `as`. See `ContractionCast` for the widen/narrow rule.

impl<D: Scalar> Cast<D> for D {
    #[inline]
    fn cast(self) -> D {
        self
    }
}

impl Cast<i32> for i8 {
    #[inline]
    fn cast(self) -> i32 {
        self as i32
    }
}

impl Cast<i8> for i32 {
    #[inline]
    fn cast(self) -> i8 {
        self as i8
    }
}

impl Cast<f32> for bf16 {
    #[inline]
    fn cast(self) -> f32 {
        crate::float::bf16_to_f32(self.to_half())
    }
}

impl Cast<bf16> for f32 {
    #[inline]
    fn cast(self) -> bf16 {
        bf16::from_half(crate::float::f32_to_bf16(self))
    }
}

impl Cast<f32> for f8e4m3 {
    #[inline]
    fn cast(self) -> f32 {
        self.to_f32()
    }
}

impl Cast<f8e4m3> for f32 {
    #[inline]
    fn cast(self) -> f8e4m3 {
        f8e4m3::from_f32(self)
    }
}

impl Cast<f32> for f8e5m2 {
    #[inline]
    fn cast(self) -> f32 {
        self.to_f32()
    }
}

impl Cast<f8e5m2> for f32 {
    #[inline]
    fn cast(self) -> f8e5m2 {
        f8e5m2::from_f32(self)
    }
}

impl Cast<i32> for i4 {
    #[inline]
    fn cast(self) -> i32 {
        self.to_i32()
    }
}

impl Cast<i4> for i32 {
    #[inline]
    fn cast(self) -> i4 {
        i4::from_i32(self)
    }
}

impl Cast<i32> for i16 {
    #[inline]
    fn cast(self) -> i32 {
        i32::from(self)
    }
}

impl Cast<i16> for i32 {
    #[inline]
    fn cast(self) -> i16 {
        self as i16
    }
}

// i5/i9 are contraction stream stagings produced only by `fetch_zero_point_sub`. They widen to the
// i32 accumulator like the other integers; the narrow direction exists only to satisfy
// `ContractionCast`'s round-trip bound and is never taken (a contraction result is never stored as i5/i9).
impl Cast<i32> for i5 {
    #[inline]
    fn cast(self) -> i32 {
        self.to_i32()
    }
}

impl Cast<i5> for i32 {
    #[inline]
    fn cast(self) -> i5 {
        i5::from_i32(self)
    }
}

impl Cast<i32> for i9 {
    #[inline]
    fn cast(self) -> i32 {
        self.to_i32()
    }
}

impl Cast<i9> for i32 {
    #[inline]
    fn cast(self) -> i9 {
        i9::from_i32(self)
    }
}

/// The element types the contraction engine multiplies, each with the accumulator it widens to
/// (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`): `i4` and `i8`, each also in the [`i5`]/[`i9`] staging
/// zero-point subtraction produces, plus `f8e4m3`, `f8e5m2` and `bf16`. Nothing else, so a contraction
/// on `i16`, `i32` or `f32` does not compile:
///
/// ```compile_fail
/// use furiosa_opt_std::prelude::*;
/// fn contraction_stream<D: ContractionCast>() {}
/// contraction_stream::<f32>();
/// ```
///
/// `i32` and `f32` are what a contraction accumulates *to*, never what it reads;
/// [`ContractionAccumulator`] names that side. The [`ContractionWeight`] supertrait ties this set to the
/// weight table. Narrows wrap (`as`); `f32 -> bf16` rounds to nearest-even.
pub trait ContractionCast: Scalar + ContractionWeight<Self> + Cast<<Self as ContractionCast>::Output> {
    /// The wider type the contraction accumulates in, and casts back to the storage type to narrow.
    type Output: ContractionAccumulator + Cast<Self>;
}

/// The widths a contraction accumulates in: `i32` for the integer operand families, `f32` for the float
/// ones. Every [`ContractionCast::Output`] is one of these.
///
/// A fold binds this where the Multiplier binds [`ContractionCast`], so neither type has to be both.
/// Accumulating at a storage width would sum an `i8` matmul in `i8`:
///
/// ```compile_fail
/// use furiosa_opt_std::prelude::*;
/// fn contraction_fold<Acc: ContractionAccumulator>() {}
/// contraction_fold::<bf16>();
/// ```
#[diagnostic::on_unimplemented(
    message = "a contraction cannot accumulate in `{Self}`",
    label = "not a contraction accumulator width",
    note = "a contraction accumulates in `i32` (integer operands) or `f32` (float operands)"
)]
pub trait ContractionAccumulator: MaterializableScalar {}

impl ContractionAccumulator for i32 {}
impl ContractionAccumulator for f32 {}

/// Weight (TRF) element types that can be contracted against a given stream
/// (activation) element type `Stream`.
///
/// This relaxes `contract_outer`'s operand-type constraint from "the weight
/// equals the stream type" to "the weight forms a valid contraction-engine operand pair with
/// the stream". Floats pair only with the same type. The integer family pairs
/// within a precision ({i4, i5} with {i4, i5}, and {i8, i9} with {i8, i9}), so
/// either operand may be the raw form (i4/i8) or its zero-point-subtracted
/// staging (i5/i9). Cross-precision (e.g. i4 against i8) and cross-kind (e.g.
/// bf16 against i8) pairs have no impl and are a compile error.
pub trait ContractionWeight<Stream: Scalar>: Scalar {}

// Integer family {i4, i5}: either operand may be raw (i4) or zero-point-subtracted (i5).
impl ContractionWeight<i4> for i4 {}
impl ContractionWeight<i5> for i4 {}
impl ContractionWeight<i4> for i5 {}
impl ContractionWeight<i5> for i5 {}
// Integer family {i8, i9}: either operand may be raw (i8) or zero-point-subtracted (i9).
impl ContractionWeight<i8> for i8 {}
impl ContractionWeight<i9> for i8 {}
impl ContractionWeight<i8> for i9 {}
impl ContractionWeight<i9> for i9 {}
// Floats pair only with the same type.
impl ContractionWeight<bf16> for bf16 {}
impl ContractionWeight<f8e4m3> for f8e4m3 {}
impl ContractionWeight<f8e5m2> for f8e5m2 {}

// The contraction operand types: integers accumulate in `i32`, narrowing floats in `f32`.

impl ContractionCast for i8 {
    type Output = i32;
}

impl ContractionCast for bf16 {
    type Output = f32;
}

impl ContractionCast for f8e4m3 {
    type Output = f32;
}

impl ContractionCast for f8e5m2 {
    type Output = f32;
}

impl ContractionCast for i4 {
    type Output = i32;
}

impl ContractionCast for i5 {
    type Output = i32;
}

impl ContractionCast for i9 {
    type Output = i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Widens a storage cell to its accumulator type via [`Cast`] (`S -> Output`), as the folds do on load.
    fn widen<S: ContractionCast>(x: S) -> <S as ContractionCast>::Output {
        Cast::cast(x)
    }

    /// Narrows an accumulated result back to storage via [`Cast`] (`Output -> S`), as the folds do at the end.
    fn narrow<S: ContractionCast>(acc: <S as ContractionCast>::Output) -> S {
        Cast::cast(acc)
    }

    /// `narrow(widen(x)) == x` for any `x` in the storage type `S` (a round-trip on storable values,
    /// not identity on arbitrary accumulator values): a value from `S` is exact in its own accumulator.
    /// Pins the widen/narrow `Cast` pair for the chosen `Output`, not the choice of `Output`.
    fn assert_roundtrip<S: ContractionCast + std::fmt::Debug>(samples: impl IntoIterator<Item = S>) {
        for x in samples {
            assert_eq!(narrow::<S>(widen(x)), x, "narrow ∘ widen must round-trip {x:?}");
        }
    }

    #[test]
    fn narrow_widen_round_trips() {
        assert_roundtrip([i8::MIN, -1, 0, 1, i8::MAX]);
        assert_roundtrip([-8, -1, 0, 1, 7].map(i4::from_i32));
        // Narrowing floats: a value built in the storage type is exact in `f32`, so it round-trips.
        assert_roundtrip([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0].map(bf16::from_f32));
        assert_roundtrip([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0].map(f8e4m3::from_f32));
        assert_roundtrip([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0].map(f8e5m2::from_f32));
    }

    /// The whole operand set, as compile-time bounds. The negative half is the `compile_fail` doctest
    /// on [`ContractionCast`].
    #[test]
    fn contraction_operands_are_the_engine_input_types() {
        fn assert_operand<D: ContractionCast>() {}
        assert_operand::<i4>();
        assert_operand::<i5>();
        assert_operand::<i8>();
        assert_operand::<i9>();
        assert_operand::<f8e4m3>();
        assert_operand::<f8e5m2>();
        assert_operand::<bf16>();
    }

    /// The integer accumulator is exactly `i32`, never a wider `i64`: a product past `i32` wraps rather
    /// than being held. `wrapping_*` keeps this independent of the build's overflow checks.
    #[test]
    fn integer_accumulator_stays_in_i32() {
        type Acc = <i8 as ContractionCast>::Output;
        let (l, r): (Acc, Acc) = (1 << 16, 1 << 16);
        let prod = l.wrapping_mul(r);
        assert_eq!(prod, 0, "2^32 wraps to 0 in i32");
        assert_ne!(i64::from(prod), i64::from(l) * i64::from(r));
    }

    /// Every integer operand widens to `i32`, so a sum past the storage range survives to the final
    /// narrow, which is the only step that loses it.
    #[test]
    fn narrow_integer_accumulates_in_i32() {
        // `i4`: -8 * -8 = 64, far past i4's -8..=7. The product is exact in the i32 accumulator.
        let (l, r) = (i4::from_i32(-8), i4::from_i32(-8));
        assert_eq!(widen(l) * widen(r), 64);
        // `i8`: 256 * 100 * 100 = 2_560_000, far past i8's -128..=127; the narrow wraps it `as i8`.
        let acc: i32 = std::iter::repeat_n((widen(100i8), widen(100i8)), 256)
            .map(|(l, r)| l * r)
            .sum();
        assert_eq!(acc, 256 * 100 * 100);
        assert_eq!(narrow::<i8>(acc), 2_560_000i32 as i8);
        // The accumulator type is exactly `i32` (not a wider `i64`) for the whole integer family.
        let _: <i8 as ContractionCast>::Output = acc;
        let _: <i4 as ContractionCast>::Output = 0i32;
        let _: <i5 as ContractionCast>::Output = 0i32;
        let _: <i9 as ContractionCast>::Output = 0i32;
    }

    /// Pins the `f32 -> bf16` narrow `Cast` as round-to-nearest-even (RNE), the rounding the wide `f32`
    /// accumulator narrows through (it delegates to `half::bf16::from_f32`). `1.0 + 2^-8` sits exactly
    /// halfway between `1.0` and the next bf16 step `1 + 2^-7`; RNE breaks the tie to the even mantissa
    /// (`1.0`). A nearby off-tie value rounds up.
    #[test]
    fn bf16_narrow_is_round_to_nearest_even() {
        let one_ulp = f32::exp2(-7.0); // bf16 step near 1.0 (7 stored mantissa bits)
        // Exact half-way between 1.0 (even mantissa) and 1 + ulp (odd): RNE ties DOWN to even 1.0.
        assert_eq!(narrow::<bf16>(1.0 + one_ulp / 2.0).to_f32(), 1.0);
        // Exact half-way between 1 + ulp (odd) and 1 + 2*ulp (even): RNE ties UP to even 1 + 2*ulp.
        assert_eq!(narrow::<bf16>(1.0 + one_ulp * 1.5).to_f32(), 1.0 + 2.0 * one_ulp);
        // Just past half-way: rounds up to 1 + ulp regardless of tie direction.
        assert_eq!(narrow::<bf16>(1.0 + one_ulp * 0.75).to_f32(), 1.0 + one_ulp);
    }
}
