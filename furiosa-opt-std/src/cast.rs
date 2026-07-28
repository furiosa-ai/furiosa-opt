use std::ops::RangeInclusive;

use crate::scalar::{bf16, f4e2m1, f8e4m3, f8e5m2, i4, i5, i9};

use super::scalar::Scalar;

/// Trait for types that can be cast during fetch operations.
pub trait FetchCast<D: Scalar>: Into<D> + Cast<D> {}

/// Input scalar types the Fetch Adapter's table-lookup stage can decode to `OutD`.
///
/// The RNGD table-lookup only supports a 4-bit key decoding to an 8-bit value
/// (paired-key table, 4b->8b only), so the only implementor is
/// `f4e2m1: TableLookup<f8e4m3>`. Wider floats and the per-block scale come from a
/// downstream `fetch_cast`. Stating the table at the type level lets
/// `fetch_table_lookup` take no
/// runtime table argument. See the book chapter
/// `computing-tensors/fetch-adapter.md` (the "Table Lookup" section).
pub trait TableLookup<D: Scalar> {
    /// Functional model of the hardware decode table.
    fn lookup(self) -> D;
}

impl TableLookup<f8e4m3> for f4e2m1 {
    fn lookup(self) -> f8e4m3 {
        self.to_f8e4m3()
    }
}

impl TableLookup<bf16> for f8e4m3 {
    /// The non-paired `f8e4m3 -> bf16` baked decode table. `f8e4m3 -> f32` is exact and `f8e4m3` has
    /// only 3 mantissa bits, so the low 16 bits of the `f32` are zero and the `bf16` truncation the
    /// hardware table performs (`build_fp8_to_bf16_lookup_table`) equals `from_f32` exactly.
    fn lookup(self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }
}

// TODO: extend `FetchCast` with the remaining int/float widening and narrowing conversions
// (including the Renegade-S-only variants).

// Identity casts
impl<D> FetchCast<D> for D where D: Scalar {}

impl FetchCast<i32> for i8 {}
impl FetchCast<f32> for bf16 {}
impl FetchCast<f32> for f8e4m3 {}
impl FetchCast<f32> for f8e5m2 {}
impl FetchCast<i32> for i4 {}

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

/// Trait for casting between scalar types.
pub trait Cast<D: Scalar> {
    /// Casts self to target type D.
    fn cast(self) -> D;
}

// `#[inline]` because each `cast` runs once per MAC in the contraction fold; the integer narrow legs
// (`i32 -> i8/i16/u8`) wrap via `as`. See `ContractionCast` for the widen/narrow rule.

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
        self.to_f32()
    }
}

impl Cast<bf16> for f32 {
    #[inline]
    fn cast(self) -> bf16 {
        bf16::from_f32(self)
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
    fn cast(self) -> f32 {
        self.to_f32()
    }
}

impl Cast<f8e5m2> for f32 {
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

impl Cast<i32> for u8 {
    #[inline]
    fn cast(self) -> i32 {
        i32::from(self)
    }
}

impl Cast<u8> for i32 {
    #[inline]
    fn cast(self) -> u8 {
        self as u8
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

/// The contraction output type and the single source of truth for the widen/narrow rule. The Outer
/// stage's Multiplier widens each operand to [`Self::Output`] before multiplying
/// (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`); the host fold (`BufStorage::contraction`)
/// reuses the same types and [`Cast`] conversions, widening on load and narrowing after the fold.
///
/// The `Cast` supertrait and `Output: Cast<Self>` bound make both directions reachable from a single
/// `D: ContractionCast` (widen `D -> Output`, narrow `Output -> D`) and pin that the pair round-trips
/// storage values. They do not constrain the width of `Output`; the per-impl author and the narrow
/// tests own that. Integer narrows wrap (`as`) and `f32 -> bf16` rounds to nearest-even.
pub trait ContractionCast: Scalar + Cast<<Self as ContractionCast>::Output> {
    /// The wider type the contraction accumulates in, and casts back to the storage type to narrow.
    type Output: Scalar + Cast<Self>;
}

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

impl ContractionCast for i16 {
    type Output = i32;
}

impl ContractionCast for u8 {
    type Output = i32;
}

impl ContractionCast for i5 {
    type Output = i32;
}

impl ContractionCast for i9 {
    type Output = i32;
}

// Already at accumulator width: the MAC runs in `i32` / `f32`, so `Output = Self` and the widen/narrow
// casts are the identity `impl Cast<D> for D`.
impl ContractionCast for i32 {
    type Output = i32;
}

impl ContractionCast for f32 {
    type Output = f32;
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
        assert_roundtrip([i16::MIN, -1, 0, 1, i16::MAX]);
        assert_roundtrip([i32::MIN, -1, 0, 1, i32::MAX]);
        assert_roundtrip([0u8, 1, u8::MAX]);
        assert_roundtrip([f32::MIN, -1.5, 0.0, 1.5, f32::MAX]);
        assert_roundtrip([-8, -1, 0, 1, 7].map(i4::from_i32));
        // Narrowing floats: a value built in the storage type is exact in `f32`, so it round-trips.
        assert_roundtrip([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0].map(bf16::from_f32));
        assert_roundtrip([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0].map(f8e4m3::from_f32));
    }

    /// `i32` is already at accumulator width (`Output = Self`, identity widen/narrow): the MAC runs in
    /// `i32`, not `i64`, so a product overflowing `i32` is computed in `i32`. Uses `wrapping_*` to stay
    /// independent of the build's overflow-check setting. The wider integer accumulators cap at this `i32`.
    #[test]
    fn i32_accumulator_stays_in_i32() {
        let (l, r) = (1i32 << 16, 1i32 << 16);
        let prod = narrow::<i32>(widen(l).wrapping_mul(widen(r)));
        assert_eq!(prod, l.wrapping_mul(r));
        assert_ne!(i64::from(prod), i64::from(l) * i64::from(r));
    }

    /// Every integer (`i4`/`i8`/`i16`/`u8`) widens to `i32` and accumulates there, so a product or sum
    /// overflowing the storage range is held exactly mid-reduction; only the final narrow `Cast` (a
    /// wrapping truncation) loses range. An `Output = Self` fold would instead wrap at every step. The
    /// narrow half pins the per-type wrap (`as i16` / `as u8`), including the unsigned `-1 -> 255` the
    /// in-range round-trip never witnesses.
    #[test]
    fn narrow_integer_accumulates_in_i32() {
        // `i4`: -8 * -8 = 64, far past i4's -8..=7. The product is exact in the i32 accumulator.
        let (l, r) = (i4::from_i32(-8), i4::from_i32(-8));
        assert_eq!(widen(l) * widen(r), 64);
        // `i8`: a 256-term dot of 100*100 = 2_560_000, far past i8's -128..=127, held exactly in i32.
        let acc: i32 = std::iter::repeat_n((widen(100i8), widen(100i8)), 256)
            .map(|(l, r)| l * r)
            .sum();
        assert_eq!(acc, 256 * 100 * 100);
        // `i16`: a 256-term dot of 1000*1000 = 256_000_000, far past i16's range, exact in i32; the
        // narrow then wraps `as i16` (256_000_000 mod 2^16).
        let acc16: i32 = std::iter::repeat_n((widen(1000i16), widen(1000i16)), 256)
            .map(|(l, r)| l * r)
            .sum();
        assert_eq!(acc16, 256_000_000);
        assert_eq!(narrow::<i16>(acc16), 256_000_000i32 as i16);
        // `u8`: the i32 accumulator holds the true sum; narrow wraps `as u8`, including the unsigned
        // wrap of a negative accumulator (`-1 -> 255`) that the in-range round-trip law never exercises.
        let accu: i32 = std::iter::repeat_n((widen(200u8), widen(200u8)), 8)
            .map(|(l, r)| l * r)
            .sum();
        assert_eq!(accu, 8 * 200 * 200);
        assert_eq!(narrow::<u8>(accu), (8 * 200 * 200i32) as u8);
        assert_eq!(narrow::<u8>(-1i32), 255u8);
        // The accumulator type is exactly `i32` (not a wider `i64`) for the whole integer family.
        let _: <i8 as ContractionCast>::Output = acc;
        let _: <i4 as ContractionCast>::Output = 0i32;
        let _: <i16 as ContractionCast>::Output = acc16;
        let _: <u8 as ContractionCast>::Output = accu;
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
