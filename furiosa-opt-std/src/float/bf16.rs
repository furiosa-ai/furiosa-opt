//! The `bf16` legs of the RNGD float conversions.
//!
//! Plain IEEE conversion differs on two input classes. A subnormal flushes to a zero of its own
//! sign, since RNGD supports no subnormal in `f32` or `bf16`. A NaN collapses to one quiet word,
//! which keeps the result bit-predictable; that word is a convention, not a payload to depend on.

use half::bf16;

/// The quiet NaNs the conversions produce, spelled in bits rather than as `f32::NAN` / `bf16::NAN`:
/// the conversion has to reproduce these exact words, and neither constant is pinned by the language
/// (`f32::NAN`) or by a dependency we control (`half`).
const QUIET_NAN_F32: u32 = 0x7fc0_0000;
const QUIET_NAN_BF16: u16 = 0x7fc0;

/// Widens `bf16` to `f32` as the Fetch Adapter does.
pub(crate) fn bf16_to_f32(v: bf16) -> f32 {
    if v.is_nan() {
        f32::from_bits(QUIET_NAN_F32)
    } else if v.is_finite() && !v.is_normal() {
        // Subnormal, or a zero the branch also catches and maps to itself.
        if v.is_sign_positive() { 0.0 } else { -0.0 }
    } else {
        v.to_f32()
    }
}

/// Narrows `f32` to `bf16` as the Fetch Adapter, Cast Engine and Commit Adapter all do.
pub(crate) fn f32_to_bf16(v: f32) -> bf16 {
    if v.is_nan() {
        bf16::from_bits(QUIET_NAN_BF16)
    } else if v.is_subnormal() {
        if v.is_sign_positive() {
            bf16::ZERO
        } else {
            bf16::NEG_ZERO
        }
    } else {
        bf16::from_f32(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A subnormal flushes to a zero of its own sign; the neighbouring normal does not.
    #[test]
    fn subnormals_flush_to_signed_zero() {
        assert_eq!(f32_to_bf16(f32::MIN_POSITIVE / 2.0).to_bits(), bf16::ZERO.to_bits());
        assert_eq!(
            f32_to_bf16(-f32::MIN_POSITIVE / 2.0).to_bits(),
            bf16::NEG_ZERO.to_bits()
        );
        assert_ne!(f32_to_bf16(f32::MIN_POSITIVE).to_bits(), bf16::ZERO.to_bits());

        let sub = bf16::from_bits(0x0001);
        assert!(sub.is_finite() && !sub.is_normal());
        assert_eq!(bf16_to_f32(sub).to_bits(), 0.0f32.to_bits());
        assert_eq!(bf16_to_f32(-sub).to_bits(), (-0.0f32).to_bits());
    }

    /// Both directions canonicalize NaN, so a payload cannot survive. Widening `0x7f81` naively
    /// would give `0x7f81_0000`, which is what the check would miss if the branch were dropped.
    #[test]
    fn nan_canonicalizes_in_both_directions() {
        assert_eq!(f32_to_bf16(f32::from_bits(0x7fa0_0001)).to_bits(), QUIET_NAN_BF16);
        assert_eq!(bf16_to_f32(bf16::from_bits(0x7f81)).to_bits(), QUIET_NAN_F32);
    }

    /// Zero keeps its sign through the subnormal branch, and a normal value is untouched.
    #[test]
    fn zero_and_normal_pass_through() {
        assert_eq!(f32_to_bf16(-0.0).to_bits(), bf16::NEG_ZERO.to_bits());
        assert_eq!(f32_to_bf16(1.5).to_bits(), bf16::from_f32(1.5).to_bits());
        assert_eq!(bf16_to_f32(bf16::from_f32(-2.5)), -2.5);
    }
}
