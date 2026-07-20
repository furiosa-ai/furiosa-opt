//! E5M2 (1 sign, 5-bit exponent, 2-bit mantissa) 8-bit float conversions.
//!
//! E5M2 is IEEE-754-like, so conversion to/from f32 is a direct bit reshuffle
//! (rebias the exponent and align the mantissa) rather than a lookup table.

pub(crate) const F8E5_ZERO: u8 = 0x00;
pub(crate) const F8E5_NEG_ZERO: u8 = 0x80;
pub(crate) const F8E5_ONE: u8 = 0x3c; // sign 0, exp 15, mantissa 0

pub(crate) fn f8_e5_to_f32(bits: u8) -> f32 {
    let sign = ((bits >> 7) & 1) as u32;
    let exp = ((bits >> 2) & 0x1f) as u32;
    let mant = (bits & 0x3) as u32;
    match exp {
        // Zero or subnormal: value = (-1)^s * mant * 2^-16.
        0 => {
            let m = mant as f32 / 65536.0;
            if sign == 1 { -m } else { m }
        }
        // Inf (mantissa 0) or NaN.
        0x1f => {
            let f = if mant == 0 { f32::INFINITY } else { f32::NAN };
            if sign == 1 { -f } else { f }
        }
        // Normal: rebias exponent (15 -> 127) and left-align the mantissa.
        // `exp + 127 - 15` (not `exp - 15 + 127`) so the u32 never underflows.
        _ => f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 21)),
    }
}

pub(crate) fn f8_e5_from_f32(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7e; // qNaN: exp 0x1f, mantissa nonzero
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    if a.is_infinite() {
        return sign | 0x7c; // inf: exp 0x1f, mantissa 0
    }
    if a == 0.0 {
        return sign;
    }
    let n = a.to_bits();
    let unbiased_exp = ((n >> 23) & 0xff) as i32 - 127;
    let man = n & 0x7fffff;
    let biased = unbiased_exp + 15;
    if biased >= 0x1f {
        return sign | 0x7c; // overflow saturates to inf
    }
    if biased <= 0 {
        // Subnormal: round a / 2^-16 to the nearest representable mantissa.
        let sub = (a * 65536.0).round() as u32;
        return sign | (sub.min(3) as u8);
    }
    // Normal: keep the top 2 mantissa bits, round half up (carry into exponent).
    let mut mant2 = man >> 21;
    let mut exp = biased as u32;
    if (man >> 20) & 1 == 1 {
        mant2 += 1;
        if mant2 == 4 {
            mant2 = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c;
            }
        }
    }
    sign | ((exp as u8) << 2) | (mant2 as u8)
}

pub(crate) fn f8_e5_is_zero(bits: u8) -> bool {
    bits == F8E5_ZERO || bits == F8E5_NEG_ZERO
}
