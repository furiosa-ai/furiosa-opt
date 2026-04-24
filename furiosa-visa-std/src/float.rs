//! Hardware numeric type operations for Renegade NPU.

use std::ops::Mul;

pub(crate) const F8E4_ZERO: u8 = 0x00;
pub(crate) const F8E4_NEG_ZERO: u8 = 0x80;
pub(crate) const F8E4_ONE: u8 = 0x38;

pub(crate) fn f8_e4_to_f32(bits: u8) -> f32 {
    f32::from_bits(F8E4_TO_F32[bits as usize])
}

pub(crate) fn f8_e4_from_f32(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f; // qNaN
    }
    let n = v.to_bits();
    let exp = (n >> 23) & 0xff;
    let man = n & 0x7fffff;
    let abs_value = if exp > 127 - 7 {
        let valid = (n >> 20) & 0x7ff;
        let val = if (n & 0xfffff) + (valid & 1) > 0x80000 {
            valid + 1
        } else {
            valid
        };
        if val >= (((127 + 8) << 3) | 0x6) {
            0x7e_u32 // MAX
        } else {
            val - ((127 - 7) << 3)
        }
    } else if exp == 127 - 7 {
        if man >= 0x700000 {
            8
        } else if man > 0x500000 {
            7
        } else if man >= 0x300000 {
            6
        } else if man > 0x100000 {
            5
        } else {
            4
        }
    } else if exp == 127 - 8 {
        if man >= 0x600000 {
            4
        } else if man > 0x200000 {
            3
        } else {
            2
        }
    } else if exp == 127 - 9 {
        if man >= 0x400000 { 2 } else { 1 }
    } else if exp == 127 - 10 && man != 0 {
        1
    } else {
        0
    };
    (((n >> 31) << 7) | abs_value) as u8
}

pub(crate) fn f8_e4_is_zero(bits: u8) -> bool {
    bits == F8E4_ZERO || bits == F8E4_NEG_ZERO
}

#[rustfmt::skip]
const F8E4_TO_F32: &[u32; 256] = &[
    0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000,
    0x3c800000, 0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000,
    0x3d000000, 0x3d100000, 0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000,
    0x3d800000, 0x3d900000, 0x3da00000, 0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000,
    0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000, 0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000,
    0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000, 0x3ed00000, 0x3ee00000, 0x3ef00000,
    0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
    0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000, 0x3ff00000,
    0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
    0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000,
    0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
    0x41800000, 0x41900000, 0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000,
    0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
    0x42800000, 0x42900000, 0x42a00000, 0x42b00000, 0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000,
    0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
    0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000, 0x43e00000, 0x7fc00000,
    0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000, 0xbc600000,
    0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
    0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000,
    0xbd800000, 0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000,
    0xbe000000, 0xbe100000, 0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000,
    0xbe800000, 0xbe900000, 0xbea00000, 0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000,
    0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000, 0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000,
    0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000, 0xbfd00000, 0xbfe00000, 0xbff00000,
    0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000, 0xc0600000, 0xc0700000,
    0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000, 0xc0f00000,
    0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
    0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000,
    0xc2000000, 0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000,
    0xc2800000, 0xc2900000, 0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000,
    0xc3000000, 0xc3100000, 0xc3200000, 0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000,
    0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000, 0xc3c00000, 0xc3d00000, 0xc3e00000, 0x7fc00000,
];

// ============================================================================
// Fixed-point ↔ float conversion
// ============================================================================

const MANTISSA_MASK: u32 = (1 << 23) - 1;
const EXPONENT_MASK: u32 = (1 << 8) - 1;
const EXPONENT_SHIFT: i32 = 23;
const SIGN_SHIFT: i32 = 31;
const EXPONENT_OFFSET: u32 = (1 << 7) - 1;

pub(crate) fn float_to_fixedpoint(input: f32, integer_width: u32) -> i32 {
    assert!(integer_width < 32);
    if input.is_nan() {
        return if input.is_sign_positive() { i32::MAX } else { i32::MIN };
    }

    let fraction_width = 31 - integer_width;
    let f64_result: f64 = f64::from(input)
        .mul(2_f64.powi(fraction_width as i32))
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64);
    f64_result as i32
}

pub(crate) fn fixedpoint_to_float(input: i32, integer_width: u32) -> f32 {
    assert!(integer_width < 32);
    let abs_value: u32 = input.unsigned_abs();

    let mask = (1 << 31) - 1;
    let leading_zeros = (abs_value & mask).leading_zeros() - 1;

    let aligned_abs_value = abs_value << leading_zeros;
    let rounded = round_to_nearest_even(aligned_abs_value, 31, 24);
    let mantissa = rounded & MANTISSA_MASK;

    let sign = if input < 0 { 1 } else { 0 };

    let exponent = if input == i32::MIN {
        integer_width + EXPONENT_OFFSET
    } else if leading_zeros == 31 {
        0
    } else {
        let mut result = integer_width + (EXPONENT_OFFSET - 1) - leading_zeros;
        if (rounded >> 24) != 0 {
            result += 1;
        }
        result
    };
    let float_as_bits = (sign << SIGN_SHIFT) | ((exponent & EXPONENT_MASK) << EXPONENT_SHIFT) | mantissa;
    f32::from_bits(float_as_bits)
}

fn round_to_nearest_even(input: u32, iw: u32, ow: u32) -> u32 {
    let in_width_mask = (1u32 << iw) - 1;
    let input_valid = input & in_width_mask;
    let shift = (iw - ow) - 2;

    let shift_mask = (1u32 << shift) - 1;
    let discarded = input_valid & shift_mask;
    let sticky = discarded != 0;

    let interest = input_valid >> shift;
    let round_factor = ((interest & 0x7) << 1) | u32::from(sticky);

    let round_up = matches!(
        round_factor,
        0b0110 | 0b1110 | 0b0101 | 0b1101 | 0b0111 | 0b1111 | 0b1100
    );

    (interest >> 2) + u32::from(round_up)
}
