//! Hardware numeric type operations for Renegade NPU.

mod bf16;
mod f8e4;
mod f8e5;

pub(crate) use f8e4::{F8E4_ONE, F8E4_ZERO, f8_e4_from_f32, f8_e4_is_zero, f8_e4_to_f32};
pub(crate) use f8e5::{F8E5_ONE, F8E5_ZERO, f8_e5_from_f32, f8_e5_is_zero, f8_e5_to_f32};

use std::ops::Mul;

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

pub(crate) use bf16::{bf16_to_f32, f32_to_bf16};
