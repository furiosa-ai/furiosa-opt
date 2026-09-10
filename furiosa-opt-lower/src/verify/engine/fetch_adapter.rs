//! Fetch size: which hardware fetch width one Fetch Unit read may use.

use crate::verify::{BITS_PER_BYTE, FLIT_BYTES, SRAM_ACCESS_BYTES, length_from_bytes};

/// Which Fetch Unit is reading, and so which fetch sizes it may choose from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchContext {
    Main,
    Sub,
}

impl FetchContext {
    const MAIN_FETCH_SIZES: &'static [usize] = &[1, 2, 4, 8, 16, 32];
    const SUB_FETCH_SIZES: &'static [usize] = &[SRAM_ACCESS_BYTES];
    /// `i4` to `i32` is the one sub-context read whose width is not an access word.
    const SUB_I4_TO_I32_FETCH_SIZES: &'static [usize] = &[4];

    /// Fetch sizes this context offers, in input bytes.
    pub fn fetch_sizes(self, input_element_bits: usize, output_element_bits: usize) -> &'static [usize] {
        match self {
            Self::Main => Self::MAIN_FETCH_SIZES,
            Self::Sub if input_element_bits == 4 && output_element_bits == 32 => Self::SUB_I4_TO_I32_FETCH_SIZES,
            Self::Sub => Self::SUB_FETCH_SIZES,
        }
    }
}

fn fetch_volume(
    fetch_size: usize,
    input_element_bits: usize,
    output_element_bits: usize,
    volume_to_fetch: usize,
) -> Option<usize> {
    let volume = length_from_bytes(input_element_bits, fetch_size).ok()?;
    let network_bits = volume.checked_mul(output_element_bits)?;
    let access_bits = SRAM_ACCESS_BYTES * BITS_PER_BYTE;
    let fits_network = network_bits > 0
        && network_bits <= FLIT_BYTES * BITS_PER_BYTE
        && (fetch_size >= SRAM_ACCESS_BYTES || network_bits.is_multiple_of(access_bits));
    let tiles = volume_to_fetch > 0 && volume_to_fetch.is_multiple_of(volume);
    (fits_network && tiles).then_some(volume)
}

/// No fetch size this context offers can serve the read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "no legal {context:?} fetch size for {volume_to_fetch} element(s) of a \
     {input_element_bits}-bit read leaving {output_element_bits} bits wide"
)]
pub struct FetchVolumeError {
    pub context: FetchContext,
    pub input_element_bits: usize,
    pub output_element_bits: usize,
    pub volume_to_fetch: usize,
}

/// The widest fetch that can serve this read, in elements.
pub fn config_fetch_volume(
    context: FetchContext,
    input_element_bits: usize,
    output_element_bits: usize,
    volume_to_fetch: usize,
) -> Result<usize, FetchVolumeError> {
    context
        .fetch_sizes(input_element_bits, output_element_bits)
        .iter()
        .filter_map(|&fetch_size| fetch_volume(fetch_size, input_element_bits, output_element_bits, volume_to_fetch))
        .max()
        .ok_or(FetchVolumeError {
            context,
            input_element_bits,
            output_element_bits,
            volume_to_fetch,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_require_whole_elements_a_legal_network_payload_and_tiling() {
        let cases = [
            ((4, 8, 32, 4), Some(4)),
            ((2, 8, 32, 4), Some(2)),
            ((1, 8, 32, 4), None),
            ((4, 8, 8, 4), None),
            ((8, 8, 8, 8), Some(8)),
            ((4, 4, 32, 8), Some(8)),
            ((1, 32, 32, 4), None),
            ((8, 32, 32, 4), Some(2)),
            ((32, 8, 32, 32), None),
            ((8, 8, 32, 4), None),
        ];
        for ((fetch_size, in_bits, out_bits, volume_to_fetch), expected) in cases {
            assert_eq!(
                fetch_volume(fetch_size, in_bits, out_bits, volume_to_fetch),
                expected,
                "{fetch_size} B, {in_bits}-bit -> {out_bits}-bit, tiling {volume_to_fetch}",
            );
        }
    }

    #[test]
    fn the_widest_serving_fetch_wins() {
        assert_eq!(config_fetch_volume(FetchContext::Main, 8, 32, 16), Ok(8));
        assert_eq!(config_fetch_volume(FetchContext::Main, 8, 32, 4), Ok(4));
    }

    #[test]
    fn reports_when_no_fetch_size_serves_the_read() {
        assert_eq!(
            config_fetch_volume(FetchContext::Main, 32, 32, 3),
            Err(FetchVolumeError {
                context: FetchContext::Main,
                input_element_bits: 32,
                output_element_bits: 32,
                volume_to_fetch: 3,
            })
        );
    }
}
