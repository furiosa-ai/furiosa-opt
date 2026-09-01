//! Safe entry points for lowering operations backed by the private implementation.

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};
use furiosa_opt_lower_types::{
    CommitError, DivideError, DivideTerm, FactorLeaf, FetchBaseError, FetchError, PadError, RelaxedDivision,
    StreamSequencerConfig, SwitchConfig, SwitchError, TileError, TransposeConfig, TransposeError,
};

use crate::sys;

/// Inputs used to configure the transpose engine.
pub struct TransposeInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
    pub element_bits: usize,
}

/// Inputs used to configure the Fetch engine's stream descriptors.
pub struct FetchInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
    pub lifted: Option<Mapping>,
}

/// Inputs used to compute Fetch lift base offsets.
pub struct FetchLiftBasesInput {
    pub grid: Mapping,
    pub dm: Mapping,
}

/// Inputs used to check Fetch base alignment.
pub struct FetchBaseAlignedInput {
    pub bases: Vec<usize>,
    pub element_bits: usize,
}

/// Inputs used to configure the commit engine's write descriptors.
pub struct CommitInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub element: Mapping,
    pub element_bits: usize,
}

/// Mappings used for exact or relaxed division.
pub struct DivideInput {
    pub dividend: Mapping,
    pub divisor: Mapping,
}

/// Inputs used to configure a switch topology.
pub struct SwitchInput {
    pub config: SwitchConfig,
    pub in_slice: Mapping,
    pub in_time: Mapping,
    pub out_slice: Mapping,
    pub out_time: Mapping,
}

/// Inputs used to derive a custom-broadcast snoop bitmap.
pub struct SwitchCustomSnoopInput {
    pub in_slice: Mapping,
    pub in_time: Mapping,
    pub out_slice: Mapping,
    pub out_time: Mapping,
    pub ring_size: usize,
}

/// Inputs used to validate a tile view.
pub struct TileInput {
    pub index: Mapping,
    pub element: Mapping,
    pub expected: Mapping,
    pub len: usize,
    /// Accessibility of cells outside the live window.
    pub hole_fill: PaddingKind,
}

/// Inputs used to validate a padded view.
pub struct PadInput {
    pub element: Mapping,
    pub expected: Mapping,
}

/// Extends a mapping with its factor decomposition.
pub trait FactorLeavesExt {
    /// Returns factor leaves from innermost to outermost.
    fn factor_leaves(&self) -> Vec<FactorLeaf>;
}

impl FactorLeavesExt for Mapping {
    fn factor_leaves(&self) -> Vec<FactorLeaf> {
        unsafe { sys::mapping_factor_leaves(self) }.into()
    }
}

/// Resolves transpose-engine hardware parameters.
pub fn config_transpose(input: TransposeInput) -> Result<TransposeConfig, TransposeError> {
    unsafe {
        sys::config_transpose(
            &input.in_time,
            &input.in_packet,
            &input.out_time,
            &input.out_packet,
            input.element_bits,
        )
    }
    .into_result()
}

/// Synthesizes the Fetch engine's stream descriptors.
pub fn config_fetch(input: FetchInput) -> Result<StreamSequencerConfig, FetchError> {
    unsafe {
        sys::config_fetch(
            &input.in_time,
            &input.in_packet,
            &input.out_time,
            &input.out_packet,
            input.lifted.as_ref().into(),
        )
    }
    .into_result()
}

/// Returns the DM offset read at each placement-grid position, in elements.
pub fn config_fetch_lift_bases(input: FetchLiftBasesInput) -> Result<Vec<usize>, FetchBaseError> {
    unsafe { sys::config_fetch_lift_bases(&input.grid, &input.dm) }
        .into_result()
        .map(Vec::from)
}

/// Checks that fetch-base addresses satisfy the supported alignment.
pub fn config_fetch_base_aligned(input: FetchBaseAlignedInput) -> Result<(), FetchBaseError> {
    unsafe { sys::config_fetch_base_aligned(input.bases.as_slice().into(), input.element_bits) }.into_result()
}

/// Synthesizes the commit engine's write descriptors.
pub fn config_commit(input: CommitInput) -> Result<StreamSequencerConfig, CommitError> {
    unsafe { sys::config_commit(&input.in_time, &input.in_packet, &input.element, input.element_bits) }.into_result()
}

/// Carves `divisor` out of `dividend` exactly.
pub fn config_divide_exact(input: DivideInput) -> Result<Vec<DivideTerm>, DivideError> {
    unsafe { sys::config_divide_exact(&input.dividend, &input.divisor) }
        .into_result()
        .map(Vec::from)
}

/// Returns a relaxed division of `dividend` by `divisor`.
pub fn config_divide_relaxed(input: DivideInput) -> RelaxedDivision {
    unsafe { sys::config_divide_relaxed(&input.dividend, &input.divisor) }
}

/// Validates a switch topology against the slice and time shapes.
pub fn config_switch(input: SwitchInput) -> Result<SwitchConfig, SwitchError> {
    unsafe {
        sys::config_switch(
            &input.config,
            &input.in_slice,
            &input.in_time,
            &input.out_slice,
            &input.out_time,
        )
    }
    .into_result()
}

/// Returns the input slice lanes read by each output slice.
pub fn switch_custom_snoop_bitmap(input: SwitchCustomSnoopInput) -> Result<Vec<Vec<usize>>, SwitchError> {
    unsafe {
        sys::switch_custom_snoop_bitmap(
            &input.in_slice,
            &input.in_time,
            &input.out_slice,
            &input.out_time,
            input.ring_size,
        )
    }
    .into_result()
    .map(|bitmap| bitmap.into_iter().map(Vec::from).collect())
}

/// Validates an indexed tile view against its expected mapping.
pub fn config_tile(input: TileInput) -> Result<(), TileError> {
    let TileInput {
        index,
        element,
        expected,
        len,
        hole_fill,
    } = input;
    let size = index.size();
    let split = if size == 1 {
        if len != 1 {
            return Err(TileError::Split);
        }
        element.normalize()
    } else {
        // A located axis may contain a padded final chunk whose live extent is not divisible by its stride.
        let stride = element.find_axis(&index).map_err(|_| TileError::Split)?;
        element
            .window_axis(stride, size, len, hole_fill)
            .map_err(|_| TileError::Split)?
    };

    let requested = expected.normalize();
    if split == requested {
        Ok(())
    } else {
        Err(TileError::UnexpectedView { split, requested })
    }
}

/// Validates a padded view against its expected mapping.
pub fn config_pad(input: PadInput) -> Result<(), PadError> {
    unsafe { sys::config_pad(&input.element, &input.expected) }.into_result()
}
