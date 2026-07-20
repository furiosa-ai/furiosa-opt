//! Tag Unit configuration for Vector Engine.

use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Ident, Index, M, MappingExt};
use furiosa_opt_macro::primitive;
use smart_default::SmartDefault;

use crate::scalar::Opt;
use crate::tensor::Tensor;

use super::scalar::VeScalar;

/// Tag mode configuration for Vector Engine.
#[primitive(ve::TagMode)]
#[derive(Debug, Clone, SmartDefault)]
pub enum TagMode {
    /// No branching - all elements processed unconditionally with Tag = 0.
    #[default]
    Zero,
    /// Toggle group id (0/1) based on axis index.
    AxisToggle {
        /// Axis identifier to toggle on (e.g., Ident::I).
        /// The group ID will be determined by (coordinate-along-axis % 2).
        axis: Ident,
    },
    /// Set branch id using valid count generator.
    ValidCount,
    /// Set each branch id bit using comparison operations.
    Comparison([InputCmp; 4]),
    /// Load execution IDs from VRF (previously stored by a Comparison pass).
    /// Lowers to a branch instruction with logging support, enabling cross-TuExec branch reuse.
    Vrf,
}

impl Display for TagMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "TagMode::Zero"),
            Self::AxisToggle { axis } => write!(f, "TagMode::AxisToggle {{ axis: {axis} }}"),
            Self::ValidCount => write!(f, "TagMode::ValidCount"),
            Self::Comparison(input_cmps) => {
                write!(f, "TagMode::Comparison(")?;
                for (i, cmp) in input_cmps.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{cmp}")?;
                }
                write!(f, ")")
            }
            Self::Vrf => write!(f, "TagMode::Vrf"),
        }
    }
}

/// comparison operations for Vector Engine Tag Unit.
#[derive(Debug, Clone)]
pub enum InputCmp {
    /// i32 comparison
    I32(InputCmpI32),
    /// f32 comparison
    F32(InputCmpF32),
}

impl Display for InputCmp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32(input_cmp_i32) => write!(f, "{input_cmp_i32}"),
            Self::F32(input_cmp_f32) => write!(f, "{input_cmp_f32}"),
        }
    }
}

/// i32 comparison operations
#[derive(Debug, Clone)]
pub enum InputCmpI32 {
    /// set bit if equal to boundary
    Equal {
        /// i32 value to compare with.
        boundary: i32,
    },
    /// set bit if less than boundary
    Less {
        /// i32 value to compare with.
        boundary: i32,
    },
    /// set bit if greater than boundary
    Greater {
        /// i32 value to compare with.
        boundary: i32,
    },
    /// set bit if less than boundary (unsigned)
    LessUnsigned {
        /// i32 value to compare with.
        boundary: i32,
    },
    /// set bit if greater than boundary (unsigned)
    GreaterUnsigned {
        /// i32 value to compare with.
        boundary: i32,
    },
    /// always true
    True,
    /// always false
    False,
}

impl Display for InputCmpI32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equal { boundary } => write!(f, "={boundary}"),
            Self::Less { boundary } => write!(f, "<{boundary}"),
            Self::Greater { boundary } => write!(f, ">{boundary}"),
            Self::LessUnsigned { boundary } => write!(f, "<u{boundary}"),
            Self::GreaterUnsigned { boundary } => write!(f, ">u{boundary}"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
        }
    }
}

/// f32 comparison operations
#[derive(Debug, Clone)]
pub enum InputCmpF32 {
    /// set bit if equal to boundary
    Equal {
        /// f32 value to compare with.
        boundary: f32,
    },
    /// set bit if less than boundary
    Less {
        /// f32 value to compare with.
        boundary: f32,
    },
    /// set bit if greater than boundary
    Greater {
        /// f32 value to compare with.
        boundary: f32,
    },
    /// set bit if less than boundary (unsigned, compares bit representation)
    LessUnsigned {
        /// f32 value to compare with.
        boundary: f32,
    },
    /// set bit if greater than boundary (unsigned, compares bit representation)
    GreaterUnsigned {
        /// f32 value to compare with.
        boundary: f32,
    },
    /// always true
    True,
    /// always false
    False,
}

impl Display for InputCmpF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equal { boundary } => write!(f, "={boundary}"),
            Self::Less { boundary } => write!(f, "<{boundary}"),
            Self::Greater { boundary } => write!(f, ">{boundary}"),
            Self::LessUnsigned { boundary } => write!(f, "<u{boundary}"),
            Self::GreaterUnsigned { boundary } => write!(f, ">u{boundary}"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
        }
    }
}

impl InputCmpI32 {
    /// Check if i32 value matches this comparison
    pub(crate) fn matches(&self, x: i32) -> bool {
        match self {
            InputCmpI32::Equal { boundary } => x == *boundary,
            InputCmpI32::Less { boundary } => x < *boundary,
            InputCmpI32::Greater { boundary } => x > *boundary,
            InputCmpI32::LessUnsigned { boundary } => (x as u32) < (*boundary as u32),
            InputCmpI32::GreaterUnsigned { boundary } => (x as u32) > (*boundary as u32),
            InputCmpI32::True => true,
            InputCmpI32::False => false,
        }
    }
}

impl InputCmpF32 {
    /// Check if f32 value matches this comparison
    pub(crate) fn matches(&self, x: f32) -> bool {
        match self {
            InputCmpF32::Equal { boundary } => x == *boundary,
            InputCmpF32::Less { boundary } => x < *boundary,
            InputCmpF32::Greater { boundary } => x > *boundary,
            InputCmpF32::LessUnsigned { boundary } => {
                let x_bits = x.to_bits();
                let boundary_bits = boundary.to_bits();
                x_bits < boundary_bits
            }
            InputCmpF32::GreaterUnsigned { boundary } => {
                let x_bits = x.to_bits();
                let boundary_bits = boundary.to_bits();
                x_bits > boundary_bits
            }
            InputCmpF32::True => true,
            InputCmpF32::False => false,
        }
    }
}

impl InputCmp {
    /// Generic matches method that dispatches to type-specific implementation
    pub(crate) fn matches<D: VeScalar>(&self, x: D) -> bool {
        use std::any::TypeId;
        match self {
            InputCmp::I32(cmp) => {
                if TypeId::of::<D>() == TypeId::of::<i32>() {
                    unsafe {
                        let x_i32 = std::mem::transmute_copy::<D, i32>(&x);
                        cmp.matches(x_i32)
                    }
                } else {
                    panic!("Type mismatch: InputCmp::I32 used with f32 data")
                }
            }
            InputCmp::F32(cmp) => {
                if TypeId::of::<D>() == TypeId::of::<f32>() {
                    unsafe {
                        let x_f32 = std::mem::transmute_copy::<D, f32>(&x);
                        cmp.matches(x_f32)
                    }
                } else {
                    panic!("Type mismatch: InputCmp::F32 used with i32 data")
                }
            }
        }
    }
}

/// GroupId: msb 1 bit of branch id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupId {
    /// Group 0
    Zero,
    /// Group 1
    One,
}

impl GroupId {
    /// Returns the bit value of the GroupId.
    pub(crate) fn bit_value(&self) -> u8 {
        match self {
            GroupId::Zero => 0,
            GroupId::One => 1,
        }
    }
}

/// Tag ID configuration for Vector Engine operations.
///
/// Controls which elements are processed based on their execution ID (set by branch unit).
/// The execution ID's MSB (bit 3) represents the group ID (0 or 1).
///
/// - `Group { id }`: Only elements whose group ID matches are processed.
///   Used for conditional execution based on branch conditions.
/// - `All`: All elements are processed regardless of their branch ID.
///   This is the default for operations that don't need branching.
#[primitive(ve::TagFilter)]
#[derive(Debug, Clone, Default)]
pub enum TagFilter {
    /// Valid only for a specific group (filtered by MSB of tag).
    Group {
        /// The group ID to filter by.
        id: GroupId,
    },
    /// Always valid regardless of branch ID.
    #[default]
    All,
}

impl TagFilter {
    /// Check if this branch config matches the given execution ID.
    /// Only Init values can match - Uninit never matches any config.
    pub(crate) fn matches(&self, exec_id: Opt<u8>) -> bool {
        match (self, exec_id) {
            (_, Opt::Uninit) => false,
            (TagFilter::All, Opt::Init(_)) => true,
            (TagFilter::Group { id }, Opt::Init(eid_val)) => ((eid_val >> 3) & 1) == id.bit_value(),
        }
    }
}

impl From<GroupId> for TagFilter {
    fn from(id: GroupId) -> Self {
        TagFilter::Group { id }
    }
}

/// Applies branch unit to generate Tag for each element.
pub(crate) fn apply_branch_config<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    config: &TagMode,
) -> Tensor<u8, Mapping> {
    match config {
        TagMode::Zero => data.map(|_| 0u8),
        TagMode::AxisToggle { axis } => Tensor::<u8, Mapping>::from_vec(axis_toggle_pattern::<Mapping>(axis)),
        TagMode::ValidCount => todo!(),
        TagMode::Vrf => todo!("TagMode::Vrf: load execution IDs from VRF (GenBranch::WithLog)"),
        TagMode::Comparison(cmps) => data.map(|x| {
            let mut exec_id: u8 = 0;
            for (bit_pos, cmp) in cmps.iter().enumerate() {
                let bit = if cmp.matches(x) { 0x1 } else { 0x0 };
                exec_id |= bit << bit_pos;
            }
            exec_id
        }),
    }
}

/// Host-side `AxisToggle` tag pattern: `(coord_along_axis % 2) << 3` per cell, padding cells → 0.
fn axis_toggle_pattern<Mapping: M>(axis: &Ident) -> Vec<u8> {
    let mapping = Mapping::to_value();
    let axes = mapping.axes();
    let Some(pos) = crate::storage::axis_position(&axes, axis) else {
        return vec![0u8; mapping.size()]; // axis absent: nothing to toggle
    };
    // The toggle axis's coordinate is its digit in the canonical offset: divide out the inner axes'
    // weight, mod its extent. The lazy wire walk gives each cell's offset (padding → `None` → 0), so
    // no per-cell `finalize`.
    let weight: usize = axes[pos + 1..].iter().map(|a| a.modulo).product();
    let modulo = axes[pos].modulo;
    mapping
        .iter(&axes, &Index::new(), true)
        .map(|offset| offset.map_or(0u8, |o| (((o / weight) % modulo % 2) as u8) << 3))
        .collect()
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::axis_toggle_pattern;

    /// `axis_toggle_pattern` decodes each cell's coordinate along the toggle axis (its digit in the
    /// canonical offset) and emits `(coord % 2) << 3`. Hand-computed for `m![A, B]` (A=4 outer, B=2
    /// inner; wire cell `(a, b)` at `a*2 + b`).
    #[test]
    fn axis_toggle_pattern_decodes_axis_parity() {
        axes![A = 4, B = 2];
        // Toggle A: cell (a, b) -> (a % 2) << 3. a = position / 2.
        assert_eq!(axis_toggle_pattern::<m![A, B]>(&Ident::A), vec![0, 0, 8, 8, 0, 0, 8, 8]);
        // Toggle B: cell (a, b) -> (b % 2) << 3. b = position % 2.
        assert_eq!(axis_toggle_pattern::<m![A, B]>(&Ident::B), vec![0, 8, 0, 8, 0, 8, 0, 8]);
    }
}
