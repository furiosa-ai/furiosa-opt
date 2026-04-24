//! Branch Unit configuration for Vector Engine.

use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Atom, Ident, M};
use furiosa_mapping_macro::primitive;
use smart_default::SmartDefault;

use crate::scalar::Opt;
use crate::tensor::Tensor;

use super::scalar::VeScalar;

/// Branch mode configuration for Vector Engine.
#[primitive(ve::BranchMode)]
#[derive(Debug, Clone, SmartDefault)]
pub enum BranchMode {
    /// No branching - all elements processed unconditionally with ExecutionId = 0.
    #[default]
    Unconditional,
    /// Toggle group id (0/1) based on axis index.
    AxisToggle {
        /// Axis identifier to toggle on (e.g., Ident::I).
        /// The group ID will be determined by (axis_index % 2).
        axis: Ident,
    },
    /// Set branch id using valid count generator.
    ValidCount,
    /// Set each branch id bit using comparison operations.
    Comparison([InputCmp; 4]),
    /// Load execution IDs from VRF (previously stored by a Comparison pass).
    /// Maps to npu-ir `GenBranch::WithLog`. Enables cross-TuExec branch reuse.
    Vrf,
}

impl Display for BranchMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unconditional => write!(f, "BranchMode::Unconditional"),
            Self::AxisToggle { axis } => write!(f, "BranchMode::AxisToggle {{ axis: {axis} }}"),
            Self::ValidCount => write!(f, "BranchMode::ValidCount"),
            Self::Comparison(input_cmps) => {
                write!(f, "BranchMode::Comparison(")?;
                for (i, cmp) in input_cmps.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{cmp}")?;
                }
                write!(f, ")")
            }
            Self::Vrf => write!(f, "BranchMode::Vrf"),
        }
    }
}

/// comparison operations for Vector Engine Branch Unit.
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
    pub fn matches(&self, x: i32) -> bool {
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
    pub fn matches(&self, x: f32) -> bool {
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
    pub fn matches<D: VeScalar>(&self, x: D) -> bool {
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
    pub fn bit_value(&self) -> u8 {
        match self {
            GroupId::Zero => 0,
            GroupId::One => 1,
        }
    }
}

/// Branch ID configuration for Vector Engine operations.
///
/// Controls which elements are processed based on their execution ID (set by branch unit).
/// The execution ID's MSB (bit 3) represents the group ID (0 or 1).
///
/// - `ValidGroup { id }`: Only elements whose group ID matches are processed.
///   Used for conditional execution based on branch conditions.
/// - `ValidAlways`: All elements are processed regardless of their branch ID.
///   This is the default for operations that don't need branching.
#[primitive(ve::ValidBranchIds)]
#[derive(Debug, Clone, Default)]
pub enum ValidBranchIds {
    /// Valid only for a specific group (filtered by MSB of execution_id).
    ValidGroup {
        /// The group ID to filter by.
        id: GroupId,
    },
    /// Always valid regardless of branch ID.
    #[default]
    ValidAlways,
}

impl ValidBranchIds {
    /// Check if this branch config matches the given execution ID.
    /// Only Init values can match - Uninit never matches any config.
    pub fn matches(&self, exec_id: Opt<u8>) -> bool {
        match (self, exec_id) {
            (_, Opt::Uninit) => false,
            (ValidBranchIds::ValidAlways, Opt::Init(_)) => true,
            (ValidBranchIds::ValidGroup { id }, Opt::Init(eid_val)) => ((eid_val >> 3) & 1) == id.bit_value(),
        }
    }
}

impl From<GroupId> for ValidBranchIds {
    fn from(id: GroupId) -> Self {
        ValidBranchIds::ValidGroup { id }
    }
}

/// Applies branch unit to generate ExecutionId for each element.
pub fn apply_branch_config<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    config: &BranchMode,
) -> Tensor<u8, Mapping> {
    match config {
        BranchMode::Unconditional => data.map(|_| Opt::Init(0u8)),
        BranchMode::AxisToggle { axis } => Tensor::from_fn(|axes, idx| {
            let axis_pos = axes.iter().position(|term| {
                if let Atom::Symbol { symbol, .. } = &term.inner {
                    symbol == axis
                } else {
                    false
                }
            });

            if let Some(pos) = axis_pos {
                let axis_idx = idx[pos];
                let group_id = (axis_idx % 2) as u8;
                let exec_id = group_id << 3;
                Opt::Init(exec_id)
            } else {
                Opt::Init(0u8)
            }
        }),
        BranchMode::ValidCount => todo!(),
        BranchMode::Vrf => todo!("BranchMode::Vrf: load execution IDs from VRF (GenBranch::WithLog)"),
        BranchMode::Comparison(cmps) => data.map(|x| match x {
            Opt::Init(x) => {
                let mut exec_id: u8 = 0;
                for (bit_pos, cmp) in cmps.iter().enumerate() {
                    let bit = if cmp.matches(*x) { 0x1 } else { 0x0 };
                    exec_id |= bit << bit_pos;
                }

                Opt::Init(exec_id)
            }
            Opt::Uninit => Opt::Uninit,
        }),
    }
}
