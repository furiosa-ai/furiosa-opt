//! Tag Unit configuration for Vector Engine.

use std::fmt::{self, Debug, Display, Formatter};

use furiosa_mapping::{Ident, Index, M, MappingExt};
use furiosa_opt_macro::primitive;
use smart_default::SmartDefault;

use crate::tensor::Tensor;

use super::scalar::VeScalar;

// The guard vocabulary is shared with the compiler IR, so it lives in `furiosa-opt-common-ir` and is
// re-exported here as part of this module's surface. `BranchedOperand` deliberately is not: a layout is
// reached through [`Branched`](super::operand::Branched), the only thing that can build a good one.
pub use furiosa_opt_common_ir::{BitReq, ExecutionId, GroupId, TagGuard};

// Named in the type aliases below, and in the op methods' signatures, so it has to be in scope here.
use furiosa_opt_common_ir::BranchedOperand;

/// Tag mode configuration for Vector Engine.
#[primitive(ve::TagMode)]
#[derive(Debug, Clone, SmartDefault)]
pub enum TagMode<D: VeScalar> {
    /// No branching - all elements processed unconditionally with Tag = 0.
    #[default]
    Zero,
    /// Toggle group id (0/1) based on axis index.
    AxisToggle {
        /// Axis identifier to toggle on (e.g., Ident::I).
        /// The group ID will be determined by (coordinate-along-axis % 2).
        axis: Ident,
    },
    /// Set each branch id bit using comparison operations.
    Comparison([Cmp<D>; 4]),
    // Withheld: neither runs on the host, and `ValidCount` does lower to the branch unit's augmenter,
    // so a kernel naming it used to compile for the device and then panic on the host engine -- an op
    // public on only one of the two paths. Restore a variant with its `apply_branch_config` arm.
    //
    // /// Set branch id using valid count generator.
    // ValidCount,
    // /// Load execution IDs from VRF (previously stored by a Comparison pass).
    // /// Lowers to a branch instruction with logging support, enabling cross-TuExec branch reuse.
    // Vrf,
}

impl<D: VeScalar + Display> Display for TagMode<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "TagMode::Zero"),
            Self::AxisToggle { axis } => write!(f, "TagMode::AxisToggle {{ axis: {axis} }}"),
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
        }
    }
}

/// One comparison the branch unit runs against each element, typed by the stream it reads.
///
/// Each variant carries the boundary it compares against. `D` is the stream's scalar, so the boundary
/// is checked against it: on an `f32` stream `Cmp::Less(0.0)` compiles and `Cmp::Less(0)` does not.
///
/// The `*Unsigned` pair compares the raw bit pattern as an unsigned integer, which is one hardware op
/// whichever scalar the stream carries; it is not `Less` on some other `D`, so it keeps its own
/// variants.
#[primitive(ve::Cmp)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cmp<D: VeScalar> {
    /// Set the bit where the element equals the boundary.
    Equal(D),
    /// Set the bit where the element is less than the boundary.
    Less(D),
    /// Set the bit where the element is greater than the boundary.
    Greater(D),
    /// Set the bit where the element's bit pattern, read as unsigned, is less than the boundary's.
    LessUnsigned(D),
    /// Set the bit where the element's bit pattern, read as unsigned, is greater than the boundary's.
    GreaterUnsigned(D),
    /// Always set the bit.
    True,
    /// Never set the bit.
    False,
}

impl<D: VeScalar> Cmp<D> {
    /// Whether `x` satisfies this comparison. No type dispatch: `D` is the stream's scalar, so the
    /// comparison is the scalar's own. Takes `self` by value like [`TagGuard::matches`], the other
    /// per-element predicate on this path.
    #[inline]
    pub(crate) fn matches(self, x: D) -> bool {
        match self {
            Cmp::Equal(boundary) => x == boundary,
            Cmp::Less(boundary) => x.lt_scalar(boundary),
            Cmp::Greater(boundary) => boundary.lt_scalar(x),
            Cmp::LessUnsigned(boundary) => x.to_raw_bits() < boundary.to_raw_bits(),
            Cmp::GreaterUnsigned(boundary) => x.to_raw_bits() > boundary.to_raw_bits(),
            Cmp::True => true,
            Cmp::False => false,
        }
    }
}

impl<D: VeScalar + Display> Display for Cmp<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equal(boundary) => write!(f, "={boundary}"),
            Self::Less(boundary) => write!(f, "<{boundary}"),
            Self::Greater(boundary) => write!(f, ">{boundary}"),
            Self::LessUnsigned(boundary) => write!(f, "<u{boundary}"),
            Self::GreaterUnsigned(boundary) => write!(f, ">u{boundary}"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
        }
    }
}

/// What a slot's register-file port reads. A VE pass has exactly one such port, so these two reads
/// are mutually exclusive and the choice belongs in the payload rather than alongside it.
#[derive(Debug)]
pub enum RfPort<D: VeScalar, Mapping: M> {
    /// Reads an external register: the VRF tensor, already transposed to the op's tensor shape.
    External(Tensor<D, Mapping>),
    /// Reads the stash. It carries no value here because the stashed tensor is the pipeline's, not
    /// the operand's.
    Stash,
}

/// One binary VE pass's slots: each immediate register holds a constant, the rf port holds a register
/// or stash read.
pub type BinaryBranchedOperand<D, Mapping> = BranchedOperand<D, RfPort<D, Mapping>>;

/// One ternary VE pass's slots: each slot also carries the `operand1` its branch feeds, which the
/// hardware selects per branch just like `operand0`, so two zipped groups can use different ones in a
/// single pass. [`VeOperandLayout`] flattens the pair for the apply path.
pub type TernaryBranchedOperand<D, Mapping> = BranchedOperand<(D, D), (RfPort<D, Mapping>, D)>;

/// How to read one pass's slots whatever they carry: [`BinaryBranchedOperand`] and
/// [`TernaryBranchedOperand`] are the same [`BranchedOperand`] with different payloads, and this trait turns
/// that into one way to iterate them. The arity surfaces only as
/// [`Operand1`](Self::Operand1) flowing into the op.
pub trait VeOperandLayout<D: VeScalar, Mapping: M> {
    /// What each slot carries besides `operand0`: `()` for a binary op, `operand1` for a ternary one.
    /// `Sync` because the apply path hands it to a parallel per-cell closure.
    type Operand1: Copy + Sync;

    /// The three immediate register slots in application order, each as
    /// `(guard, operand0, operand1)`.
    fn regs(&self) -> [Option<(&TagGuard, D, Self::Operand1)>; 3];

    /// The register-file port, as `(guard, where it reads, operand1)`.
    fn port(&self) -> Option<(&TagGuard, &RfPort<D, Mapping>, Self::Operand1)>;

    /// Whether the port reads the stash rather than a register. The apply path clones the stash only
    /// when this holds.
    fn reads_stash(&self) -> bool {
        matches!(self.port(), Some((_, RfPort::Stash, _)))
    }
}

impl<D: VeScalar, Mapping: M> VeOperandLayout<D, Mapping> for BinaryBranchedOperand<D, Mapping> {
    type Operand1 = ();

    fn regs(&self) -> [Option<(&TagGuard, D, ())>; 3] {
        self.reg_slots()
            .map(|slot| slot.as_ref().map(|(guard, operand0)| (guard, *operand0, ())))
    }

    fn port(&self) -> Option<(&TagGuard, &RfPort<D, Mapping>, ())> {
        self.rf_slot().as_ref().map(|(guard, port)| (guard, port, ()))
    }
}

impl<D: VeScalar, Mapping: M> VeOperandLayout<D, Mapping> for TernaryBranchedOperand<D, Mapping> {
    type Operand1 = D;

    fn regs(&self) -> [Option<(&TagGuard, D, D)>; 3] {
        self.reg_slots().map(|slot| {
            slot.as_ref()
                .map(|(guard, (operand0, operand1))| (guard, *operand0, *operand1))
        })
    }

    fn port(&self) -> Option<(&TagGuard, &RfPort<D, Mapping>, D)> {
        self.rf_slot()
            .as_ref()
            .map(|(guard, (port, operand1))| (guard, port, *operand1))
    }
}

/// Applies branch unit to generate Tag for each element.
pub(crate) fn apply_branch_config<D: VeScalar, Mapping: M>(
    data: &Tensor<D, Mapping>,
    config: &TagMode<D>,
) -> Tensor<u8, Mapping> {
    match config {
        TagMode::Zero => data.map(|_| 0u8),
        TagMode::AxisToggle { axis } => Tensor::<u8, Mapping>::from_vec(axis_toggle_pattern::<Mapping>(axis)),
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

    use super::{BitReq, Cmp, ExecutionId, GroupId, TagGuard, TagMode, apply_branch_config, axis_toggle_pattern};
    use crate::tensor::Tensor;

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

    /// `TagMode::Comparison` puts comparison `i` in bit `i` -- the order every kernel's [`TagGuard`]
    /// pattern is written against, and that the ViSA translator and the hardware's comparator registers
    /// agree on.
    ///
    /// The expected ids are hand-computed; reimplementing the bit loop here would pass whichever way the
    /// shift went.
    #[test]
    fn comparison_bits_land_in_cmp_order() {
        axes![A = 3];
        let cmps = [Cmp::Equal(0.0f32), Cmp::Greater(0.0), Cmp::Less(0.0), Cmp::True];
        let data = Tensor::<f32, m![A]>::from_vec([0.0, 1.5, -2.5]);

        let tags = apply_branch_config(&data, &TagMode::Comparison(cmps));

        // bit3 is `True` throughout, so every id has it set.
        assert_eq!(
            tags.into_vec(),
            vec![
                0b1001, // zero: equal + true
                0b1010, // positive: greater + true
                0b1100, // negative: less + true
            ]
        );
    }

    /// The next hop of the same convention: a guard written for comparison `i` selects exactly the cells
    /// that comparison set. `comparison_bits_land_in_cmp_order` pins `cmp[i] -> bit i`, this pins
    /// `bit i -> guard bit i`.
    ///
    /// Both halves are hand-computed, so reversing the bit order breaks this test even though
    /// `TagGuard::matches` and `apply_branch_config` would still agree with each other.
    #[test]
    fn a_guard_selects_the_cells_its_comparison_matched() {
        axes![A = 4];
        // Bit 0 is `Equal`, bit 1 `Greater`, bit 2 `Less` -- the tag mode the branch-unit example uses.
        let cmps = [Cmp::Equal(0.0f32), Cmp::Greater(0.0), Cmp::Less(0.0), Cmp::True];
        let data = Tensor::<f32, m![A]>::from_vec([0.0, 1.5, -2.5, -0.0]);
        let tags = apply_branch_config(&data, &TagMode::Comparison(cmps)).into_vec();

        let selects = |guard: TagGuard| {
            tags.iter()
                .map(|id| guard.admits(ExecutionId::try_new(*id).expect("the tag unit writes four bits")))
                .collect::<Vec<_>>()
        };

        let negative = TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::One, BitReq::Ignore]);
        assert_eq!(selects(negative), vec![false, false, true, false], "the `Less` cell");

        let zero = TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
        assert_eq!(
            selects(zero),
            vec![true, false, false, true],
            "both zeros, `Equal` being `==`"
        );

        // Bit 3 is the group bit as well as this tag mode's `True`, so a group guard reads it here.
        assert_eq!(
            selects(TagGuard::group(GroupId::One)),
            vec![true; 4],
            "`True` sets bit 3 for all"
        );
        assert_eq!(selects(TagGuard::group(GroupId::Zero)), vec![false; 4]);
    }

    /// `Equal` on floats is the scalar's own `==`, so it does not distinguish the two zeros. Worth
    /// pinning because a kernel branching on "is zero" relies on `-0.0` taking the same branch.
    #[test]
    fn equal_on_f32_treats_negative_zero_as_zero() {
        assert!(Cmp::Equal(0.0f32).matches(-0.0));
        assert!(Cmp::Equal(-0.0f32).matches(0.0));
    }

    /// The `*Unsigned` pair compares raw bit patterns, not values, so a negative reads as large:
    /// `-1.0f32` is `0xbf80_0000` against `1.0f32`'s `0x3f80_0000`, and `-1i32` is `0xffff_ffff`. This
    /// is the whole reason they are separate variants rather than `Less`/`Greater` on another scalar,
    /// and signed and unsigned disagree on exactly the negatives. Both scalars `Cmp<D>` is used at.
    #[test]
    fn unsigned_compares_read_the_bit_pattern_not_the_value() {
        assert!(Cmp::GreaterUnsigned(1.0f32).matches(-1.0));
        assert!(!Cmp::Greater(1.0f32).matches(-1.0));
        assert!(Cmp::LessUnsigned(-1.0f32).matches(1.0));

        assert!(Cmp::GreaterUnsigned(1i32).matches(-1));
        assert!(!Cmp::Greater(1i32).matches(-1));
        assert!(!Cmp::LessUnsigned(0i32).matches(-1), "nothing is below zero unsigned");
        assert!(Cmp::Less(0i32).matches(-1), "but -1 is below zero signed");
    }
}
