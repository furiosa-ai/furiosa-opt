#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]
#![feature(register_tool)]
// `#[primitive(..)]` expands to a `#[furiosa_opt::primitive = ".."]` tool attribute, which the ViSA
// translator reads off the item's `DefId`. The namespace has to be registered wherever such an item
// is defined, so it is registered here as well as in `furiosa-opt-std`.
#![register_tool(furiosa_opt)]

//! The VE branch vocabulary the eDSL and the compiler IR both speak: the guard types and the slot
//! layout, so the compiler side can name them without depending on the eDSL's standard library.
//!
//! Only the vocabulary. Each side's own payloads stay where they are used.

use std::fmt::{self, Display, Formatter};

use furiosa_opt_macro::primitive;
use serde::{Deserialize, Serialize};

/// One of the two groups Filter and pair mode split elements into: bit 3 of the execution id.
///
/// A kernel names it to write [`TagGuard::group`], which is the supported way to gate a slot on a
/// group without spelling the bit pattern out.
#[primitive(ve::GroupId)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, serde_lite::Deserialize)]
pub enum GroupId {
    /// Group 0
    Zero,
    /// Group 1
    One,
}

impl Display for GroupId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "GroupId::Zero"),
            Self::One => write!(f, "GroupId::One"),
        }
    }
}

/// One element's execution id: four bits, so sixteen values.
///
/// A newtype because [`TagGuard::admits`] reads the low four bits and would otherwise silently alias
/// a larger `u8` onto them -- `0x10` answering as `0x00`. Construction is the one place that can go
/// wrong, so it is the one place that checks.
// No `Ord`: an execution id is four independent bits, so ordering them would invite range comparisons
// that mean nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionId(u8);

impl ExecutionId {
    const COUNT: u8 = 16;

    /// The id `RAW` names. Out of range is a compile error, so a literal carries no fallback.
    ///
    /// The bound is a `const` block rather than a `const fn` returning `Option`, because a `const fn`
    /// runs at compile time only where the *caller* puts it in a const context: as an ordinary call it
    /// is an ordinary call. Taking the value as a const parameter moves the check to the definition,
    /// which is the shape `furiosa-opt-std`'s `constraints` module uses for the same reason.
    pub const fn new<const RAW: u8>() -> Self {
        const { assert!(RAW < Self::COUNT, "an execution id is four bits, so 0..=15") };
        Self(RAW)
    }

    /// The id `raw` names, or `None` if it does not fit in four bits.
    ///
    /// For a value that is only known at run time -- a byte read out of a tag tensor. A literal should
    /// use [`new`](Self::new), which rejects an out-of-range one at compile time.
    pub const fn try_new(raw: u8) -> Option<Self> {
        if raw < Self::COUNT { Some(Self(raw)) } else { None }
    }

    /// Every execution id, in order: what a claim about guards is checked against.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).map(Self)
    }

    const fn bit(self, bit: usize) -> bool {
        self.0 & (1 << bit) != 0
    }
}

impl Display for ExecutionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:#06b}", self.0)
    }
}

/// Requirement on a single execution-id bit. [`Ignore`](BitReq::Ignore) is the default: the bit is not
/// constrained, so a pattern of four `Ignore`s matches every initialized execution id.
///
/// The requirement is on the bit's *value*, not on a comparison having held: which comparison filled a
/// bit is the tag mode's business, and a mode need not be comparing anything -- `AxisToggle` sets bit 3
/// from the axis index, and bit 3 is the group either way. `Cmp` is where a comparison is named.
#[primitive(ve::BitReq)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, serde_lite::Deserialize)]
pub enum BitReq {
    /// The bit must be 1.
    One,
    /// The bit must be 0.
    Zero,
    /// The bit is unconstrained.
    #[default]
    Ignore,
}

impl Display for BitReq {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "BitReq::{self:?}")
    }
}

/// Predicate gating one operand slot on the element's execution id.
///
/// A pattern is one [`BitReq`] per execution-id bit, in bit order: `[bit0, bit1, bit2, bit3]`, the
/// same order as the comparison array that produced the id (a `TagMode::Comparison` sets bit `i` from
/// comparison `i`). Bit 3 doubles as the [`GroupId`].
///
/// Opaque, and canonical because of it: the spellings that mean the same thing are folded on the way
/// in, so two guards are equal exactly when they admit the same execution ids and nothing downstream
/// has to normalize. Read the shape back with [`pattern`](Self::pattern) and
/// [`as_group`](Self::as_group).
///
/// A wire payload is a way in like any other, so deserialization folds too, through the [`From`] impl
/// below rather than by wrapping what it read.
#[primitive(ve::TagGuard)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "Repr", into = "Repr")]
pub struct TagGuard(Repr);

/// `TagGuard`'s two shapes: a pattern, checked as written or negated. Private: the invariant is that
/// every value came through a constructor, and a public variant would be a way around them.
///
/// There is no unconditional shape of its own, because a pattern constraining no bit already is one:
/// `Matches([Ignore; 4])` admits every execution id. Two ways to spell one predicate would be two
/// values to compare, which is what [`all`](TagGuard::all) returning this shape avoids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, serde_lite::Deserialize)]
enum Repr {
    /// Applies where the execution id satisfies the pattern.
    Matches([BitReq; 4]),
    /// Applies where the execution id does *not* satisfy the pattern.
    NotMatches([BitReq; 4]),
}

/// The unconditional guard, which is what an unconstrained pattern is.
impl Default for TagGuard {
    fn default() -> Self {
        Self::all()
    }
}

/// The guard a `Repr` spells, folded like a constructed one.
///
/// A `Repr` read off the wire is a pattern and a polarity, which is what a constructor takes, so it
/// goes through the constructors: they own the folding, and a guard that skipped it would be a second
/// value for a predicate that already has one -- `NotMatches([Ignore, Ignore, Ignore, One])` sitting
/// beside the `group(Zero)` it means, unrecognised by [`as_group`](TagGuard::as_group) and lowered as
/// a comparison-match instead of the compact group form.
impl From<Repr> for TagGuard {
    fn from(repr: Repr) -> Self {
        match repr {
            Repr::Matches(bits) => Self::matches(bits),
            Repr::NotMatches(bits) => Self::not_matches(bits),
        }
    }
}

/// The way back out, which needs no folding: a guard holds a folded `Repr` already.
impl From<TagGuard> for Repr {
    fn from(guard: TagGuard) -> Self {
        guard.0
    }
}

// Delegated rather than derived so the wire shape stays the enum's: a guard is written as
// `{ "Matches": [..] }`, before and after the newtype.
impl serde_lite::Deserialize for TagGuard {
    fn deserialize(val: &serde_lite::Intermediate) -> Result<Self, serde_lite::Error> {
        <Repr as serde_lite::Deserialize>::deserialize(val).map(Self::from)
    }
}

/// Whether a pattern constrains no bit at all, which is what makes it the unconditional one.
const fn is_unconstrained(bits: &[BitReq; 4]) -> bool {
    matches!(bits, [BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore])
}

impl TagGuard {
    /// Applies to every element with an initialized execution id: the pass's `else`.
    ///
    /// The pattern that constrains no bit, named. `matches([Ignore; 4])` is not merely equal to this,
    /// it is this.
    #[primitive(ve::TagGuard::all)]
    pub const fn all() -> Self {
        Self(Repr::Matches([BitReq::Ignore; 4]))
    }

    /// Applies where the execution id satisfies `bits`.
    #[primitive(ve::TagGuard::matches)]
    pub const fn matches(bits: [BitReq; 4]) -> Self {
        Self(Repr::Matches(bits))
    }

    /// Applies where the execution id does *not* satisfy `bits`.
    ///
    /// Bit 3 has two values, so negating a pattern that constrains it alone names the other group, and is
    /// folded to the [`group`](Self::group) that builds it so lowering recognises either spelling.
    /// Negating an unconstrained pattern is kept as it is: it admits no execution id, and the layout
    /// checks report that dead slot against the kernel's own span.
    #[primitive(ve::TagGuard::not_matches)]
    pub const fn not_matches(bits: [BitReq; 4]) -> Self {
        match bits {
            [BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::One] => Self::group(GroupId::Zero),
            [BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::Zero] => Self::group(GroupId::One),
            bits => Self(Repr::NotMatches(bits)),
        }
    }

    /// Applies to one [`GroupId`]: bit 3 must equal the group bit, bits 0-2 unconstrained.
    #[primitive(ve::TagGuard::group)]
    pub const fn group(id: GroupId) -> Self {
        // Matched exhaustively rather than compared against a bit value, so a new `GroupId` variant
        // is a compile error here instead of silently folding to `Zero`.
        let bit3 = match id {
            GroupId::Zero => BitReq::Zero,
            GroupId::One => BitReq::One,
        };
        Self(Repr::Matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, bit3]))
    }

    /// The pattern this guard checks and whether it inverts the answer, or `None` for
    /// [`all`](Self::all), which checks nothing.
    ///
    /// The read side of the constructors: a pattern plus a polarity is what the hardware's
    /// comparison-match form takes, so this is enough to encode without exposing a buildable shape.
    pub const fn pattern(&self) -> Option<(&[BitReq; 4], bool)> {
        match &self.0 {
            // A pattern constraining nothing checks nothing, which is what makes this `None` rather
            // than a pattern the caller would have to recognise as vacuous.
            Repr::Matches(bits) if is_unconstrained(bits) => None,
            Repr::Matches(bits) => Some((bits, false)),
            Repr::NotMatches(bits) => Some((bits, true)),
        }
    }

    /// Whether `exec_id` satisfies this guard: each constrained bit must match, and a negated pattern
    /// inverts the answer.
    #[inline]
    pub fn admits(self, exec_id: ExecutionId) -> bool {
        let Some((pattern, negate)) = self.pattern() else {
            return true;
        };
        let satisfied = pattern.iter().enumerate().all(|(bit, req)| match req {
            BitReq::One => exec_id.bit(bit),
            BitReq::Zero => !exec_id.bit(bit),
            BitReq::Ignore => true,
        });
        satisfied != negate
    }

    /// Whether no execution id satisfies this guard at all, which makes the slot dead: it is filled,
    /// it costs a slot, and it can never fire.
    ///
    /// Exactly one shape does this: a positive pattern always admits *some* id, so the only guard
    /// admitting none is the negation of a pattern that matches everything.
    fn matches_nothing(&self) -> bool {
        matches!(self.pattern(), Some((bits, true)) if is_unconstrained(bits))
    }

    /// Whether this guard applies to every element with an initialized execution id, which makes the
    /// slot a pass's last.
    ///
    /// Exactly [`all`](Self::all) does, which is the unconstrained pattern, which is the one
    /// [`pattern`](Self::pattern) reports as checking nothing.
    fn is_unconditional(&self) -> bool {
        self.pattern().is_none()
    }

    /// The [`GroupId`] this guard selects, when it constrains *only* the group bit -- the shape
    /// [`group`](Self::group) builds, and the one [`not_matches`](Self::not_matches) folds its
    /// negation into.
    /// Lowering uses this to keep a group-scoped guard in the hardware's compact group form instead of
    /// a comparison-match pattern.
    pub fn as_group(&self) -> Option<GroupId> {
        let Some(([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, bit3], false)) = self.pattern() else {
            return None;
        };
        match bit3 {
            BitReq::One => Some(GroupId::One),
            BitReq::Zero => Some(GroupId::Zero),
            BitReq::Ignore => None,
        }
    }
}

impl Display for TagGuard {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let pattern = |f: &mut Formatter<'_>, [bit0, bit1, bit2, bit3]: &[BitReq; 4]| -> fmt::Result {
            write!(f, "[{bit0}, {bit1}, {bit2}, {bit3}]")
        };
        // Prints the constructor that builds it, so a diagnostic quoting a guard reads back as code.
        match self.pattern() {
            None => write!(f, "TagGuard::all()"),
            Some((bits, negated)) => {
                let name = if negated { "not_matches" } else { "matches" };
                write!(f, "TagGuard::{name}(")?;
                pattern(f, bits)?;
                write!(f, ")")
            }
        }
    }
}

/// The rhs slots one VE pass drives: three immediate registers and the register-file port, each
/// `None` while unused and gated by its own [`TagGuard`].
///
/// Slot order is match priority, not hardware register numbering: an element takes the first filled
/// slot whose guard it satisfies -- first match, not every match, which the ISA specifies and the
/// hardware, the host engine and the LIR executor all implement. So overlapping guards are resolved by
/// slot order rather than accumulated, and an unconditional slot is the pass's `else`, with nothing
/// after it. Command generation assigns the physical registers from the branch list this lowers to.
///
/// The order is enforced twice: the eDSL builder structurally, so a slot cannot be filled twice or out
/// of order, and the ViSA translator again as a diagnostic, because `#[primitive]` interception drops
/// the builder's typing on the way into MIR.
///
/// `Reg` and `Port` are what each layer puts in the slots: the eDSL tensors and scalars, the IR its
/// own encodable payloads.
///
/// The slots are private and this type is not in the eDSL's prelude. A kernel reaches a layout only
/// through the eDSL builder; a compiler stage goes through [`try_from_slots`](Self::try_from_slots) or
/// one of the fills, and a wire payload through [`try_from_slots`](Self::try_from_slots) as well --
/// each of which validates.
#[primitive(ve::BranchedOperand)]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BranchedOperand<Reg, Port> {
    reg0: Option<(TagGuard, Reg)>,
    reg1: Option<(TagGuard, Reg)>,
    reg2: Option<(TagGuard, Reg)>,
    /// The register-file port: an external register read or the stash read.
    rf: Option<(TagGuard, Port)>,
}

// Hand-written: `derive(Default)` would demand `Default` of both payloads, but an empty layout needs
// neither -- every slot is simply `None`.
impl<Reg, Port> Default for BranchedOperand<Reg, Port> {
    fn default() -> Self {
        Self {
            reg0: None,
            reg1: None,
            reg2: None,
            rf: None,
        }
    }
}

/// Which of a pass's three immediate registers a slot is.
///
/// Named rather than a `usize` so there is no out-of-range case to report or to panic on: a pass has
/// exactly these three, and command generation picks the physical register behind each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmSlot {
    /// The first immediate register.
    Reg0,
    /// The second.
    Reg1,
    /// The third.
    Reg2,
}

impl ImmSlot {
    const ALL: [Self; 3] = [Self::Reg0, Self::Reg1, Self::Reg2];
}

impl Display for ImmSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Reg0 => "reg0",
            Self::Reg1 => "reg1",
            Self::Reg2 => "reg2",
        };
        f.write_str(name)
    }
}

/// One of a pass's four operand slots: the three immediate registers, then the register-file port.
///
/// The order is application order, which is match priority -- see [`BranchedOperand`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// An immediate register.
    Imm(ImmSlot),
    /// The register-file port: an external register read or the stash read. Last, and at most one.
    Rf,
}

impl Slot {
    /// Positionally the same as [`slot_guards`](BranchedOperand::slot_guards).
    const ALL: [Self; 4] = [
        Self::Imm(ImmSlot::Reg0),
        Self::Imm(ImmSlot::Reg1),
        Self::Imm(ImmSlot::Reg2),
        Self::Rf,
    ];
}

impl Display for Slot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Imm(imm) => write!(f, "{imm}"),
            Self::Rf => f.write_str("rf"),
        }
    }
}

impl<Reg, Port> BranchedOperand<Reg, Port> {
    /// A layout from its slots, unchecked.
    fn from_slots(regs: [Option<(TagGuard, Reg)>; 3], rf: Option<(TagGuard, Port)>) -> Self {
        let [reg0, reg1, reg2] = regs;
        Self { reg0, reg1, reg2, rf }
    }

    /// Fills immediate register `slot`, and validates what that produced.
    ///
    /// The incremental counterpart of [`try_from_slots`](Self::try_from_slots), for a caller that has one
    /// slot at a time. Validating after the write is what lets the rules be about a whole layout; see
    /// [`validate`](Self::validate). A refused fill leaves the layout untouched.
    pub fn fill_imm(&mut self, slot: ImmSlot, guard: TagGuard, reg: Reg) -> Result<(), SlotDefect> {
        if self.imm_slot_mut(slot).is_some() {
            return Err(SlotDefect::Occupied { slot: Slot::Imm(slot) });
        }
        *self.imm_slot_mut(slot) = Some((guard, reg));
        self.validate().inspect_err(|_| *self.imm_slot_mut(slot) = None)
    }

    /// Fills the next immediate register that is free, and validates what that produced.
    ///
    /// Which one that is is not observable -- command generation allocates the physical registers --
    /// so a caller that does not care where its value lands says so by using this.
    pub fn fill_next_imm(&mut self, guard: TagGuard, reg: Reg) -> Result<(), SlotDefect> {
        let free = ImmSlot::ALL
            .into_iter()
            .zip(self.reg_slots())
            .find_map(|(slot, cell)| cell.is_none().then_some(slot))
            .ok_or(SlotDefect::NoFreeImmediate)?;
        self.fill_imm(free, guard, reg)
    }

    /// Fills the register-file port, and validates what that produced.
    pub fn fill_rf(&mut self, guard: TagGuard, rf: Port) -> Result<(), SlotDefect> {
        if self.rf.is_some() {
            return Err(SlotDefect::Occupied { slot: Slot::Rf });
        }
        self.rf = Some((guard, rf));
        self.validate().inspect_err(|_| self.rf = None)
    }

    /// The three immediate register slots in application order.
    pub fn reg_slots(&self) -> [&Option<(TagGuard, Reg)>; 3] {
        [&self.reg0, &self.reg1, &self.reg2]
    }

    fn imm_slot_mut(&mut self, slot: ImmSlot) -> &mut Option<(TagGuard, Reg)> {
        match slot {
            ImmSlot::Reg0 => &mut self.reg0,
            ImmSlot::Reg1 => &mut self.reg1,
            ImmSlot::Reg2 => &mut self.reg2,
        }
    }

    /// The register-file port slot: the pass's one external register or stash read.
    pub fn rf_slot(&self) -> &Option<(TagGuard, Port)> {
        &self.rf
    }

    /// Each slot's guard in application order, `None` where the slot is empty. Lets a caller reason
    /// about which elements the filled slots claim without caring what those slots carry.
    pub fn slot_guards(&self) -> [Option<&TagGuard>; 4] {
        let [reg0, reg1, reg2] = self.reg_slots().map(|slot| slot.as_ref().map(|(guard, _)| guard));
        [reg0, reg1, reg2, self.rf.as_ref().map(|(guard, _)| guard)]
    }

    /// The first way this layout is ill-formed, or `Ok(())`.
    ///
    /// One place for the rules, because three have to apply them and cannot share a way of *reporting*:
    /// the eDSL builder panics, the ViSA translator reports against the kernel's span, and lowering to
    /// `VePass` returns an `eyre` error. Each wraps this [`SlotDefect`] in its own idiom.
    ///
    /// Checked in slot order, so the defect reported is the earliest a reader would find.
    pub fn validate(&self) -> Result<(), SlotDefect> {
        let guards = self.slot_guards();
        if guards.iter().all(Option::is_none) {
            return Err(SlotDefect::Empty);
        }
        let mut unconditional = None;
        for (slot, guard) in Slot::ALL.into_iter().zip(guards) {
            let Some(guard) = guard else { continue };
            if guard.matches_nothing() {
                return Err(SlotDefect::DeadGuard { slot: Some(slot) });
            }
            if let Some(unconditional) = unconditional {
                return Err(SlotDefect::AfterUnconditional { unconditional, slot });
            }
            if guard.is_unconditional() {
                unconditional = Some(slot);
            }
        }
        Ok(())
    }

    /// A validated layout from its slots, for a compiler stage that has all four in hand at once.
    ///
    /// The checked counterpart of [`Default`] plus the slot accessors, which is how the eDSL builder
    /// gets there instead -- one slot at a time, validating as it goes.
    pub fn try_from_slots(
        regs: [Option<(TagGuard, Reg)>; 3],
        rf: Option<(TagGuard, Port)>,
    ) -> Result<Self, SlotDefect> {
        let layout = Self::from_slots(regs, rf);
        layout.validate()?;
        Ok(layout)
    }
}

impl<Reg, Port> BranchedOperand<Reg, Port> {
    /// Prints only the filled slots, each payload rendered by the caller. A ternary slot's payload is
    /// a plain pair, which has no `Display` of its own, so the two impls below differ in the
    /// formatter they pass and nothing else.
    fn fmt_slots(
        &self,
        f: &mut Formatter<'_>,
        reg: impl Fn(&mut Formatter<'_>, &Reg) -> fmt::Result,
        rf: impl Fn(&mut Formatter<'_>, &Port) -> fmt::Result,
    ) -> fmt::Result {
        write!(f, "BranchedOperand {{ ")?;
        let mut first = true;
        let mut separate = |f: &mut Formatter<'_>| -> fmt::Result {
            if !std::mem::take(&mut first) {
                write!(f, ", ")?;
            }
            Ok(())
        };
        for (slot, name) in self.reg_slots().into_iter().zip(ImmSlot::ALL) {
            if let Some((guard, payload)) = slot {
                separate(f)?;
                write!(f, "{name}: ({guard}, ")?;
                reg(f, payload)?;
                write!(f, ")")?;
            }
        }
        if let Some((guard, payload)) = &self.rf {
            separate(f)?;
            write!(f, "rf: ({guard}, ")?;
            rf(f, payload)?;
            write!(f, ")")?;
        }
        write!(f, " }}")
    }
}

/// What makes a [`BranchedOperand`] ill-formed.
///
/// [`Display`] renders the defect and why it is one; a caller adds its own context -- a kernel span,
/// the node being lowered -- around that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SlotDefect {
    /// No slot is filled, so the pass would spend an ALU and a pipeline stage driving nothing.
    #[error(
        "this operand fills no slot, so the pass would spend an ALU and a pipeline stage driving \
         nothing"
    )]
    Empty,
    /// A guard admits no execution id, so what it gates can never fire. `slot` names the slot, or is
    /// `None` for a unary node, whose guard gates the node itself.
    // The two cases read as different sentences rather than one with a hole in it, which no format
    // string spells, so this variant formats itself.
    #[error(fmt = fmt_dead_guard)]
    DeadGuard {
        /// The dead slot, or `None` for a unary node.
        slot: Option<Slot>,
    },
    /// A slot already holds a payload, so filling it again would drop one.
    #[error("slot {slot} is already filled, and filling it again would drop what it holds")]
    Occupied {
        /// The slot asked for.
        slot: Slot,
    },
    /// Every immediate register is spoken for; a pass has three.
    #[error("this pass has no immediate register left: it has three, and they are all spoken for")]
    NoFreeImmediate,
    /// A slot follows an unconditional one, which already claims every element left over.
    #[error(
        "slot {unconditional} is unconditional, which claims every element the earlier guards left \
         over, so no slot may follow it; slot {slot} could never fire. Fill it before the \
         unconditional one, or guard the unconditional one"
    )]
    AfterUnconditional {
        /// The unconditional slot.
        unconditional: Slot,
        /// The slot after it, which can never fire.
        slot: Slot,
    },
}

/// A dead guard on a slot costs a slot that could have fired, which is worth saying; a dead guard on
/// a unary node costs only itself, which is not.
fn fmt_dead_guard(slot: &Option<Slot>, f: &mut Formatter<'_>) -> fmt::Result {
    let (what, advice) = match slot {
        Some(slot) => (
            format!("slot {slot}"),
            ". A pass has only three immediate registers and one rf port; spend them on slots that \
             can fire",
        ),
        None => ("this op".to_owned(), ""),
    };
    write!(
        f,
        "{what}'s guard is satisfied by no execution id at all, so it would never fire{advice}"
    )
}

/// Prints only the filled slots, so an operand reads as the slots the pass actually drives.
impl<Reg: Display, Port: Display> Display for BranchedOperand<Reg, Port> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_slots(f, |f, reg| write!(f, "{reg}"), |f, rf| write!(f, "{rf}"))
    }
}

/// Written out rather than derived, because a layout's rules are about the four slots together: they
/// cannot be applied field by field as the fields are read, and the derive offers nothing after. So
/// the derive reads the slots into the shape below -- private to this function, the wire's shape and
/// not a layout -- and [`try_from_slots`](Self::try_from_slots) turns those into a layout or names the
/// defect, the same door a compiler stage with four slots in hand goes through.
impl<'de, Reg, Port> Deserialize<'de> for BranchedOperand<Reg, Port>
where
    Reg: Deserialize<'de>,
    Port: Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename = "BranchedOperand")]
        struct Slots<Reg, Port> {
            reg0: Option<(TagGuard, Reg)>,
            reg1: Option<(TagGuard, Reg)>,
            reg2: Option<(TagGuard, Reg)>,
            rf: Option<(TagGuard, Port)>,
        }

        let Slots { reg0, reg1, reg2, rf } = Slots::deserialize(deserializer)?;
        Self::try_from_slots([reg0, reg1, reg2], rf).map_err(serde::de::Error::custom)
    }
}

/// Written out rather than derived, and the same goes for [`Operands`] below: the derive puts no bound
/// on a type's generic parameters, so what it generates does not compile against the published
/// `serde-lite`. This workspace patches in a fork that adds them; a project from `base-template` --
/// the build a kernel author runs -- resolves the published one and does not.
///
/// Reads what the derive read: the four slots as named fields, all four attempted before any error is
/// returned, so a bad payload names every slot it could not read rather than only the first. What the
/// four make is then checked, by the constructor rather than here, exactly as on the `serde` side.
impl<Reg, Port> serde_lite::Deserialize for BranchedOperand<Reg, Port>
where
    Reg: serde_lite::Deserialize,
    Port: serde_lite::Deserialize,
{
    fn deserialize(val: &serde_lite::Intermediate) -> Result<Self, serde_lite::Error> {
        let obj = val
            .as_map()
            .ok_or_else(|| serde_lite::Error::invalid_value_static("object"))?;
        let mut errors = serde_lite::ErrorList::new();

        let reg0 = read_slot(obj, "reg0", &mut errors);
        let reg1 = read_slot(obj, "reg1", &mut errors);
        let reg2 = read_slot(obj, "reg2", &mut errors);
        let rf = read_slot(obj, "rf", &mut errors);

        match (reg0, reg1, reg2, rf) {
            (Some(reg0), Some(reg1), Some(reg2), Some(rf)) => {
                Self::try_from_slots([reg0, reg1, reg2], rf).map_err(serde_lite::Error::custom)
            }
            _ => Err(serde_lite::Error::NamedFieldErrors(errors)),
        }
    }
}

/// One slot of a [`BranchedOperand`] being deserialized. A slot that is missing or unreadable leaves
/// its reason in `errors` under the slot's name and yields `None`, which is what lets the caller read
/// all four before deciding.
fn read_slot<T: serde_lite::Deserialize>(
    obj: &serde_lite::Map,
    name: &'static str,
    errors: &mut serde_lite::ErrorList<serde_lite::NamedFieldError>,
) -> Option<Option<(TagGuard, T)>> {
    let slot = obj.get(name).map_or_else(
        || Err(serde_lite::Error::MissingField),
        <Option<(TagGuard, T)> as serde_lite::Deserialize>::deserialize,
    );

    match slot {
        Ok(slot) => Some(slot),
        Err(err) => {
            errors.push(serde_lite::NamedFieldError::new_static(name, err));
            None
        }
    }
}

/// What one VE node's operands are, with the node's arity as the variant so no slot has to carry an
/// `Option` that is really an arity tag. A unary node has no rhs at all, just the guard naming where
/// it runs.
///
/// `Reg` and `Port` are generic because each layer has its own payloads. The ternary `operand1` is
/// deliberately not: it is `f32` everywhere, being the only form the float cluster's ternary ops
/// encode, so widening it would buy a type parameter and a rejection path further down and nothing
/// else.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operands<Reg, Port> {
    /// A binary node: guarded slots, each carrying one value or the port read.
    Binary(BranchedOperand<Reg, Port>),
    /// A ternary node: the same slots, each also carrying its branch's `operand1`.
    /// second operand is fixed to f32, since only float cluster has ternary alu.
    Ternary(BranchedOperand<(Reg, f32), (Port, f32)>),
    /// A unary node: no rhs, just the guard saying where it runs.
    ///
    /// Reachable from a kernel: `vector_fp_unary` takes its op bare to run everywhere, or paired with
    /// a guard to run where that guard matches. The guard is arbitrary here and there too.
    // The slot variants are checked as they are read, being `BranchedOperand`s, which leaves this one:
    // a unary node has no slots, so no slot rule covers the guard that gates the node itself.
    Unary(#[serde(deserialize_with = "read_live_guard")] TagGuard),
}

/// The guard of a unary node being deserialized, refused where it is dead.
///
/// The rule is [`Operands::validate`]'s, which is why it is asked rather than restated: a guard that
/// admits no execution id makes the node it gates unreachable.
fn read_live_guard<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<TagGuard, D::Error> {
    let guard = TagGuard::deserialize(deserializer)?;
    // The payload types play no part in the unary rule, so any pair of them asks it.
    Operands::<(), ()>::Unary(guard)
        .validate()
        .map_err(serde::de::Error::custom)?;
    Ok(guard)
}

impl<Reg, Port> Operands<Reg, Port> {
    /// The first way this node's operands are ill-formed, or `Ok(())`.
    ///
    /// Delegates to [`BranchedOperand::validate`], so a node is checked by the same rules whatever its
    /// arity. A unary node has no slots -- the guard it carries names where the node runs -- so there is
    /// nothing about slots to check.
    pub fn validate(&self) -> Result<(), SlotDefect> {
        match self {
            Self::Binary(slots) => slots.validate(),
            Self::Ternary(slots) => slots.validate(),
            // A unary node has no slots, so it is checked here rather than in `BranchedOperand`.
            Self::Unary(guard) if guard.matches_nothing() => Err(SlotDefect::DeadGuard { slot: None }),
            Self::Unary(_) => Ok(()),
        }
    }

    /// Each slot's guard in slot order, `None` where the slot is empty, whichever arity the node is.
    ///
    /// A unary node has no slots at all -- the guard it carries names where the node runs, not a slot
    /// -- so it reports four empty slots rather than trying to pass that guard off as one.
    pub fn slot_guards(&self) -> [Option<&TagGuard>; 4] {
        match self {
            Self::Binary(slots) => slots.slot_guards(),
            Self::Ternary(slots) => slots.slot_guards(),
            Self::Unary(_) => [None; 4],
        }
    }
}

impl<Reg: Display, Port: Display> Display for Operands<Reg, Port> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary(slots) => write!(f, "{slots}"),
            Self::Ternary(slots) => slots.fmt_slots(
                f,
                |f, (operand0, operand1)| write!(f, "{operand0}, operand1: {operand1}"),
                |f, (port, operand1)| write!(f, "{port}, operand1: {operand1}"),
            ),
            Self::Unary(guard) => write!(f, "unary @ {guard}"),
        }
    }
}

/// Written out for the reason [`BranchedOperand`]'s impl above gives. Reads the externally tagged
/// form the derive read: a one-key map naming the variant. The bare variant name carries no operands,
/// and every variant here has some, so that form is an error whichever name it gives.
///
/// What it read is then validated. The slot variants have checked themselves already, so what this
/// adds is the unary node's guard -- the same gap [`read_live_guard`] fills on the `serde` side, asked
/// as the whole rule here because one call covers it.
impl<Reg, Port> serde_lite::Deserialize for Operands<Reg, Port>
where
    Reg: serde_lite::Deserialize,
    Port: serde_lite::Deserialize,
{
    fn deserialize(val: &serde_lite::Intermediate) -> Result<Self, serde_lite::Error> {
        let operands = if let Some(obj) = val.as_map() {
            if let Some(content) = obj.get("Binary") {
                serde_lite::Deserialize::deserialize(content).map(Self::Binary)
            } else if let Some(content) = obj.get("Ternary") {
                serde_lite::Deserialize::deserialize(content).map(Self::Ternary)
            } else if let Some(content) = obj.get("Unary") {
                serde_lite::Deserialize::deserialize(content).map(Self::Unary)
            } else {
                Err(serde_lite::Error::UnknownEnumVariant)
            }
        } else if let Some(variant) = val.as_str() {
            match variant {
                "Binary" | "Ternary" | "Unary" => Err(serde_lite::Error::MissingEnumVariantContent),
                _ => Err(serde_lite::Error::UnknownEnumVariant),
            }
        } else {
            Err(serde_lite::Error::invalid_value_static("enum variant"))
        }?;

        operands.validate().map_err(serde_lite::Error::custom)?;
        Ok(operands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every execution id 0..16, checked against the mask compare written out by hand. Which bit each
    /// [`BitReq`] constrains is the claim worth pinning: an off-by-one in the bit positions would still
    /// look plausible in isolation. Bit 3 is covered too, being the group bit as well.
    #[test]
    fn a_pattern_constrains_the_bits_in_order() {
        let pattern = [BitReq::One, BitReq::Zero, BitReq::Ignore, BitReq::One];
        let (yes, no) = (TagGuard::matches(pattern), TagGuard::not_matches(pattern));
        let group1 = TagGuard::group(GroupId::One);
        for id in ExecutionId::all() {
            // Against the mask written out by hand, not against each other: comparing the two forms
            // would pass with the bit positions wrong, since one is the other's negation by
            // construction.
            let expected = id.bit(0) && !id.bit(1) && id.bit(3);
            assert_eq!(yes.admits(id), expected, "matches @ {id}");
            assert_eq!(no.admits(id), !expected, "not_matches @ {id}");
            // A group is bit 3 and nothing else, which is why `group` and the pattern spelling of it
            // are one value.
            assert_eq!(group1.admits(id), id.bit(3), "group @ {id}");
        }
        assert_eq!(
            group1,
            TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::One])
        );

        // The read side carries the same order, which is what lowering maps onto the hardware's
        // comparison-match form one bit at a time. Asserted here because that mapping lives a crate up
        // -- `CmpMatches` is `tu-ops`' -- so this is the last place the order is this crate's to state.
        assert_eq!(yes.pattern(), Some((&pattern, false)), "`is` reads back in bit order");
        assert_eq!(
            no.pattern(),
            Some((&pattern, true)),
            "`not_matches` is the same pattern, negated"
        );
    }

    /// The constructors fold, so guards that admit the same execution ids *are* the same value and no
    /// caller has to normalize. Two spellings fold: an unconstrained positive pattern is
    /// [`TagGuard::all`], and a negation constraining the group bit alone is the other group.
    ///
    /// Checked against the predicate itself -- equal ids admitted, over all sixteen -- rather than
    /// against a second copy of the folding rules.
    #[test]
    fn equal_predicates_are_equal_values() {
        const ANY: [BitReq; 4] = [BitReq::Ignore; 4];
        let folded = [
            (TagGuard::matches(ANY), TagGuard::all()),
            (
                TagGuard::not_matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::One]),
                TagGuard::group(GroupId::Zero),
            ),
            (
                TagGuard::not_matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::Zero]),
                TagGuard::group(GroupId::One),
            ),
        ];
        for (written, canonical) in folded {
            assert_eq!(written, canonical, "{written} should fold to {canonical}");
            for id in ExecutionId::all() {
                assert_eq!(written.admits(id), canonical.admits(id), "@ {id}");
            }
        }

        // The one negation with nothing to fold into: no execution id satisfies it, and it is kept as
        // it is so the layout checks can report the dead slot.
        assert_ne!(TagGuard::not_matches(ANY), TagGuard::all());
        assert!(TagGuard::not_matches(ANY).matches_nothing());
    }

    /// A guard prints as the constructor that builds it, so a diagnostic quoting one reads back as
    /// code. Pinned because that string is the constructor's name spelled a second time, where no
    /// compiler would notice the two drifting apart.
    #[test]
    fn a_guard_prints_as_the_call_that_builds_it() {
        assert_eq!(TagGuard::all().to_string(), "TagGuard::all()");
        assert_eq!(
            TagGuard::matches([BitReq::One, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]).to_string(),
            "TagGuard::matches([BitReq::One, BitReq::Zero, BitReq::Ignore, BitReq::Ignore])"
        );
        assert_eq!(
            TagGuard::not_matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]).to_string(),
            "TagGuard::not_matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore])"
        );
    }

    /// `as_group` recognises exactly the guards that select one group, whichever way the kernel wrote
    /// them, and nothing else. Lowering keys the compact hardware group form off this, so a guard the
    /// kernel spelled as a negation has to be recognised too -- otherwise the same request would be
    /// accepted or refused depending on its spelling.
    #[test]
    fn as_group_recognises_every_spelling_of_one_group() {
        for id in [GroupId::Zero, GroupId::One] {
            assert_eq!(TagGuard::group(id).as_group(), Some(id));
        }
        assert_eq!(
            TagGuard::not_matches([BitReq::Ignore, BitReq::Ignore, BitReq::Ignore, BitReq::One]).as_group(),
            Some(GroupId::Zero),
            "\"not group one\" is group zero: bit 3 has only two values"
        );

        assert_eq!(TagGuard::all().as_group(), None);
        assert_eq!(TagGuard::matches([BitReq::Ignore; 4]).as_group(), None);
        assert_eq!(
            TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::One]).as_group(),
            None,
            "constraining bit0 as well is not a group guard"
        );
        assert_eq!(
            TagGuard::not_matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::One]).as_group(),
            None,
            "negating a two-bit pattern is not one group either"
        );
    }

    /// The two slots these tests name, spelled short so an expected defect reads as one line.
    const REG0: Slot = Slot::Imm(ImmSlot::Reg0);
    const REG1: Slot = Slot::Imm(ImmSlot::Reg1);

    /// One case per [`SlotDefect`] variant, each built the way that defect actually arises, plus the
    /// shapes that are fine so the rules are not passing by refusing everything. This is the one place
    /// the rules live -- the eDSL builder, the ViSA translator and lowering to `VePass` all report
    /// what `validate` decides -- so it is the one place they are pinned.
    #[test]
    fn unittest_slot_defect() {
        type Layout = BranchedOperand<i32, i32>;
        type Node = Operands<i32, i32>;
        let bit0 = TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
        let bit1 = TagGuard::matches([BitReq::Ignore, BitReq::One, BitReq::Ignore, BitReq::Ignore]);
        let bit2 = TagGuard::matches([BitReq::Ignore, BitReq::Ignore, BitReq::One, BitReq::Ignore]);
        let dead = TagGuard::not_matches([BitReq::Ignore; 4]);

        // `Empty`: an operand that drives nothing at all.
        assert_eq!(Layout::default().validate(), Err(SlotDefect::Empty));

        // `DeadGuard` naming a slot: the slot is filled, it costs a slot, and it can never fire.
        assert_eq!(
            Layout::try_from_slots([Some((dead, 1)), None, None], None),
            Err(SlotDefect::DeadGuard { slot: Some(REG0) })
        );

        // `DeadGuard` naming none: a unary node has no slots, so its guard gates the node itself.
        // `Operands` is the only place this shape comes from, since `BranchedOperand` never sees one.
        assert_eq!(Node::Unary(dead).validate(), Err(SlotDefect::DeadGuard { slot: None }));

        // `Occupied`: filling a slot that already holds a payload would drop what it holds.
        let mut occupied = Layout::default();
        occupied
            .fill_imm(ImmSlot::Reg0, bit0, 1)
            .expect("the first immediate is free");
        assert_eq!(
            occupied.fill_imm(ImmSlot::Reg0, bit1, 2),
            Err(SlotDefect::Occupied { slot: REG0 })
        );

        // `NoFreeImmediate`: a pass has three immediate registers, and a fourth has nowhere to go.
        let mut spent = Layout::default();
        for (slot, guard) in [(ImmSlot::Reg0, bit0), (ImmSlot::Reg1, bit1), (ImmSlot::Reg2, bit2)] {
            spent.fill_imm(slot, guard, 1).expect("a pass has three immediates");
        }
        assert_eq!(
            spent.fill_next_imm(TagGuard::all(), 4),
            Err(SlotDefect::NoFreeImmediate)
        );

        // `AfterUnconditional`: the unconditional slot claims every element the earlier guards left
        // over, so a slot behind it could never fire.
        assert_eq!(
            Layout::try_from_slots([Some((TagGuard::all(), 1)), Some((bit0, 2)), None], None),
            Err(SlotDefect::AfterUnconditional {
                unconditional: REG0,
                slot: REG1
            })
        );

        // Fine: a guarded slot with an unconditional one after it -- the pass's `else`, and the shape
        // the rules exist to allow. The two overlap, and which fires is slot order, the hardware's
        // priority.
        assert!(Layout::try_from_slots([Some((bit0, 1)), None, None], Some((TagGuard::all(), 2))).is_ok());
        // Fine: one unconditional slot on its own is a whole pass, and a guard that admits some
        // element gates a node as happily as it gates a slot.
        assert!(Layout::try_from_slots([Some((TagGuard::all(), 1)), None, None], None).is_ok());
        assert!(Node::Unary(TagGuard::group(GroupId::One)).validate().is_ok());
    }

    /// A fill is refused for any of the reasons a bulk construction is, plus the two only it can have
    /// -- a slot already taken, and a fourth immediate -- and a refused fill leaves the layout as it
    /// was, which is what lets a caller report and carry on.
    #[test]
    fn a_refused_fill_changes_nothing() {
        let guarded = TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
        let other = TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
        let mut layout = BranchedOperand::<i32, i32>::default();

        layout
            .fill_imm(ImmSlot::Reg0, guarded, 1)
            .expect("the first immediate is free");
        assert_eq!(
            layout.fill_imm(ImmSlot::Reg0, other, 2),
            Err(SlotDefect::Occupied { slot: REG0 })
        );
        assert_eq!(
            layout.reg_slots()[0],
            &Some((guarded, 1)),
            "the refused fill left it alone"
        );

        layout.fill_next_imm(other, 2).expect("two immediates are still free");
        layout
            .fill_next_imm(
                TagGuard::matches([BitReq::Ignore, BitReq::One, BitReq::Ignore, BitReq::Ignore]),
                3,
            )
            .unwrap();
        assert_eq!(
            layout.fill_next_imm(TagGuard::all(), 4),
            Err(SlotDefect::NoFreeImmediate)
        );

        // The rf port is its own slot, so it is still free when the immediates are spent.
        layout
            .fill_rf(TagGuard::all(), 9)
            .expect("the port is separate from the immediates");
        assert_eq!(
            layout.fill_rf(TagGuard::all(), 10),
            Err(SlotDefect::Occupied { slot: Slot::Rf })
        );

        // The layout rules apply to a fill too, not just to a bulk construction.
        let mut after = BranchedOperand::<i32, i32>::default();
        after.fill_imm(ImmSlot::Reg0, TagGuard::all(), 1).unwrap();
        assert_eq!(
            after.fill_imm(ImmSlot::Reg1, guarded, 2),
            Err(SlotDefect::AfterUnconditional {
                unconditional: REG0,
                slot: REG1
            })
        );
        assert!(after.reg_slots()[1].is_none(), "the refused fill left it alone");

        let mut dead = BranchedOperand::<i32, i32>::default();
        assert_eq!(
            dead.fill_imm(ImmSlot::Reg0, TagGuard::not_matches([BitReq::Ignore; 4]), 1),
            Err(SlotDefect::DeadGuard { slot: Some(REG0) })
        );
    }

    /// [`TagGuard::all`] on the wire: the pattern that constrains no bit, which is the only shape it
    /// has now that there is no unconditional variant of its own.
    fn unconditional_guard() -> serde_lite::Intermediate {
        let mut map = serde_lite::Map::new();
        map.insert_with_static_key(
            "Matches",
            serde_lite::Intermediate::Array(vec!["Ignore".into(), "Ignore".into(), "Ignore".into(), "Ignore".into()]),
        );
        serde_lite::Intermediate::Map(map)
    }

    /// One filled slot, as the four named fields a `serde-lite` payload carries.
    fn one_filled_slot() -> serde_lite::Intermediate {
        let mut map = serde_lite::Map::new();
        map.insert_with_static_key("reg0", serde_lite::Intermediate::None);
        map.insert_with_static_key(
            "reg1",
            serde_lite::Intermediate::Array(vec![unconditional_guard(), 7i64.into()]),
        );
        map.insert_with_static_key("reg2", serde_lite::Intermediate::None);
        map.insert_with_static_key("rf", serde_lite::Intermediate::None);
        serde_lite::Intermediate::Map(map)
    }

    /// What makes the hand-written [`serde_lite::Deserialize`] impls a replacement for the derive
    /// rather than a rewrite: the payload is the four named slots, spelled as an `Intermediate` by
    /// hand so a shape change on either side shows up here.
    #[test]
    fn the_serde_lite_shape_is_the_named_slots() {
        let operand: BranchedOperand<i32, i32> =
            BranchedOperand::from_slots([None, Some((TagGuard::all(), 7)), None], None);
        assert_eq!(
            <BranchedOperand<i32, i32> as serde_lite::Deserialize>::deserialize(&one_filled_slot()).unwrap(),
            operand
        );
    }

    /// A payload short of slots reports every one it is missing, not just the first. The impl reads
    /// all four before deciding precisely so this holds; returning early would still pass a test that
    /// only checked the error variant.
    #[test]
    fn a_missing_slot_is_reported_under_its_name() {
        let mut map = serde_lite::Map::new();
        map.insert_with_static_key(
            "reg1",
            serde_lite::Intermediate::Array(vec![unconditional_guard(), 7i64.into()]),
        );

        let err =
            <BranchedOperand<i32, i32> as serde_lite::Deserialize>::deserialize(&serde_lite::Intermediate::Map(map))
                .unwrap_err();

        let serde_lite::Error::NamedFieldErrors(errors) = err else {
            panic!("expected NamedFieldErrors, got {err}");
        };
        let missing: Vec<&str> = errors.iter().map(serde_lite::NamedFieldError::field).collect();
        assert_eq!(missing, ["reg0", "reg2", "rf"]);
    }

    /// [`Operands`] is externally tagged: a one-key map naming the arity. Both a variant carrying
    /// slots and the unary one carrying only a guard are pinned, since they take different paths
    /// through the impl.
    #[test]
    fn serde_lite_operands_are_externally_tagged() {
        let mut binary = serde_lite::Map::new();
        binary.insert_with_static_key("Binary", one_filled_slot());
        assert_eq!(
            <Operands<i32, i32> as serde_lite::Deserialize>::deserialize(&serde_lite::Intermediate::Map(binary))
                .unwrap(),
            Operands::Binary(BranchedOperand::from_slots(
                [None, Some((TagGuard::all(), 7)), None],
                None
            ))
        );

        let mut unary = serde_lite::Map::new();
        unary.insert_with_static_key("Unary", unconditional_guard());
        assert_eq!(
            <Operands<i32, i32> as serde_lite::Deserialize>::deserialize(&serde_lite::Intermediate::Map(unary))
                .unwrap(),
            Operands::Unary(TagGuard::all())
        );
    }
}
