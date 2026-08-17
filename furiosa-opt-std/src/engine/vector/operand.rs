//! Operand types for Vector Engine operations.
//!
//! This module provides types for specifying operands in VE binary and ternary operations:
//! - [`Stash`]: read-once stash operand, passed to any VE binary/clip/ternary op
//! - [`Branched`]: opens a guarded operand by naming the hardware slot it drives, which is how a
//!   kernel writes a condition; a bare operand needs none
//! - [`VeOperandBuilder`]: what a [`Branched`] constructor returns, carrying the slots filled so far
//!   and offering exactly the ones that may still follow, read-once enforced per pass
//! - [`IntoBranchedOperand`]: every operand form (a constant, a VRF register, [`Stash`], the ternary
//!   `(rhs, operand1)`, a [`Branched`] layout) as the slots it fills
//! - [`StashTransition`]: how using an operand transitions the pipeline's stash typestate
//!
//! # Writing sugar on top, and what it cannot do
//!
//! A crate downstream of this one can wrap these types in helpers of its own. Two limits are worth
//! knowing before you try, because neither is obvious from the signatures.
//!
//! **A downstream crate cannot add a new operand type.** The op methods bound their operand on
//! [`IntoBranchedOperand`] + [`StashTransition`], and while both traits are public, neither is
//! implementable from outside. [`Plain`], the blanket route, is sealed; and `StashTransition` is
//! closed off by its own signature, which names the pipeline's `VeState` -- a `pub(crate)` type a
//! downstream crate cannot write down. That is deliberate: an operand type carries a claim about what
//! it does to the stash, and getting it wrong is silent.
//!
//! What a downstream crate *can* do is build layouts and hand them back, which covers the sugar worth
//! writing:
//!
//! ```
//! use furiosa_opt_std::prelude::*;
//! type Map = m![1];
//!
//! // Returns a layout the caller passes straight to an op.
//! fn clamp_band(lo: TagGuard, hi: TagGuard) -> VeOperandBuilder<f32, Map, f32, NoStash, TwoRegs> {
//!     Branched::imm(lo, -1.0f32).imm(hi, 1.0f32)
//! }
//!
//! // And it may append an immediate from wherever the caller has already got to, by naming the
//! // capability rather than a position: this works on a one- or two-immediate builder alike.
//! fn with_floor<Slot: CanAppendImm>(
//!     b: VeOperandBuilder<f32, Map, f32, NoStash, Slot>,
//!     guard: TagGuard,
//! ) -> VeOperandBuilder<f32, Map, f32, NoStash, Slot::Next> {
//!     b.imm(guard, 0.0f32)
//! }
//! ```
//!
//! **A helper cannot fill a slot conditionally.** How many slots are spoken for is part of the
//! builder's type, so the two arms of an `if` have different types and there is no return type to
//! write:
//!
//! ```compile_fail,E0308
//! use furiosa_opt_std::prelude::*;
//! type Map = m![1];
//! fn maybe_second(
//!     b: VeOperandBuilder<f32, Map, f32, NoStash, OneReg>,
//!     cond: bool,
//! ) -> VeOperandBuilder<f32, Map, f32, NoStash, TwoRegs> {
//!     if cond { b.imm(TagGuard::all(), 2.0f32) } else { b } // OneReg vs TwoRegs
//! }
//! ```
//!
//! Vary the payload or the guard instead of the slot count, or branch at the call site around the
//! whole op. What will not work is reaching for a guard that fires on nothing to stand in for "no
//! slot": that is exactly the dead slot [`Branched`] rejects.

use std::marker::PhantomData;

use furiosa_mapping::M;
use furiosa_opt_macro::primitive;

// From the modules that define these, not through this crate's own prelude: the prelude is the
// surface a kernel author reaches for, not an internal path.
use crate::engine::vector::branch::{BinaryBranchedOperand, RfPort, TagGuard, TernaryBranchedOperand};
use crate::engine::vector::op::FpUnaryOp;
use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::stage::markers::Way;
use crate::engine::vector::stage::state::VeState;
use crate::engine::vector::stash_slot::{Occupied, Spent, StashState};
use crate::tensor::Tensor;
use crate::tensor::memory::VrfTensor;
// Straight from the common IR, not through `branch`, which deliberately does not re-export it: a
// kernel author reaches a layout only through [`Branched`]. Named here because the builder holds one.
use furiosa_opt_common_ir::{BranchedOperand, ImmSlot, SlotDefect};

/// The operand that reads the stash: `vector_fp_binary(op, Stash)`, `vector_fp_ternary(op,
/// (Stash, c))`. A value read more than once is not a stash -- use a read-many [`VrfTensor`].
#[primitive(op::Stash)]
#[derive(Debug, Clone, Copy)]
pub struct Stash;

mod sealed {
    /// Private supertrait of [`Plain`](super::Plain): a new operand type cannot silently be treated
    /// as stash-agnostic. Adding an operand means an explicit `impl Plain` (identity stash
    /// transition) or an explicit [`StashTransition`](super::StashTransition), never a default.
    pub trait Sealed {}

    /// Private supertrait of the traits that decide what a slot carries and how it is filled.
    ///
    /// They reach the raw slots and pick the ISA encoding, so the set of operand types is this crate's
    /// to enumerate. Left open, a downstream crate could hand the builder a payload the ViSA translator
    /// cannot encode, or fill the same slot twice through `fill_slots`.
    ///
    /// Separate from [`Sealed`] rather than reusing it: that one deliberately leaves
    /// [`Stash`](super::Stash) out so it cannot pass as stash-agnostic, and these traits do accept it.
    pub trait SlotSealed {}
}

/// An operand that leaves the stash slot alone (everything but [`Stash`]). Only exists to exclude
/// `Stash` from the identity [`StashTransition`] blanket below -- Rust has no negative bounds, so a
/// catch-all `impl<T>` plus the `Stash` override would collide. Adding an operand therefore means an
/// explicit `impl Plain` or an explicit [`StashTransition`], so forgetting the transition is a compile
/// error rather than silent identity.
///
/// Sealed; see the [module docs](self) for what a downstream crate can do instead.
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::Plain;
/// struct MyOperand;
/// impl Plain for MyOperand {} // sealed: the private supertrait is unreachable from here
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be used as an operand here",
    label = "not usable as this op's operand",
    note = "if this is `Stash`: a `vector_stash` has to come earlier, only one op may read it, and the \
            stream has to carry the same scalar and way it carried at that write (see `StashTransition`)"
)]
pub trait Plain: sealed::Sealed {}

/// How an operand moves the stash slot: `S` -> [`Self::Next`], for a reader of scalar `RD` on a
/// way-`W` pipeline. [`Plain`] operands are the identity; [`Stash`] is impl'd only on
/// `Occupied<RD, _, W>`, so an empty, spent, repeated, cross-scalar or cross-way read has no impl.
/// See the stash-slot [module docs] for the state machine.
///
/// [module docs]: crate::engine::vector::stash_slot
///
/// A live stash at the reader's scalar and way reads:
/// ```
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Occupied, Stash, StashState, StashTransition, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<f32, Broadcast<1>, { Way::Way8 }>, f32, { Way::Way8 }, Stash>();
/// ```
/// A stash written as `i32` cannot be read by an `f32` op, so undo the conversion or reinterpret first:
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Occupied, Stash, StashState, StashTransition, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<i32, Broadcast<1>, { Way::Way8 }>, f32, { Way::Way8 }, Stash>();
/// ```
/// Nor 4-way by an 8-way op: the operand register is addressed per way, so the reader would get the
/// wrong elements.
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Occupied, Stash, StashState, StashTransition, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<f32, Broadcast<1>, { Way::Way4 }>, f32, { Way::Way8 }, Stash>();
/// ```
/// Never written, and read twice, are the same missing impl:
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Fresh, Stash, StashState, StashTransition, VeScalar};
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Fresh, f32, { Way::Way8 }, Stash>();
/// ```
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Spent, Stash, StashState, StashTransition, VeScalar};
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Spent, f32, { Way::Way8 }, Stash>();
/// ```
#[diagnostic::on_unimplemented(
    message = "the stash cannot be read here",
    label = "this operand reads the stash",
    note = "the stash keeps the scalar and the way the stream carried at the `vector_stash`, and only a \
            stream carrying that same pair can read it: reinterpret (or narrow / widen) back first",
    note = "it is written once and read once, so with no earlier `vector_stash` the slot is empty, and a \
            second `Stash` operand finds it spent"
)]
pub trait StashTransition<S: StashState, RD: VeScalar, const W: Way> {
    /// Stash typestate after this operand.
    type Next: StashState;

    /// Runs the transition on the value (drops the stash tensor iff this operand read it). Needed
    /// because the state stores the stash by value, so `S` -> `Next` is a real move, not a retag.
    fn transition(state: VeState<S>) -> VeState<Self::Next>;

    /// The stashed tensor this operand reads, in the reader's mapping, or `None` for an operand that
    /// does not read the stash. Which one it is follows from the operand type, so the engine needs
    /// no runtime look at the lowered operands.
    fn stashed<Mapping: M>(state: &VeState<S>) -> Option<Tensor<RD, Mapping>>;
}

impl<S: StashState, RD: VeScalar, T: Plain, const W: Way> StashTransition<S, RD, W> for T {
    type Next = S;
    fn transition(state: VeState<S>) -> VeState<S> {
        state
    }
    fn stashed<Mapping: M>(_state: &VeState<S>) -> Option<Tensor<RD, Mapping>> {
        None
    }
}

impl<D: VeScalar, StashMapping: M, const W: Way> StashTransition<Occupied<D, StashMapping, W>, D, W> for Stash {
    type Next = Spent;
    fn transition(state: VeState<Occupied<D, StashMapping, W>>) -> VeState<Spent> {
        state.consume_stash()
    }
    fn stashed<Mapping: M>(state: &VeState<Occupied<D, StashMapping, W>>) -> Option<Tensor<D, Mapping>> {
        Some(state.stash_tensor())
    }
}

// Ternary `(Stash, c)`: same read-once rule; the stash is operand0. Delegates to the `Stash`
// transition so the two share one body.
impl<D: VeScalar, StashMapping: M, const W: Way> StashTransition<Occupied<D, StashMapping, W>, D, W> for (Stash, f32) {
    type Next = Spent;
    fn transition(state: VeState<Occupied<D, StashMapping, W>>) -> VeState<Spent> {
        <Stash as StashTransition<Occupied<D, StashMapping, W>, D, W>>::transition(state)
    }
    fn stashed<Mapping: M>(state: &VeState<Occupied<D, StashMapping, W>>) -> Option<Tensor<D, Mapping>> {
        <Stash as StashTransition<Occupied<D, StashMapping, W>, D, W>>::stashed(state)
    }
}

// ============================================================================
// Stash markers for the multi-slot operand builder
// ============================================================================

/// Type-level marker: this operand leaves the stash slot empty, so it leaves the pipeline's stash
/// alone. See [`VeOperandBuilder`].
#[derive(Debug, Clone, Copy)]
pub struct NoStash;

/// Type-level marker: this operand fills the stash slot. A VE op runs *one* pass, so the slot is one
/// guarded read of the one physical stash -- a single `Occupied` -> `Spent` transition. See
/// [`VeOperandBuilder`].
#[derive(Debug, Clone, Copy)]
pub struct WithStash;

// ============================================================================
// Rhs slot positions - how much of the pass is spoken for, as a typestate
// ============================================================================

// How many of a [`VeOperandBuilder`]'s slots are spoken for.
//
// A pass has three immediate registers and one rf port, and the port is filled last. The kernel does
// not choose *which* immediate, so all the typestate carries is how many are gone and whether the port
// is filled. There is no "nothing filled yet" position: [`Branched`] opens a builder by filling a slot,
// so a layout that drives nothing is not a state the API can be in.
//
// The legal transitions are the capability traits below rather than a method per position, so a helper
// can name "a builder with an immediate left" instead of picking one position.

/// One immediate register is spoken for: `imm` and `rf` may still follow.
#[derive(Debug, Clone, Copy)]
pub struct OneReg;

/// Two immediate registers are spoken for: `imm` and `rf` may still follow.
#[derive(Debug, Clone, Copy)]
pub struct TwoRegs;

/// All three immediate registers are spoken for, so only `rf` may follow.
#[derive(Debug, Clone, Copy)]
pub struct ThreeRegs;

/// The rf port was the last slot filled, so this position has no methods at all.
///
/// That is also what makes a second stash read unrepresentable, with no `NoStash` bound needed
/// anywhere: [`rf`](VeOperandBuilder::rf) is the only way to read the stash, and it lands here.
///
/// ```compile_fail,E0599
/// use furiosa_opt_std::prelude::{Branched, Stash, TagGuard};
/// use furiosa_mapping::Broadcast;
/// let _ = Branched::rf::<f32, Broadcast<1>, f32, _>(TagGuard::all(), Stash)
///     .rf(TagGuard::all(), Stash); // the port is filled twice: no method
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AtRf;

/// A slot position with an immediate register still to spend, and the position that spending it
/// reaches. [`ThreeRegs`] and [`AtRf`] have no impl, which makes a fourth immediate a compile error.
///
/// This is the bound for a helper that appends an immediate and does not care how many are already
/// gone -- `fn scale<S: CanAppendImm>(b: Builder<.., S>) -> Builder<.., S::Next>`. Naming [`OneReg`]
/// instead would tie the helper to being called first.
#[diagnostic::on_unimplemented(
    message = "this VE pass has no immediate register left",
    label = "all three immediate registers are already spoken for",
    note = "a pass has three immediates and one rf port; drop one of the earlier `imm` calls, or move \
            the value onto the rf port with `rf` if it is a register read"
)]
pub trait CanAppendImm {
    /// The position reached by filling one more immediate register.
    type Next;

    /// The immediate register the next `imm` call lands in. The typestate is what keeps this dense
    /// and in order, so there is no free-slot search.
    const NEXT_IMM: ImmSlot;
}

impl CanAppendImm for OneReg {
    type Next = TwoRegs;
    const NEXT_IMM: ImmSlot = ImmSlot::Reg1;
}

impl CanAppendImm for TwoRegs {
    type Next = ThreeRegs;
    const NEXT_IMM: ImmSlot = ImmSlot::Reg2;
}

/// A slot position whose rf port is still free: every position but [`AtRf`].
///
/// Separate from [`CanAppendImm`] because the two part ways at [`ThreeRegs`], where the immediates are
/// spent and the port is not.
#[diagnostic::on_unimplemented(
    message = "this VE pass has already filled its register-file port",
    label = "the rf port is filled once, and it is the last slot",
    note = "a pass reads the register file once, so one `rf` call -- and because the stash is read \
            through `rf`, this is also what stops a second stash read"
)]
pub trait CanFillRf {}

impl CanFillRf for OneReg {}
impl CanFillRf for TwoRegs {}
impl CanFillRf for ThreeRegs {}

// ============================================================================
// VeOperandBuilder - naming the hardware slots directly
// ============================================================================

/// What one arity's slots carry. `Self` is the immediate register's payload, and the rf port's
/// payload follows from it: a binary slot's port carries the port alone, a ternary slot's carries the
/// port plus the `operand1` that branch feeds.
///
/// The trait sits on the *register* payload rather than on a separate arity marker so that naming the
/// register type pins the whole layout, port included. That is what lets one [`Branched`] slot
/// constructor serve both arities: its argument names `Self`, and the port slot follows without a
/// turbofish.
pub trait RegPayload<D: VeScalar, Mapping: M>: sealed::SlotSealed {
    /// What the pass's one rf port slot carries.
    type Port;
}

/// A binary `i32` pass: the port is a register or stash read, with nothing riding along.
impl<Mapping: M> RegPayload<i32, Mapping> for i32 {
    type Port = RfPort<i32, Mapping>;
}

/// A binary `f32` pass, the same shape as the `i32` one.
impl<Mapping: M> RegPayload<f32, Mapping> for f32 {
    type Port = RfPort<f32, Mapping>;
}

/// A ternary pass: every slot carries its own `operand1`, so the port payload grows one `f32`.
/// Ternary ops are f32-only, which is why there is no `i32` counterpart.
impl<Mapping: M> RegPayload<f32, Mapping> for (f32, f32) {
    type Port = (RfPort<f32, Mapping>, f32);
}

/// Opens a guarded operand. The namespace a conditional operand is written through:
///
/// ```ignore
/// tensor.vector_logic(LogicBinaryOpI32::BitAnd, Branched::imm(guard, 0x7fff_ffff))
/// tensor.vector_fxp(FxpBinaryOp::AddFxpSat, Branched::imm(g0, 1).rf(g1, &scale_vrf))
/// tensor.vector_fp_ternary(FpTernaryOp::FmaF, Branched::imm(guard, (2.0f32, 3.0f32)))
/// ```
///
/// The rule for a kernel author is one sentence: with no condition write the operand bare (`100`,
/// `&vrf`, [`Stash`], `(2.0, 3.0)`), and with a condition say which *kind* of slot it drives -- an
/// immediate register (`imm`) or the register-file port (`rf`).
///
/// `imm` takes the next free immediate register; the kernel does not pick one, and which physical
/// register a value lands in is command generation's to decide. What *is* preserved is call order: the
/// slots reach the hardware in the order they were filled.
///
/// The arity comes from the constructor's argument: a scalar builds a [`BinaryBranchedOperand`], an
/// `(operand0, operand1)` pair a [`TernaryBranchedOperand`].
///
/// Each constructor returns a [`VeOperandBuilder`] with that one slot already filled, which is where
/// the remaining rules live. There is no empty state to open: an operand that fills no slot would
/// consume an ALU and move the pipeline's stage while driving nothing.
#[derive(Debug)]
pub enum Branched {}

impl Branched {
    /// Opens with one immediate register: `v`, applied where `guard` matches.
    #[primitive(op::Branched::imm)]
    pub fn imm<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>>(
        guard: TagGuard,
        v: Reg,
    ) -> VeOperandBuilder<D, Mapping, Reg, NoStash, OneReg> {
        fill_reg(BranchedOperand::default(), ImmSlot::Reg0, guard, v)
    }

    /// Opens with the pass's rf port: `arg`, applied where `guard` matches. Nothing may follow it.
    #[primitive(op::Branched::rf)]
    pub fn rf<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, A: PortArg<D, Mapping, Reg>>(
        guard: TagGuard,
        arg: A,
    ) -> VeOperandBuilder<D, Mapping, Reg, A::Mark, AtRf> {
        fill_rf(BranchedOperand::default(), guard, arg)
    }
}

// The two fills, in one place each: every constructor and follow-on method is one of these calls.
// Free functions rather than methods because the incoming and outgoing `Slot` differ and the incoming
// one is often not yet decided, leaving an inherent `Self::` call nothing to infer its position from.

/// Fills immediate register `imm`, landing at whatever `Slot` the caller's return type names. The
/// stash marker rides along: an immediate never reads the stash, so filling one cannot change it.
fn fill_reg<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, StashMark, Slot>(
    mut operand: BranchedOperand<Reg, Reg::Port>,
    imm: ImmSlot,
    guard: TagGuard,
    v: Reg,
) -> VeOperandBuilder<D, Mapping, Reg, StashMark, Slot> {
    assert_fill(operand.fill_imm(imm, guard, v));
    VeOperandBuilder {
        operand,
        _mark: PhantomData,
    }
}

/// Fills the rf port, landing at [`AtRf`] and taking the stash marker from what the port read.
fn fill_rf<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, A: PortArg<D, Mapping, Reg>>(
    mut operand: BranchedOperand<Reg, Reg::Port>,
    guard: TagGuard,
    arg: A,
) -> VeOperandBuilder<D, Mapping, Reg, A::Mark, AtRf> {
    assert_fill(operand.fill_rf(guard, arg.into_port()));
    VeOperandBuilder {
        operand,
        _mark: PhantomData,
    }
}

/// Panics if a fill was refused.
///
/// The rules are `BranchedOperand::validate`'s, shared with the ViSA translator and with lowering to
/// `VePass` so they cannot drift apart; only the reporting is local.
///
/// Panicking is a host-side backstop, not the enforcement: a `#[device]` function's body is read as MIR
/// and never run, so this never fires for a compiled kernel. The translator checks the same rules
/// against the kernel's span, which is what actually rejects them.
fn assert_fill(filled: Result<(), SlotDefect>) {
    if let Err(defect) = filled {
        panic!("{defect}");
    }
}

/// Fluent builder for one VE pass's slots, created by a [`Branched`] constructor. The register payload
/// decides the arity: a scalar builds a [`BinaryBranchedOperand`], an `(operand0, operand1)` pair a
/// [`TernaryBranchedOperand`], and the rf slot follows suit.
///
/// `imm` may be called three times and `rf` once, last -- each position below carries exactly the methods
/// that may still follow it. `StashMark` records whether the rf port read the stash, which is what
/// [`StashTransition`] keys on for read-once.
///
/// Positive: two immediates and the rf port, the most a pass can drive on the immediate side plus its
/// one read.
/// ```
/// use furiosa_opt_std::prelude::{Branched, BitReq::{Ignore, One, Zero}, TagGuard};
/// use furiosa_mapping::Broadcast;
/// let _ = Branched::imm::<f32, Broadcast<1>, _>(TagGuard::matches([One, Ignore, Ignore, Ignore]), 1.0)
///     .imm(TagGuard::matches([Zero, One, Ignore, Ignore]), 2.0);
/// ```
/// Positive: the same chain as a *ternary* operand, where each register carries its own `operand1`
/// and the rf slot takes one too.
/// ```
/// use furiosa_opt_std::prelude::{Branched, BitReq::{Ignore, One}, Stash, TagGuard};
/// use furiosa_mapping::Broadcast;
/// let _ = Branched::imm::<f32, Broadcast<1>, _>(TagGuard::not_matches([One, Ignore, Ignore, Ignore]), (1.0, 2.0))
///     .rf(TagGuard::all(), (Stash, 3.0));
/// ```
/// Negative: a fourth immediate -- `ThreeRegs` does not implement `CanAppendImm`, the pass having three.
/// ```compile_fail,E0599
/// use furiosa_opt_std::prelude::{Branched, BitReq::{Ignore, One, Zero}, TagGuard};
/// use furiosa_mapping::Broadcast;
/// let _ = Branched::imm::<f32, Broadcast<1>, _>(TagGuard::matches([One, Ignore, Ignore, Ignore]), 1.0)
///     .imm(TagGuard::matches([Zero, One, Ignore, Ignore]), 2.0)
///     .imm(TagGuard::matches([Zero, Zero, One, Ignore]), 3.0)
///     .imm(TagGuard::matches([Zero, Zero, Zero, One]), 4.0);
/// ```
/// Negative: an immediate after the rf port -- `AtRf` implements neither capability.
/// ```compile_fail,E0599
/// use furiosa_opt_std::prelude::{Branched, Stash, TagGuard};
/// use furiosa_mapping::Broadcast;
/// let _ = Branched::rf::<f32, Broadcast<1>, f32, _>(TagGuard::all(), Stash)
///     .imm(TagGuard::all(), 1.0);
/// ```
///
/// `StashMark` is [`WithStash`] once [`rf`](Self::rf) has taken a [`Stash`], [`NoStash`] otherwise,
/// and [`StashTransition`] keys on it for read-once -- see its docs for the state machine.
///
/// Positive: an operand with a stash slot satisfies the exact op-method bound
/// (`IntoBranchedOperand` + `StashTransition`) on a live (`Occupied`) slot -- one pass, one read.
/// ```
/// # #![feature(adt_const_params)]
/// use furiosa_opt_std::prelude::{
///     stage::Way, StashTransition, IntoBranchedOperand, VeOperandBuilder, AtRf, Stash, TagGuard,
///     WithStash, Occupied, StashState, VeScalar, M, Branched,
/// };
/// use furiosa_mapping::Broadcast;
/// // The same bound `vector_fp_binary`/`vector_fxp` put on their operand `Op`.
/// fn op_operand<SD, S, Map, const W: Way, Op>()
/// where
///     SD: VeScalar, S: StashState, Map: M,
///     Op: IntoBranchedOperand<SD, Map> + StashTransition<S, SD, W>,
/// {}
/// // A `WithStash` builder sits at `AtRf` by construction: the stash fills the rf port.
/// op_operand::<
///     f32,
///     Occupied<f32, Broadcast<1>, { Way::Way8 }>,
///     Broadcast<1>,
///     { Way::Way8 },
///     VeOperandBuilder<f32, Broadcast<1>, f32, WithStash, AtRf>,
/// >();
/// // Filling the stash slot is what produces that `WithStash` operand.
/// let _ = Branched::rf::<f32, Broadcast<1>, f32, _>(TagGuard::all(), Stash);
/// ```
/// Negative: the same operand on a `Spent` slot -- the stash was already read by an earlier op, so a
/// second read anywhere in the chain finds no impl.
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, StashTransition, VeOperandBuilder, WithStash, AtRf, Spent, StashState, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Spent, f32, { Way::Way8 }, VeOperandBuilder<f32, Broadcast<1>, f32, WithStash, AtRf>>();
/// ```
/// Negative: the same operand on a `Fresh` (never-written) slot -- nothing to read.
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, StashTransition, VeOperandBuilder, WithStash, AtRf, Fresh, StashState, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Fresh, f32, { Way::Way8 }, VeOperandBuilder<f32, Broadcast<1>, f32, WithStash, AtRf>>();
/// ```
#[derive(Debug, Clone)]
pub struct VeOperandBuilder<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, StashMark = NoStash, Slot = OneReg> {
    operand: BranchedOperand<Reg, Reg::Port>,
    _mark: PhantomData<(D, Mapping, StashMark, Slot)>,
}

// The transition graph, written out. Each block is one position and holds exactly the calls that may
// still follow it, so a fourth `imm` or an `imm` after `rf` has no impl to reach.
//
// [`AtRf`] gets no block: the pass has one rf port and it is applied last, so nothing follows it.

/// Appending an immediate, from any position that still has one.
impl<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, StashMark, Slot: CanAppendImm>
    VeOperandBuilder<D, Mapping, Reg, StashMark, Slot>
{
    /// The next immediate register: `v`, applied where `guard` matches.
    #[primitive(op::VeOperandBuilder::imm)]
    pub fn imm(self, guard: TagGuard, v: Reg) -> VeOperandBuilder<D, Mapping, Reg, StashMark, Slot::Next> {
        fill_reg(self.operand, Slot::NEXT_IMM, guard, v)
    }
}

/// Filling the rf port, from any position that has not filled it.
impl<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, StashMark, Slot: CanFillRf>
    VeOperandBuilder<D, Mapping, Reg, StashMark, Slot>
{
    /// The pass's rf port: `arg`, applied where `guard` matches.
    #[primitive(op::VeOperandBuilder::rf)]
    pub fn rf<A: PortArg<D, Mapping, Reg>>(
        self,
        guard: TagGuard,
        arg: A,
    ) -> VeOperandBuilder<D, Mapping, Reg, A::Mark, AtRf> {
        fill_rf(self.operand, guard, arg)
    }
}

/// A unary op, with or without the guard saying where it runs.
///
/// The unary counterpart of [`IntoBranchedOperand`]: there, a bare operand means
/// [`TagGuard::all`] and a [`Branched`] layout carries the guards the kernel wrote. A unary op has no
/// operand to carry one, so the guard pairs with the op instead -- `FpUnaryOp::Sigmoid` to run
/// everywhere, `(guard, FpUnaryOp::Sigmoid)` to run where `guard` matches.
///
/// Sealed for the same reason as the slot-filling traits: the forms a kernel may write are this
/// crate's to enumerate.
///
/// ```
/// use furiosa_opt_std::prelude::{BitReq::{Ignore, One}, FpUnaryOp, IntoGuardedUnaryOp, TagGuard};
/// fn unary_op<Op: IntoGuardedUnaryOp>(_op: Op) {}
/// unary_op(FpUnaryOp::Sigmoid);
/// unary_op((TagGuard::matches([One, Ignore, Ignore, Ignore]), FpUnaryOp::Sigmoid));
/// ```
///
/// A guard on its own is not an op, so it does not compile:
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::{IntoGuardedUnaryOp, TagGuard};
/// fn unary_op<Op: IntoGuardedUnaryOp>(_op: Op) {}
/// unary_op(TagGuard::all());
/// ```
pub trait IntoGuardedUnaryOp: sealed::SlotSealed {
    /// The guard this op runs under, and the op.
    fn into_guarded_unary_op(self) -> (TagGuard, FpUnaryOp);
}

/// A bare op runs everywhere, which is what an unguarded operand means on the binary side.
impl IntoGuardedUnaryOp for FpUnaryOp {
    fn into_guarded_unary_op(self) -> (TagGuard, FpUnaryOp) {
        (TagGuard::all(), self)
    }
}

/// Paired with a guard, the op runs on the elements that guard admits.
impl IntoGuardedUnaryOp for (TagGuard, FpUnaryOp) {
    fn into_guarded_unary_op(self) -> (TagGuard, FpUnaryOp) {
        self
    }
}

impl sealed::SlotSealed for FpUnaryOp {}
impl sealed::SlotSealed for TagGuard {}

/// What the pass's one rf port reads, as the argument to [`Branched::rf`] and to the builder's `rf`
/// methods: `&vrf` or [`Stash`] for a binary pass, `(&vrf, operand1)` or `(Stash, operand1)` for a
/// ternary one. The argument names the arity, exactly as an `imm` value does, so no call needs a
/// turbofish for it.
pub trait PortArg<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>>: sealed::SlotSealed {
    /// Whether filling the port with this argument consumes the pipeline's stash: [`WithStash`] for a
    /// stash read, [`NoStash`] for a register read.
    type Mark;

    /// Builds the slot's payload, transposing a register to the op's tensor shape on the way.
    fn into_port(self) -> Reg::Port;
}

/// A binary register read.
impl<D: VeScalar, Mapping: M, Chip: M, Cluster: M, Slice: M, Element: M> PortArg<D, Mapping, D>
    for &VrfTensor<D, Chip, Cluster, Slice, Element>
where
    D: RegPayload<D, Mapping, Port = RfPort<D, Mapping>>,
{
    type Mark = NoStash;

    fn into_port(self) -> RfPort<D, Mapping> {
        RfPort::External(self.inner.transpose::<Mapping>(true))
    }
}

/// A binary stash read.
impl<D: VeScalar, Mapping: M> PortArg<D, Mapping, D> for Stash
where
    D: RegPayload<D, Mapping, Port = RfPort<D, Mapping>>,
{
    type Mark = WithStash;

    fn into_port(self) -> RfPort<D, Mapping> {
        RfPort::Stash
    }
}

/// A ternary register read, carrying this branch's `operand1` alongside `operand0`.
impl<Mapping: M, Chip: M, Cluster: M, Slice: M, Element: M> PortArg<f32, Mapping, (f32, f32)>
    for (&VrfTensor<f32, Chip, Cluster, Slice, Element>, f32)
{
    type Mark = NoStash;

    fn into_port(self) -> (RfPort<f32, Mapping>, f32) {
        let (vrf, operand1) = self;
        (RfPort::External(vrf.inner.transpose::<Mapping>(true)), operand1)
    }
}

/// A ternary stash read, carrying this branch's `operand1`.
impl<Mapping: M> PortArg<f32, Mapping, (f32, f32)> for (Stash, f32) {
    type Mark = WithStash;

    fn into_port(self) -> (RfPort<f32, Mapping>, f32) {
        let (_, operand1) = self;
        (RfPort::Stash, operand1)
    }
}

// A builder with no stash slot leaves the stash alone, so it is `Plain` like any other
// stash-free operand.
impl<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, Slot> sealed::Sealed
    for VeOperandBuilder<D, Mapping, Reg, NoStash, Slot>
{
}
impl<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, Slot> Plain
    for VeOperandBuilder<D, Mapping, Reg, NoStash, Slot>
{
}

// A stash slot reads the stash once for the pass: `Occupied` -> `Spent`, impl'd only on `Occupied`,
// so a second read later in the chain has no impl (read-once). Pinned to `AtRf` because only an `rf`
// fill produces a `WithStash` builder, which makes "a `WithStash` builder is already at the rf port"
// a fact of the type rather than a comment.
impl<D: VeScalar, StashMapping: M, Mapping: M, Reg: RegPayload<D, Mapping>, const W: Way>
    StashTransition<Occupied<D, StashMapping, W>, D, W> for VeOperandBuilder<D, Mapping, Reg, WithStash, AtRf>
{
    type Next = Spent;
    fn transition(state: VeState<Occupied<D, StashMapping, W>>) -> VeState<Spent> {
        state.consume_stash()
    }
    fn stashed<TargetMapping: M>(state: &VeState<Occupied<D, StashMapping, W>>) -> Option<Tensor<D, TargetMapping>> {
        <Stash as StashTransition<Occupied<D, StashMapping, W>, D, W>>::stashed(state)
    }
}

// Leaf operands that leave the stash slot alone.
impl sealed::Sealed for i32 {}
impl Plain for i32 {}
impl sealed::Sealed for f32 {}
impl Plain for f32 {}
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> sealed::Sealed
    for &VrfTensor<D, Chip, Cluster, Slice, Element>
{
}
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> Plain for &VrfTensor<D, Chip, Cluster, Slice, Element> {}

// Composites are [`Plain`] iff every element is: a stash read (`Stash`) is never `Plain`, so a
// tuple containing one is not `Plain` and keeps its explicit `StashTransition`. This is what makes
// `(2.0, 3.0)` (ternary consts) stash-free while `(Stash, 3.0)` transitions the slot.
impl<A: Plain, B: Plain> sealed::Sealed for (A, B) {}
impl<A: Plain, B: Plain> Plain for (A, B) {}

// Every operand form the slot-filling traits accept, and nothing else. `Stash` is here where it is
// absent from `Plain` above: a slot may read the stash, it just may not do so silently.
impl sealed::SlotSealed for i32 {}
impl sealed::SlotSealed for f32 {}
impl sealed::SlotSealed for Stash {}
impl sealed::SlotSealed for () {}
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> sealed::SlotSealed
    for &VrfTensor<D, Chip, Cluster, Slice, Element>
{
}
// `(operand0, operand1)` pairs, which is every ternary form: `(2.0, 3.0)`, `(&vrf, 3.0)`,
// `(Stash, 3.0)`.
impl<A: sealed::SlotSealed, B: sealed::SlotSealed> sealed::SlotSealed for (A, B) {}
// The builder, at any slot and either stash marker -- `IntoBranchedOperand` accepts it whatever it
// read, unlike `Plain` above, which is `NoStash` only.
impl<D: VeScalar, Mapping: M, Reg: RegPayload<D, Mapping>, StashMark, Slot> sealed::SlotSealed
    for VeOperandBuilder<D, Mapping, Reg, StashMark, Slot>
{
}
// A per-group operand that is already built, which `IntoGroupTernaryOperand` passes through.
impl<Mapping: M> sealed::SlotSealed for GroupTernaryOperand<Mapping> {}

// ============================================================================
// IntoBranchedOperand - every operand form, as the slots it fills
// ============================================================================

/// How a binary operand fills the slots of one VE pass: the three guarded immediate registers, the
/// guarded register-file port, and the guarded stash read of a [`BinaryBranchedOperand`].
///
/// Two forms reach it, and they are the same job at different depths:
///
/// - a **bare rhs** knows only *which slot* it drives (a constant an immediate register, a
///   [`VrfTensor`] the rf port, [`Stash`] the stash read) and takes its guard from whoever is filling
///   it in, which for a bare operand is [`Always`](TagGuard::all());
/// - a [`Branched`] layout brings slots that already carry their own guards, so the guard handed down
///   has nothing left to say.
///
/// ```ignore
/// tensor.vector_fxp(op, 16384i32)                       // reg0 = (Always, 16384)
/// tensor.vector_fp_binary(op, Stash)                    // rf   = (Always, Stash)
/// tensor.vector_fxp(op, Branched::imm(guard, 16384))     // reg0 = (guard, 16384)
/// ```
///
/// Sealed. `fill_slots` hands out the raw slots, so an outside impl could fill one twice or out of
/// order and walk straight past the builder's typestate. The operand forms are this crate's to
/// enumerate; see [`sealed::SlotSealed`](self) for what else that covers.
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::{IntoBranchedOperand, TagGuard, M, VeScalar};
/// struct MyOperand;
/// impl<D: VeScalar, Map: M> IntoBranchedOperand<D, Map> for MyOperand {
///     fn fill_slots(self, _operand: &mut furiosa_opt_std::prelude::BinaryBranchedOperand<D, Map>, _guard: TagGuard) {}
/// }
/// ```
pub trait IntoBranchedOperand<D: VeScalar, TargetMapping: M>: Sized + sealed::SlotSealed {
    /// Writes this operand's slot(s) into `operand`, applied where `guard` matches.
    fn fill_slots(self, operand: &mut BinaryBranchedOperand<D, TargetMapping>, guard: TagGuard);

    /// The slot layout of one VE pass. A bare operand is unconditional; a [`Branched`] layout carries
    /// the guards the kernel wrote.
    fn into_branched_operand(self) -> BinaryBranchedOperand<D, TargetMapping> {
        let mut operand = BinaryBranchedOperand::default();
        self.fill_slots(&mut operand, TagGuard::all());
        operand
    }
}

/// An `i32` constant drives an immediate register.
impl<Mapping: M> IntoBranchedOperand<i32, Mapping> for i32 {
    fn fill_slots(self, operand: &mut BinaryBranchedOperand<i32, Mapping>, guard: TagGuard) {
        assert_fill(operand.fill_imm(ImmSlot::Reg0, guard, self));
    }
}

/// An `f32` constant drives an immediate register.
impl<Mapping: M> IntoBranchedOperand<f32, Mapping> for f32 {
    fn fill_slots(self, operand: &mut BinaryBranchedOperand<f32, Mapping>, guard: TagGuard) {
        assert_fill(operand.fill_imm(ImmSlot::Reg0, guard, self));
    }
}

/// A VRF register drives the rf port, transposed to the op's tensor shape. VRF is read-many, so the
/// same register may feed several ops.
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, Mapping: M> IntoBranchedOperand<D, Mapping>
    for &VrfTensor<D, Chip, Cluster, Slice, Element>
{
    fn fill_slots(self, operand: &mut BinaryBranchedOperand<D, Mapping>, guard: TagGuard) {
        assert_fill(operand.fill_rf(guard, RfPort::External(self.inner.transpose::<Mapping>(true))));
    }
}

/// [`Stash`] reads the stash on the rf port: `RfPort::Stash` in place of a VRF tensor, since the
/// stash *is* the value.
impl<D: VeScalar, Mapping: M> IntoBranchedOperand<D, Mapping> for Stash {
    fn fill_slots(self, operand: &mut BinaryBranchedOperand<D, Mapping>, guard: TagGuard) {
        assert_fill(operand.fill_rf(guard, RfPort::Stash));
    }
}

/// An operand already in slot form: the builder's slots carry their own guards, so the guard handed
/// down has nothing left to say. Only a builder whose registers hold a bare `D` is a *binary* operand;
/// a ternary one goes through [`IntoTernaryOperand`] instead.
impl<D: VeScalar, Mapping: M, StashMark, Slot> IntoBranchedOperand<D, Mapping>
    for VeOperandBuilder<D, Mapping, D, StashMark, Slot>
where
    D: RegPayload<D, Mapping, Port = RfPort<D, Mapping>>,
{
    fn fill_slots(self, operand: &mut BinaryBranchedOperand<D, Mapping>, _guard: TagGuard) {
        *operand = self.operand;
    }
}

// ============================================================================
// IntoTernaryOperand - the ternary operand forms
// ============================================================================

/// How a ternary operand fills a [`TernaryBranchedOperand`]. Being a separate trait is what makes a
/// binary rhs unusable for a ternary op: `vector_fp_ternary(op, 2.0)` finds no impl, so `operand1`
/// can never be silently missing.
///
/// Every slot carries its own `operand1`, mirroring the hardware: `operand1` is selected per branch
/// just like `operand0`.
///
/// ```ignore
/// tensor.vector_fp_ternary(op, (2.0f32, 3.0f32))                    // reg0 = (Always, (2.0, 3.0))
/// tensor.vector_fp_ternary(op, (Stash, 1.0f32))                     // rf   = (Always, (Stash, 1.0))
/// tensor.vector_fp_ternary(op, Branched::imm(guard, (2.0, 3.0)))     // reg0 = (guard, (2.0, 3.0))
/// ```
///
/// Negative: a binary rhs on a ternary op has no impl, which is the missing-`operand1` compile error.
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::{IntoTernaryOperand, M, VeScalar};
/// use furiosa_mapping::Broadcast;
/// fn ternary_operand<D: VeScalar, Map: M, Op: IntoTernaryOperand<D, Map>>(_op: Op) {}
/// ternary_operand::<f32, Broadcast<1>, _>(2.0f32);
/// ```
pub trait IntoTernaryOperand<D: VeScalar, TargetMapping: M>: Sized + sealed::SlotSealed {
    /// Writes this operand's slot(s) into `operand`, applied where `guard` matches.
    fn fill_ternary_slots(self, operand: &mut TernaryBranchedOperand<D, TargetMapping>, guard: TagGuard);

    /// The ternary slot layout of one VE pass. A bare operand is unconditional; a [`Branched`] layout
    /// carries the guards the kernel wrote.
    fn into_ternary_operand(self) -> TernaryBranchedOperand<D, TargetMapping> {
        let mut operand = TernaryBranchedOperand::default();
        self.fill_ternary_slots(&mut operand, TagGuard::all());
        operand
    }
}

/// `(operand0, operand1)` constants drive one immediate register carrying both.
impl<Mapping: M> IntoTernaryOperand<f32, Mapping> for (f32, f32) {
    fn fill_ternary_slots(self, operand: &mut TernaryBranchedOperand<f32, Mapping>, guard: TagGuard) {
        assert_fill(operand.fill_imm(ImmSlot::Reg0, guard, self));
    }
}

/// A VRF `operand0` drives the rf port, which carries this branch's `operand1` with it.
impl<Chip: M, Cluster: M, Slice: M, Element: M, Mapping: M> IntoTernaryOperand<f32, Mapping>
    for (&VrfTensor<f32, Chip, Cluster, Slice, Element>, f32)
{
    fn fill_ternary_slots(self, operand: &mut TernaryBranchedOperand<f32, Mapping>, guard: TagGuard) {
        let (vrf, operand1) = self;
        let port = RfPort::External(vrf.inner.transpose::<Mapping>(true));
        assert_fill(operand.fill_rf(guard, (port, operand1)));
    }
}

/// A stash `operand0` reads the rf port and carries this branch's `operand1`.
impl<Mapping: M> IntoTernaryOperand<f32, Mapping> for (Stash, f32) {
    fn fill_ternary_slots(self, operand: &mut TernaryBranchedOperand<f32, Mapping>, guard: TagGuard) {
        let (_, operand1) = self;
        assert_fill(operand.fill_rf(guard, (RfPort::Stash, operand1)));
    }
}

/// A ternary operand already in slot form: the builder's registers hold `(operand0, operand1)` pairs,
/// which is exactly what a ternary slot carries, so the layout transfers as it stands.
impl<Mapping: M, StashMark, Slot> IntoTernaryOperand<f32, Mapping>
    for VeOperandBuilder<f32, Mapping, (f32, f32), StashMark, Slot>
{
    fn fill_ternary_slots(self, operand: &mut TernaryBranchedOperand<f32, Mapping>, _guard: TagGuard) {
        *operand = self.operand;
    }
}

// ============================================================================
// IntoGroupOperand - Ergonomic operand conversion for VectorTensorPair
// ============================================================================

/// Optional per-group operand for VectorTensorPair operations. `None` skips the operation for
/// that group; `Some(operand)` applies it.
pub type GroupOperand<D, Mapping> = Option<BinaryBranchedOperand<D, Mapping>>;

/// Trait for converting a per-group operand into a [`GroupOperand`]. Accepts an `i32` or `f32`
/// constant, a `&VrfTensor`, or `()` to skip the operation for this group (`None`).
///
/// Neither a [`Branched`] layout nor a [`Stash`] read is among them: a pair op spends both of a pass's
/// guards on its two groups, and runs no stash read. Having no impl is what says so, at the kernel
/// rather than deeper down; the two `compile_fail` blocks below pin it.
///
/// A constant is accepted, whichever scalar the cluster takes:
/// ```
/// use furiosa_opt_std::prelude::{IntoGroupOperand, M, VeScalar};
/// use furiosa_mapping::Broadcast;
/// fn group_operand<D: VeScalar, Map: M, Op: IntoGroupOperand<D, Map>>(_op: Op) {}
/// group_operand::<i32, Broadcast<1>, _>(10i32);
/// group_operand::<f32, Broadcast<1>, _>(1.0f32);
/// group_operand::<f32, Broadcast<1>, _>(()); // skips this group
/// ```
/// Negative: a stash read, bare. Spelled with `i32`, where the other forms above do compile, so what
/// this pins is the stash rule and not a missing scalar.
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::{IntoGroupOperand, Stash, M, VeScalar};
/// use furiosa_mapping::Broadcast;
/// fn group_operand<D: VeScalar, Map: M, Op: IntoGroupOperand<D, Map>>(_op: Op) {}
/// group_operand::<i32, Broadcast<1>, _>(Stash);
/// ```
/// Negative: a [`Branched`] layout, which a pair op has no room for and the translator no case for.
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::{Branched, IntoGroupOperand, TagGuard, M, VeScalar};
/// use furiosa_mapping::Broadcast;
/// fn group_operand<D: VeScalar, Map: M, Op: IntoGroupOperand<D, Map>>(_op: Op) {}
/// group_operand::<f32, Broadcast<1>, _>(Branched::imm::<f32, Broadcast<1>, _>(TagGuard::all(), 1.0));
/// ```
pub trait IntoGroupOperand<D: VeScalar, Mapping: M>: sealed::SlotSealed {
    /// Converts into a [`GroupOperand`]. `None` skips the operation for this group.
    fn into_group_operand(self) -> GroupOperand<D, Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for () {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        None
    }
}

/// An `i32` constant applies to the group it is passed as: `vector_fxp(op, 10, 20)` drives each
/// group with its own immediate.
///
/// Each form below is spelled out rather than blanketed over [`IntoBranchedOperand`], which would
/// overlap the `()` impl: coherence will not rule out `(): IntoBranchedOperand` on the strength of
/// there being no such impl today..
impl<Mapping: M> IntoGroupOperand<i32, Mapping> for i32 {
    fn into_group_operand(self) -> GroupOperand<i32, Mapping> {
        Some(self.into_branched_operand())
    }
}

/// An `f32` constant, the float clusters' peer of the `i32` form above.
impl<Mapping: M> IntoGroupOperand<f32, Mapping> for f32 {
    fn into_group_operand(self) -> GroupOperand<f32, Mapping> {
        Some(self.into_branched_operand())
    }
}

/// A VRF register drives the rf port. It is read-many, so both groups may name the same one.
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, Mapping: M> IntoGroupOperand<D, Mapping>
    for &VrfTensor<D, Chip, Cluster, Slice, Element>
{
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        Some(self.into_branched_operand())
    }
}

// ============================================================================
// IntoGroupTernaryOperand - Ergonomic ternary operand conversion for VectorTensorPair
// ============================================================================

/// Optional per-group ternary operand for VectorTensorPair operations.
pub type GroupTernaryOperand<Mapping> = Option<TernaryBranchedOperand<f32, Mapping>>;

/// Trait for converting a per-group ternary operand into a [`GroupTernaryOperand`]. Accepts
/// `(operand0, operand1)` where `operand0` is a constant or `&vrf`, a `GroupTernaryOperand`
/// (pass-through), or `()` to skip the operation for this group.
///
/// `(Stash, c)` is excluded, by the `T: Plain` bound below: a pair op reads no stash.
pub trait IntoGroupTernaryOperand<Mapping: M>: sealed::SlotSealed {
    /// Converts into a [`GroupTernaryOperand`] with the specified mapping.
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<Mapping: M> IntoGroupTernaryOperand<Mapping> for () {
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping> {
        None
    }
}

/// A `GroupTernaryOperand` passes through.
impl<Mapping: M> IntoGroupTernaryOperand<Mapping> for GroupTernaryOperand<Mapping> {
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping> {
        self
    }
}

/// `(rhs, operand1)` applies to this group. Spelled out rather than blanketed over
/// [`IntoBranchedOperand`] for the same coherence reason as [`IntoGroupOperand`].
impl<T, Mapping: M> IntoGroupTernaryOperand<Mapping> for (T, f32)
where
    (T, f32): IntoTernaryOperand<f32, Mapping>,
    // `T: Plain` keeps `(Stash, c)` out: a pair op has no stash to read. See [`IntoGroupOperand`].
    T: Plain,
{
    fn into_group_ternary_operand(self) -> GroupTernaryOperand<Mapping> {
        Some(self.into_ternary_operand())
    }
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::Broadcast;

    use super::*;
    use crate::engine::vector::branch::BitReq::{Ignore, One, Zero};
    use crate::engine::vector::branch::{ExecutionId, VeOperandLayout};
    use crate::tensor::Tensor;

    type Map = Broadcast<1>;

    /// The execution id `RAW` names. Every id in these tests is a literal, so the range check is the
    /// compiler's and there is nothing to unwrap.
    fn id<const RAW: u8>() -> ExecutionId {
        ExecutionId::new::<RAW>()
    }

    // Every test below opens at a `Branched` constructor the way kernel code does. Only the annotated
    // result type is spelled out, standing in for the op that would otherwise pin `Mapping`; `Reg`
    // (and with it the arity) comes from the argument each slot is given.

    // A guard on bit 3 alone, which is what a group id means to the hardware. Spelled out rather than
    // built through `TagGuard::group` so the bit these rest on is visible.
    const GROUP0: TagGuard = TagGuard::matches([Ignore, Ignore, Ignore, Zero]);
    const GROUP1: TagGuard = TagGuard::matches([Ignore, Ignore, Ignore, One]);

    /// The three immediate slots and whether the rf slot is filled, for asserting which ones a pass
    /// drives. Generic over the register payload rather than the scalar, so one helper reads a binary
    /// layout's `D` and a ternary one's `(D, D)`.
    fn slots<Reg: Copy, Port>(operand: &BranchedOperand<Reg, Port>) -> ([Option<(TagGuard, Reg)>; 3], bool) {
        let regs = operand
            .reg_slots()
            .map(|slot| slot.as_ref().map(|(guard, v)| (*guard, *v)));
        (regs, operand.rf_slot().is_some())
    }

    /// One immediate register: `reg0` gated `Always`, every other slot left empty, so the pass drives
    /// exactly one register. A scalar argument is what makes the operand binary.
    #[test]
    fn one_immediate_register_leaves_the_rest_unused() {
        let operand: BinaryBranchedOperand<f32, Map> = Branched::imm(TagGuard::all(), 2.0f32).into_branched_operand();

        let ([reg0, reg1, reg2], has_rf) = slots(&operand);
        assert_eq!(reg0, Some((TagGuard::all(), 2.0f32)));
        assert!(reg1.is_none() && reg2.is_none() && !has_rf);

        // Same path for an i32 op: the literal's type is the only thing that differs.
        let operand: BinaryBranchedOperand<i32, Map> =
            Branched::imm(TagGuard::all(), 0x777f_i32).into_branched_operand();

        let ([reg0, reg1, reg2], has_rf) = slots(&operand);
        assert_eq!(reg0, Some((TagGuard::all(), 0x777f_i32)));
        assert!(reg1.is_none() && reg2.is_none() && !has_rf);
    }

    /// A `&VrfTensor` argument sends the slot to the rf port instead of an immediate: the register is
    /// transposed to the op's tensor shape and the three immediates stay empty. The port carrying the
    /// register's data is what distinguishes it from a stash read on the same slot.
    #[test]
    fn register_fills_the_rf_port_and_leaves_the_rest_unused() {
        let vrf: VrfTensor<i32, Map, Map, Map, Map> = VrfTensor::from_parts(Tensor::splat(7));
        let operand: BinaryBranchedOperand<i32, Map> = Branched::rf(TagGuard::all(), &vrf).into_branched_operand();

        let (rf_guard, port) = operand.rf_slot().as_ref().unwrap();
        let RfPort::External(rf_data) = port else {
            panic!("a register read is `External`")
        };
        assert_eq!(*rf_guard, TagGuard::all());
        assert_eq!(rf_data.clone().into_vec(), vec![7]);
        assert!(!operand.reads_stash());
        assert!(operand.reg_slots().iter().all(|slot| slot.is_none()));
    }

    /// A guard on bit 3 alone is how a slot is scoped to one group, and it needs no separate path: it
    /// is a pattern like any other.
    ///
    /// The rf half reads a register rather than the stash, deliberately. A bit-3-only guard on a
    /// *stash* read is the one combination lowering refuses -- it cannot be told apart from a zip's
    /// cross-group cache read -- so building one here would pin a shape a kernel cannot compile.
    #[test]
    fn a_bit3_guard_gates_its_slot_on_that_group() {
        let operand: BinaryBranchedOperand<f32, Map> = Branched::imm(GROUP0, 2.0f32).into_branched_operand();

        let ([reg0, _, _], has_rf) = slots(&operand);
        let (reg0_guard, reg0_value) = reg0.unwrap();
        assert_eq!(reg0_value, 2.0);
        assert!(reg0_guard.admits(id::<0>()) && !reg0_guard.admits(id::<0b1000>()));
        assert!(!has_rf);

        // The rf slot takes such a guard the same way, group 1 this time.
        let vrf: VrfTensor<f32, Map, Map, Map, Map> = VrfTensor::from_parts(Tensor::splat(9.0));
        let operand: BinaryBranchedOperand<f32, Map> = Branched::rf(GROUP1, &vrf).into_branched_operand();
        let Some((rf_guard, RfPort::External(_))) = operand.rf_slot() else {
            panic!("a register read is `RfPort::External`")
        };
        assert!(rf_guard.admits(id::<0b1000>()) && !rf_guard.admits(id::<0>()));
        assert!(operand.reg_slots().iter().all(|slot| slot.is_none()));
    }

    /// A pair argument makes the operand ternary: every slot then carries its own `operand1`,
    /// mirroring the hardware's per-branch selection. Nothing spells the arity out.
    #[test]
    fn a_pair_argument_makes_the_slots_ternary() {
        let ternary: TernaryBranchedOperand<f32, Map> =
            Branched::imm(TagGuard::all(), (2.0f32, 3.0f32)).into_ternary_operand();

        let ([reg0, _, _], has_rf) = slots(&ternary);
        let (guard, (operand0, operand1)) = reg0.unwrap();
        assert_eq!(guard, TagGuard::all());
        assert_eq!((operand0, operand1), (2.0, 3.0));
        assert!(!has_rf);

        // The ternary rf slot pairs its read with `operand1`, so `(Stash, c)` fills one slot.
        let ternary: TernaryBranchedOperand<f32, Map> =
            Branched::rf::<f32, Map, (f32, f32), _>(TagGuard::all(), (Stash, 1.0f32)).into_ternary_operand();
        assert!(ternary.reads_stash());
        let Some((guard, (RfPort::Stash, operand1))) = ternary.rf_slot() else {
            panic!("a stash read is `RfPort::Stash`")
        };
        assert_eq!(*guard, TagGuard::all());
        assert_eq!(*operand1, 1.0);
        assert!(ternary.reg_slots().iter().all(|slot| slot.is_none()));
    }

    /// Two zipped groups whose `operand1` differs fit in one pass by taking a register each: the
    /// hardware selects `operand1` per branch, so nothing has to be shared.
    #[test]
    fn per_branch_ternary_operands_fill_one_register_each() {
        let ternary: TernaryBranchedOperand<f32, Map> = Branched::imm(GROUP0, (2.0f32, 1.0f32))
            .imm(GROUP1, (3.0f32, 2.0f32))
            .into_ternary_operand();

        let ([reg0, reg1, reg2], has_rf) = slots(&ternary);

        let (group0_guard, (operand0, operand1)) = reg0.unwrap();
        assert_eq!((operand0, operand1), (2.0, 1.0));
        assert!(group0_guard.admits(id::<0>()) && !group0_guard.admits(id::<0b1000>()));

        let (group1_guard, (operand0, operand1)) = reg1.unwrap();
        assert_eq!((operand0, operand1), (3.0, 2.0));
        assert!(group1_guard.admits(id::<0b1000>()) && !group1_guard.admits(id::<0>()));

        assert!(reg2.is_none() && !has_rf);
    }

    /// `imm` takes the next immediate, so successive calls fill densely and in call order, leaving no
    /// gap. That order is the whole of what the API promises about placement -- which physical
    /// register a value lands in is command generation's choice -- so it is the claim worth pinning.
    #[test]
    fn regs_fill_densely_in_call_order() {
        let first = TagGuard::matches([One, Ignore, Ignore, Ignore]);
        let second = TagGuard::matches([Zero, One, Ignore, Ignore]);
        let operand: BinaryBranchedOperand<f32, Map> =
            Branched::imm(first, 1.0f32).imm(second, 2.0f32).into_branched_operand();

        let ([reg0, reg1, reg2], has_rf) = slots(&operand);
        assert!(
            matches!(reg0, Some((g, v)) if g == first && v == 1.0),
            "first call, first slot"
        );
        assert!(
            matches!(reg1, Some((g, v)) if g == second && v == 2.0),
            "second call, second slot"
        );
        assert!(reg2.is_none(), "the third is untouched");
        assert!(!has_rf);
    }

    /// All three immediates plus the rf port: the most one pass can drive, and the shape the typestate
    /// stops one call short of.
    #[test]
    fn three_immediates_and_the_rf_port_fill_the_pass() {
        let vrf: VrfTensor<f32, Map, Map, Map, Map> = VrfTensor::from_parts(Tensor::splat(9.0));
        let operand: BinaryBranchedOperand<f32, Map> =
            Branched::imm(TagGuard::matches([One, Ignore, Ignore, Ignore]), 1.0f32)
                .imm(TagGuard::matches([Zero, One, Ignore, Ignore]), 2.0f32)
                .imm(TagGuard::matches([Zero, Zero, One, Ignore]), 3.0f32)
                .rf(TagGuard::matches([Zero, Zero, Zero, One]), &vrf)
                .into_branched_operand();

        let ([reg0, reg1, reg2], has_rf) = slots(&operand);
        assert!(matches!(reg0, Some((_, v)) if v == 1.0));
        assert!(matches!(reg1, Some((_, v)) if v == 2.0));
        assert!(matches!(reg2, Some((_, v)) if v == 3.0));
        assert!(has_rf);
    }

    /// The stash slot flips the builder to `WithStash`, so the op's `StashTransition` consumes the
    /// stash once.
    #[test]
    fn builder_stash_slot_reads_the_stash() {
        let operand: BinaryBranchedOperand<f32, Map> =
            Branched::imm(TagGuard::not_matches([One, Ignore, Ignore, Ignore]), 1.0f32)
                .rf(TagGuard::matches([One, Ignore, Ignore, Ignore]), Stash)
                .into_branched_operand();

        assert!(operand.reads_stash());
        let Some((stash_guard, RfPort::Stash)) = operand.rf_slot() else {
            panic!("a stash read is `RfPort::Stash`")
        };
        assert!(stash_guard.admits(id::<0b0001>()));
    }
}
