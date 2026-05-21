# Intra-Slice Chain

The Intra-Slice Chain performs elementwise, binary, and intra-slice reduce operations on each slice's data independently.
It handles post-contraction processing such as activation and normalization.
For example, computing \\(\operatorname{sigmoid}(XW + b)\\) runs \\(XW\\) on the [Contraction Engine](../contraction-engine/index.md), and the addition plus sigmoid activation in the Intra-Slice Chain.

## Interface

The example below applies a fixed-point bias, runs sigmoid on the float path (narrowing to 4-way and widening back), and finishes with a ReLU clip.
It computes \\(output[a, b] = \max(\operatorname{sigmoid}(input[a, b] + 100), 0)\\).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn staged_pipeline<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
    .vector_init()
    .vector_intra_slice_tag(TagMode::Zero)
    // input + 100
    .vector_fxp(FxpBinaryOp::AddFxp, 100)
    // i32 → f32 (fixed-point, int_width = 31)
    .vector_fxp_to_fp(31)
    // Way8 → Way4 for the float path
    .vector_narrow_trim::<m![A % 2 # 4]>()
    // sigmoid(input + 100)
    .vector_fp_unary(FpUnaryOp::Sigmoid)
    // Way4 → Way8
    .vector_widen_pad::<m![A % 2 # 8]>()
    // f32 → i32
    .vector_fp_to_fxp(31)
    // max(sigmoid(input + 100), 0)
    .vector_clip(ClipBinaryOpI32::Max, 0)
    .vector_final()
}
```

### Pipeline

The chain starts (as shown above) with `vector_intra_slice_tag()` (right after [`vector_init()`](./index.md#interface) or on the inter-slice reducer's output), or with `vector_intra_slice_unzip()` directly after `vector_init()` for [Pair Mode](#pair-mode) (see below).

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_intra_slice_tag}}

{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_intra_slice_unzip}}
```

After entry, the chain steps through the pipeline stages below in a fixed order; software chains the relevant ones and skips the rest, as the example skips `Logic`, `FpDiv`, and `Filter`.
Each row lists the stage's position in the chain (`#`), its name (`Stage`), the API method that triggers it (`Method`), the way it runs in (`Way`, either 8 or 4 elements per cycle), and whether it accepts an operand (`Operand`).
The type system enforces every stage transition at compile time, so methods become callable only after the preceding chain reaches a compatible state.
Per-stage detail is in [Stages](#stages) below.

| # | Stage | Method | Way | Operand | → Inter-Slice Reducer |
|---|-------|--------|-----|-------------|------------------------|
| 1 | Entry | `vector_intra_slice_tag()` | Way8 | – | – |
| 2 | Logic | `vector_logic()` | Way8 | yes | yes |
| 3 | Fxp | `vector_fxp()` | Way8 | yes | yes |
| 4 | FxpToFp | `vector_fxp_to_fp()` | Way8 | – | yes |
| 5 | Narrow | `vector_narrow_split()` / `vector_narrow_trim()` | Way8 → Way4 | – | – |
| 6 | Float | `vector_fp_unary/binary/ternary()` | Way4 | yes | – |
| 7 | IntraSliceReduce | `vector_intra_slice_reduce()` | Way4 | – | – |
| 8 | FpDiv | `vector_fp_div()` | Way4 | yes | – |
| 9 | Widen | `vector_widen_concat()` / `vector_widen_pad()` | Way4 → Way8 | – | yes |
| 10 | FpToFxp | `vector_fp_to_fxp()` | Way8 | – | yes |
| 11 | Clip | `vector_clip()` | Way8 | yes | yes |
| 12 | Filter | `vector_filter()` | Way8 | – | – |

Stages run either 8-way (8 elements per cycle) or 4-way (4 elements per cycle).
The floating-point cluster runs 4-way to amortize its half-throughput ALUs against the rest of the chain.
A chain that uses the float path therefore enters 8-way, calls `Narrow` (`vector_narrow_split` or `vector_narrow_trim`) before the float stages, and calls `Widen` (`vector_widen_concat` or `vector_widen_pad`) afterward to return to 8-way.
The example does exactly this: `vector_narrow_trim` then `vector_fp_unary(Sigmoid)` then `vector_widen_pad`.

<a id="transitioning-to-the-inter-slice-reducer"></a>

The chain exits via `vector_final()` (as in the example) or `vector_inter_slice_reduce()` (handing off to the [Inter-Slice Reducer](./inter-slice-reducer.md)).
Both exits require 8-way, so any active 4-way stage must pass through `Widen` first.

### Operands

Binary and ternary ops have two slot kinds: a **stream** (the running tensor, i.e., the `self` of the method chain, fixed by the chain) and one or two **operands** (the extra inputs).
Each operand comes from one of three sources.

| Source | Example | Description |
|--------|---------|-------------|
| Constant | `100`, `2.5f32` | Scalar broadcast to all elements |
| VRF tensor | `VeRhs::vrf(&vrf_tensor)` | Pre-loaded via `.to_vrf()` before entering the Vector Engine |
| Stash | `Stash` | Snapshot of an earlier chain step, described below |

Ternary ops (`FmaF`) take a pair `(operand0, operand1)`.

The same op method picks up different sources by the argument type:

```rust,ignore
.vector_fxp(FxpBinaryOp::AddFxp, 100)                // operand from constant
.vector_fxp(FxpBinaryOp::MulInt, VeRhs::vrf(&vrf))   // operand from VRF tensor
.vector_clip(ClipBinaryOpI32::Max, Stash)            // operand from stash (set earlier)
```

The `Stash` source comes from `vector_stash()`, which snapshots the running tensor so a later binary or ternary op can read it back as the `Stash` operand.
The typical use is a residual or skip-connection like `max(f(x), x)`, where the original `x` must survive across intermediate stages.
Call `vector_stash()` at any `Stashable` stage (`Branch`, `Logic`, `Fxp`, `Narrow`, `Fp`, `FpDiv`, `Clip`); the snapshot stays live until the Tensor Unit invocation ends and feeds any later binary or ternary call that takes `Stash`.
The slot is single-use (a second `vector_stash()` is a compile-time error) and typed, so an `f32` stash only feeds `f32` ops.
The mapping follows the running tensor, so a stash taken before `Narrow` is still usable after `Widen`.
Stash is unavailable in [Pair Mode](#pair-mode), and the `IntraFirst` transition to the [inter-slice reducer](./inter-slice-reducer.md) drops it, so anything stashed before `vector_inter_slice_reduce()` is gone afterward.

An **argument mode** then picks which slots hold the stream versus the operands so the same op can compute, e.g., `stream + operand` or `operand - stream`.
For example, `BinaryArgMode::Mode10` swaps the slots so `SubFxp` computes `operand - stream`:

```rust,ignore
.vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7)  // computes 7 - stream
```

`BinaryArgMode` picks which two of a binary op's slots are stream vs operand (unary ops have no mode and always run as `op(stream)`):

<a id="argument-modes"></a>

| BinaryArgMode | Slots | Computation |
|---------------|-------|-------------|
| `Mode00` | stream / stream | `op(stream, stream)` |
| `Mode01` | stream / operand | `op(stream, operand)` (default) |
| `Mode10` | operand / stream | `op(operand, stream)` |
| `Mode11` | operand / operand | `op(operand, operand)` |

`TernaryArgMode` does the same for ternary ops:

| TernaryArgMode | Slots | Computation |
|----------------|-------|-------------|
| `Mode012` | stream / operand0 / operand1 | `op(stream, operand0, operand1)` (default) |
| `Mode002` | stream / stream / operand1 | `op(stream, stream, operand1)` |
| `Mode102` | operand0 / stream / operand1 | `op(operand0, stream, operand1)` |
| `Mode112` | operand0 / operand0 / operand1 | `op(operand0, operand0, operand1)` |
| `Mode020` | stream / operand1 / stream | `op(stream, operand1, stream)` |
| `Mode021` | stream / operand1 / operand0 | `op(stream, operand1, operand0)` |
| `Mode120` | operand0 / operand1 / stream | `op(operand0, operand1, stream)` |

### Pair Mode

Pair mode runs the chain on a tensor whose elements split into two interleaved groups, so an op can relate the two groups (e.g., pair-wise add, asymmetric scale).
Entry is `vector_intra_slice_unzip()` directly after `vector_init()`, applied to a collected tensor that carries a 2-way grouping axis.
Starting the chain with `vector_intra_slice_unzip()` precludes the [Filter](#filter) stage downstream.
Under the hood, `vector_intra_slice_unzip()` uses `TagMode::AxisToggle` to derive each element's `GroupId` from the 2-way grouping axis.

The flow has four steps:
1. `vector_intra_slice_unzip()` splits the input into two parallel streams (group 0 and group 1).
2. The chain runs through stages with both groups in lock-step (the **paired** phase).
3. A `_zip` op fuses the two streams back into one (the **merged** phase).
4. The merged stream continues to `vector_final()` like a normal chain.

Stages during the paired phase fall into two flavors:
- **Common stages** (`vector_fxp_to_fp`, `vector_narrow_split`, `vector_widen_concat`, `vector_fp_to_fxp`) act on both groups uniformly.
  `vector_narrow_trim` and `vector_widen_pad` are not available on pairs; use the `_split` / `_concat` variants instead.
- **Per-group ops** take one argument per group:
  - Binary and ternary (`vector_fxp`, `vector_fp_binary`, `vector_fp_ternary`, `vector_clip`, etc.) accept `()` on a side to skip it, or different operands on each side.
  - Unary (`vector_fp_unary`) is the exception: it takes flags `(op, group0_apply, group1_apply)` and `false` skips that group.

Pair mode reinterprets [`BinaryArgMode`](#argument-modes) depending on the op: per-group ops (`vector_fxp_with_mode`, `vector_fp_binary_with_mode`, etc.) apply the mode inside each group independently (`0` is that group's stream, `1` is that group's operand), while `_zip` ops (`vector_fxp_zip_with_mode`, etc.) take the two slots as the two grouped streams (`0` is Group 0's stream, `1` is Group 1's stream):

| `_zip` `BinaryArgMode` | Slots | Computation |
|------------------------|-------|-------------|
| `Mode00` | group0 / group0 | `op(group0, group0)` |
| `Mode01` | group0 / group1 | `op(group0, group1)` (default) |
| `Mode10` | group1 / group0 | `op(group1, group0)` |
| `Mode11` | group1 / group1 | `op(group1, group1)` |

Pair-mode constraints:
- `stash()` and `filter()` are unavailable throughout pair mode (both paired and merged phases).
- Before `_zip` (the paired phase), the chain cannot transition to the [inter-slice reducer](./inter-slice-reducer.md), since `vector_inter_slice_reduce()` is not available on per-group tensors. After `_zip` (the merged phase), the result is `Commitable` again and can call `vector_inter_slice_reduce()` if the current stage supports the transition.
- ALU usage is shared across the two groups: an ALU used in either group counts as consumed for both.

## Stages

Within a stage, each ALU runs at most once per Tensor Unit invocation.
This matters mainly in `Logic`, `Fxp`, `Fp`, and `Clip`, where multiple operators share a stage-local ALU pool.
For example, `tanh(sqrt(x))` cannot fit in a single Tensor Unit invocation because both `tanh` and `sqrt` consume the `FpFpu` ALU.

### Tag

The Tag stage is the chain's entry point and assigns each 32-bit element in a flit a 4-bit `Tag` (0-15), which later stages use to apply conditional operations.
Bit 3 (the MSB) is `GroupId`, used by Filter and pair mode to split elements into Group 0 / Group 1.
Bits 0..2 are general-purpose flag bits filled by comparison results.

The `TagMode` selects how the 4 bits are computed for each element:

| `TagMode` | How each tag bit is filled |
|-----------|----------------------------|
| `Zero` | All four bits are 0. Every element has tag = 0. |
| `AxisToggle { axis }` | Bit 3 (`GroupId`) = `axis_index % 2` along `axis`. Bits 0..2 stay 0. |
| `Comparison([cmp0, cmp1, cmp2, cmp3])` | For each element `x`, bit `i` = `1` iff `cmp_i(x)` holds. The four comparisons see the same `x` and must match its dtype (all `InputCmpI32` or all `InputCmpF32`). Each `cmp_i` independently picks (op, boundary) from `InputCmp` (`Less`, `Greater`, `Equal`, `LessUnsigned`, `GreaterUnsigned`, `True`, `False`). |
| `ValidCount` | Bits derived from the [Valid Count Generator](./vcg.md) output. |
| `Vrf` | Bits loaded from VRF, previously written by an earlier TuExec (enables tag reuse across invocations). |

For example, with `i32` data and `Comparison([Less{0}, Equal{5}, Greater{100}, True])`, an element `x = 7` yields bits `0/0/0/1` (LSB first), so its tag is `0b1000 = 8`.


Once tags are assigned, later binary and ternary ops can condition an operand on the `GroupId` MSB so different tag groups see different values.
`BinaryOperandTag::always(operand)` applies to all groups, `BinaryOperandTag::group(operand, GroupId::Zero)` applies only to group 0.
`TernaryOperandTag` is the ternary form.

### Logic Cluster

The Logic Cluster performs bitwise operations on `i32` or `f32` (bit-level).
It runs 8-way.

The stage exposes five ALU classes (`LogicAnd`, `LogicOr`, `LogicXor`, `LogicLshift`, `LogicRshift`), each runnable once per Tensor Unit invocation.
Operators sharing the same class cannot fuse into one invocation.

`i32` operations:

| Op | ALU | Note |
|----|-----|------|
| `BitAnd` | `LogicAnd` | bitwise and |
| `BitOr` | `LogicOr` | bitwise or |
| `BitXor` | `LogicXor` | bitwise xor |
| `LeftShift` | `LogicLshift` | logical left shift |
| `LogicRightShift` | `LogicRshift` | logical right shift |
| `ArithRightShift` | `LogicRshift` | arithmetic right shift |

`f32` operations:

| Op | ALU | Note |
|----|-----|------|
| `BitAnd` | `LogicAnd` | bitwise and on fp bit patterns |
| `BitOr` | `LogicOr` | bitwise or on fp bit patterns |
| `BitXor` | `LogicXor` | bitwise xor on fp bit patterns |

### Fxp Cluster

The Fxp Cluster performs integer and fixed-point arithmetic on `i32`.
It runs 8-way.

The stage exposes four ALU classes (`FxpAdd`, `FxpLshift`, `FxpMul`, `FxpRshift`), each runnable once per Tensor Unit invocation.
Operators sharing the same class cannot fuse into one invocation.

| Op | ALU | Note |
|----|-----|------|
| `AddFxp` | `FxpAdd` | wrapping add |
| `AddFxpSat` | `FxpAdd` | saturating add |
| `SubFxp` | `FxpAdd` | wrapping subtract |
| `SubFxpSat` | `FxpAdd` | saturating subtract |
| `LeftShift` | `FxpLshift` | logical left shift |
| `LeftShiftSat` | `FxpLshift` | saturating left shift |
| `MulFxp` | `FxpMul` | fixed-point multiply |
| `MulInt` | `FxpMul` | integer multiply |
| `LogicRightShift` | `FxpRshift` | logical right shift |
| `ArithRightShift` | `FxpRshift` | arithmetic right shift |
| `ArithRightShiftRound` | `FxpRshift` | arithmetic right shift with rounding |

The single-ALU rule rejects, for example, two ops that both target `FxpAdd`:

```rust,ignore
// PANICS: "FxpAdd is already in use"
input
    .vector_init()
    .vector_intra_slice_tag(TagMode::Zero)
    .vector_fxp(FxpBinaryOp::AddFxp, 10)    // uses FxpAdd
    .vector_fxp(FxpBinaryOp::MulInt, 2)     // uses FxpMul ✓
    .vector_fxp(FxpBinaryOp::SubFxp, 5)     // uses FxpAdd again ✗
    .vector_final()
```

### FxpToFp Conversion

The FxpToFp Conversion stage converts `i32` to `f32`.
The `int_width` parameter specifies the integer bit width for the conversion; `int_width = 31` is the standard `i32` ↔ `f32` conversion.

| Method | Effect |
|--------|--------|
| `vector_fxp_to_fp(int_width)` | convert `i32` stream to `f32` |

### Narrow

The `Narrow` stage switches 8-way to 4-way.
An 8-way packet carries 8 active elements (`Packet = m![... # 8]`) and a 4-way packet carries 4 (`Packet = m![... # 4]`).
Narrowing halves throughput on the float and reduce path, so the same logical tensor shape takes twice as many packets or Tensor Unit invocations.

| Method | Use When | Effect |
|--------|----------|--------|
| `vector_narrow_split()` | both halves contain real data | split one 8-way flit into a front-4 and back-4 packet, updating `Time` and `Packet` |
| `vector_narrow_trim()` | back 4 elements are already padding or irrelevant | keep only the front 4 elements |

Shape semantics:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, S = 64];

fn split_semantics<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![B], m![S # 16 / 4], m![S # 16 % 4], m![A % 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorNarrowTensor<'l, T, i32, m![1], m![B], m![S # 16 / 4], m![S # 16 % 4, A / 4 % 2], m![A % 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input.vector_narrow_split::<m![S # 16 % 4, A / 4 % 2], m![A % 4]>()
    // shape semantics: [T], [P] -> [T, P / 2], [P % 4]
}

fn trim_way4_semantics<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorNarrowTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input.vector_narrow_trim::<m![A % 2 # 4]>()
    // shape semantics: [T], [P] -> [T], [P = 4]
}
```

### Float Cluster

The Float Cluster provides unary, binary, and ternary floating-point operations on `f32`.
It runs 4-way, so the input must already have passed through `Narrow`.

It exposes five independent ALUs (`FpFma`, `FpFpu`, `FpExp`, `FpMul0`, `FpMul1`), each runnable once per Tensor Unit invocation.
This stage is where ALU planning matters most.

Unary ops:

| Op | ALU | Note |
|----|-----|------|
| `Exp` | `FpExp` | exponential |
| `NegExp` | `FpExp` | negative exponential |
| `Sqrt` | `FpFpu` | square root |
| `Tanh` | `FpFpu` | hyperbolic tangent |
| `Sigmoid` | `FpFpu` | sigmoid |
| `Erf` | `FpFpu` | error function |
| `Log` | `FpFpu` | natural logarithm |
| `Sin` | `FpFpu` | sine |
| `Cos` | `FpFpu` | cosine |

Binary ops:

| Op | ALU | Note |
|----|-----|------|
| `AddF` | `FpFma` | floating-point add |
| `SubF` | `FpFma` | floating-point subtract |
| `MulF(FpMulAlu::Mul0)` | `FpMul0` | multiply |
| `MulF(FpMulAlu::Mul1)` | `FpMul1` | multiply |
| `MulF(FpMulAlu::Fma)` | `FpFma` | multiply |
| `DivF` | `FpFpu` | division inside `Fp` stage |

Ternary ops:

| Op | ALU | Note |
|----|-----|------|
| `FmaF` | `FpFma` | fused multiply-add |

For example, to compute `exp(sqrt(((x + 1) * 2) * 3))`:
- `x1 = x + 1` via FpFma (`FpBinaryOp::AddF`)
- `x2 = x1 * 2` via FpMul0 (`FpBinaryOp::MulF(FpMulAlu::Mul0)`)
- `x3 = x2 * 3` via FpMul1 (`FpBinaryOp::MulF(FpMulAlu::Mul1)`)
- `x4 = sqrt(x3)` via FpFpu (`FpUnaryOp::Sqrt`)
- `x5 = exp(x4)` via FpExp (`FpUnaryOp::Exp`)

### IntraSliceReduce

The IntraSliceReduce stage reduces axes within a single slice.
It runs 4-way.
The stage uses a dedicated accumulator-tree ALU, so no user-selectable ALU is exposed.

| Data Type | Supported Ops |
|-----------|---------------|
| `i32` | `AddSat`, `Max`, `Min` |
| `f32` | `Add`, `Max`, `Min` |

See [Intra-Slice Reduce](./intra-slice-reduce.md) for details.

### FpDiv

The FpDiv stage performs floating-point division.
It runs 4-way.
The stage uses a dedicated floating-point divider, so no user-selectable ALU is exposed.

| Op | Note |
|----|------|
| `FpDivBinaryOp::DivF` | dedicated floating-point division |

### Widen

The `Widen` stage transitions from 4-way back to 8-way.
Later stages (`FpToFxp`, `Clip`, `Filter`, `Output`) then see 8-element packets again.

| Method | Use When | Effect |
|--------|----------|--------|
| `vector_widen_concat()` | reversing a prior `vector_narrow_split()` | merge two 4-way packets back into one 8-way flit |
| `vector_widen_pad()` | reversing a prior `vector_narrow_trim()` | pad a 4-way packet back to 8 elements with invalid fillers |

Shape semantics:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, S = 64];

fn concat_semantics<'l, const T: Tu>(
    input: VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![B], m![S # 16 / 4], m![A / 4 % 2], m![A % 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorWidenTensor<'l, T, i32, m![1], m![B], m![S # 16 / 4], m![1], m![A % 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input.vector_widen_concat::<m![1], m![A % 8]>()
    // shape semantics: [T, P / 2], [P % 4] -> [T], [P]
}

fn pad_way8_semantics<'l, const T: Tu>(
    input: VectorFpTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorWidenTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input.vector_widen_pad::<m![A % 2 # 8]>()
    // shape semantics: [T], [P] -> [T], [P # 8]
}
```

### FpToFxp Conversion

The FpToFxp Conversion stage converts `f32` back to `i32`.
The `int_width` parameter specifies the integer bit width.

| Method | Effect |
|--------|--------|
| `vector_fp_to_fxp(int_width)` | convert `f32` stream back to `i32` |

### Clip Cluster

The Clip Cluster performs clamping and comparison operations.
It runs 8-way.

The stage exposes three ALU classes (`ClipAdd`, `ClipMax`, `ClipMin`), each runnable once per Tensor Unit invocation.

`i32` operations:

| Op | ALU | Note |
|----|-----|------|
| `Min` | `ClipMin` | minimum |
| `Max` | `ClipMax` | maximum |
| `AbsMin` | `ClipMin` | absolute minimum |
| `AbsMax` | `ClipMax` | absolute maximum |
| `AddFxp` | `ClipAdd` | wrapping add |
| `AddFxpSat` | `ClipAdd` | saturating add |

`f32` operations:

| Op | ALU | Note |
|----|-----|------|
| `Min` | `ClipMin` | minimum |
| `Max` | `ClipMax` | maximum |
| `AbsMin` | `ClipMin` | absolute minimum |
| `AbsMax` | `ClipMax` | absolute maximum |
| `Add` | `ClipAdd` | floating-point add |

### Filter

The Filter stage applies an execution mask derived from `TagFilter` (matching on the `GroupId` MSB of each element's `Tag`) to filter output flits.
Available 8-way and `Standalone` context only.
The source `impl` lives on `VectorTensor` for any stage with `CanTransitionTo<Filter>`, which covers every intra-slice stage and `InterSliceReduce`.


### Output

The Output stage exits the Vector Engine pipeline.
The result can continue to the [Cast Engine](../cast-engine.md), [Transpose Engine](../transpose-engine.md), or [Commit Engine](../../moving-tensors/commit-engine.md).

## Examples

### `i32` Pipeline

A minimal `i32` chain that adds a constant after branching.
The Fxp stage runs 8-way, so no narrow or widen is needed.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2];

fn add_constant<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
}
```

### `f32` Pipeline

`vector_narrow_trim()` is the `Narrow` step that converts the tensor from 8-way to 4-way before the float operation.
`vector_widen_pad()` is the `Widen` step that converts back to 8-way afterward.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn sigmoid<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![A % 2 # 4]>() // Narrow: Way8 -> Way4
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_widen_pad::<m![A % 2 # 8]>() // Widen: Way4 -> Way8
        .vector_final()
}
```

### Single-Stream Argument Mode

`BinaryArgMode::Mode10` swaps the stream and operand positions, so `SubFxp` computes `operand - stream` (here, `7 - x`) rather than the default `stream - operand`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2];

fn bias_minus_x<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, 7) // compute 7 - x
        .vector_final()
}
```

### VRF Operand

Pre-loaded VRF data as an operand:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, N = 256];

fn vrf_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![N], m![A % 8]>,
    vrf: &VrfTensor<i32, m![1], m![B], m![A / 8], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![N], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, VeRhs::vrf(vrf))
        .vector_final()
}
```

### Stash on the Fp-Only Path

Stash at an early stage, then use it later in a Clip operation.
This implements `max(2 * x, x)`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn residual_max<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()                                        // enter VE
        .vector_intra_slice_tag(TagMode::Zero) // start the intra-slice path
        .vector_stash()                                       // save original x
        .vector_narrow_trim::<m![A % 2 # 4]>()                  // narrow to Way4
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2.0f32) // compute 2 * x
        .vector_widen_pad::<m![A % 2 # 8]>()                   // widen back to Way8
        .vector_clip(ClipBinaryOpF32::Max, Stash)             // max(2 * x, x)
        .vector_final()
}
```

### Stash on the Fxp-Only Path

Stash at an early stage, then use it later in a Clip operation.
This implements `max(x + bias, x)`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2];

fn stash_at_fxp<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()                                      // enter VE
        .vector_intra_slice_tag(TagMode::Zero) // start the intra-slice path
        .vector_stash()                                     // save original x
        .vector_fxp(FxpBinaryOp::AddFxp, 100)               // compute x + bias
        .vector_clip(ClipBinaryOpI32::Max, Stash)           // compute max(x + bias, x)
        .vector_final()
}
```

### Stash Across Narrow and Widen

Stash before narrowing, consume after widening.
This computes `max(sigmoid(x), x)`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2];

fn stash_across_narrow_widen<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()                                     // enter VE
        .vector_intra_slice_tag(TagMode::Zero) // start the intra-slice path
        .vector_stash()                                    // save x (Way8)
        .vector_narrow_trim::<m![A % 2 # 4]>()               // narrow to Way4
        .vector_fp_unary(FpUnaryOp::Sigmoid)               // compute sigmoid(x) in Way4
        .vector_widen_pad::<m![A % 2 # 8]>()                // widen back to Way8
        .vector_clip(ClipBinaryOpF32::Max, Stash)          // compute max(sigmoid(x), x)
        .vector_final()
}
```

### Pair Add

Zip two interleaved groups with integer add:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

fn pair_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}
```

### Pair Per-Side Preprocessing

Asymmetric preprocessing scales only group 0 before zip:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

fn pair_preprocess_one_side<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_fxp(FxpBinaryOp::MulInt, 10, ())   // group 0 only
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}
```

### Pair Float Pipeline with Zip

Both groups traverse the float path (narrow -> fp -> zip -> widen):

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, I = 2];

fn pair_fp_mul_zip<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![B], m![A % 2 # 4]>()        // both groups: Way8 -> Way4
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))   // group0 * group1 (Way4)
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()           // Way4 -> Way8
        .vector_final()
}
```

### Pair Per-Group Preprocessing

Apply a different operation to each group before zipping:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, I = 2];

fn pair_asymmetric_preprocess<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![B], m![A % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Exp, true, false)         // group 0: exp(x), group 1: skip
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))   // exp(group0) * group1
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
}
```

### Pair Zip Argument Mode

`BinaryArgMode::Mode10` swaps the two grouped streams when zipping:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, I = 2];

fn pair_sub_reverse<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![A / 2], m![I], m![A % 2 # 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![A / 2], m![1], m![A % 2 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_narrow_split::<m![B], m![A % 2 # 4]>()
        .vector_fp_zip_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10) // compute group1 - group0
        .vector_widen_concat::<m![1], m![A % 2 # 8]>()
        .vector_final()
}
```

## Performance

Throughput is full 8-way (8 elements per cycle) on the Logic, Fxp, and Clip clusters.
The Float cluster runs at 4-way, and the Narrow/Widen wrapping around the float path halves effective throughput in practice.

Latency adds one cycle per ALU used.
Operations spanning multiple ALUs accumulate their latencies.
For example, `exp(sqrt(x))` adds 2 cycles (FpFpu for sqrt plus FpExp for exp).
