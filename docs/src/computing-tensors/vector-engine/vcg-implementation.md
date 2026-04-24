# Valid Count Generator's Implementation

This document describes what the Valid Count Generator (VCG) hardware can express, independent of mapping expressions or tensor shapes.
VCG tags are consumed by the [Intra-Slice Reduce stage](./intra-slice-reduce.md) to exclude padding from reductions.
For how mapping expressions control VCG behavior (supported placements, constraints, and examples), see [Valid Count Generator's Interface](./vcg-interface.md).

## Data Model

Data flows into the VectorEngine as a stream of **flits** (packets).
Each flit contains 8 elements.
The VCG operates at the VectorEngine's input, tagging each 8-way flit with a valid count.
The 4-way halving and its valid count derivation are described in [Downstream: 4-Way Operations](#downstream-4-way-operations).

A flit is identified by two coordinates.
A **slice** corresponds to the Slice dimension in the mapping; a **time step** indexes sequential flits within a slice.

| Coordinate | Range | Meaning |
|------------|-------|---------|
| `s` (slice number) | `[0, num_slices)` | Which slice processes this flit |
| `t` (time step) | `[0, num_flits)` | Sequential position within a slice |

The VCG assigns a **valid count** (abbreviated `vc` in formulas and diagrams) to each flit:

$$\text{vc}(s, t) \in \{0, 1, \ldots, 8\}$$

Element `p` (where `p` is in `[0, 8)`) within flit `(s, t)` is valid if and only if `p < vc(s, t)`.
Valid elements always form a **contiguous prefix**; this is a fundamental hardware constraint.
The VCG cannot express "elements 0, 1, 3 are valid but 2 is not."

## Valid Count Formula

The VCG computes `vc(s, t)` through a pipeline of stages:

$$t \overset{\text{Sequencer}}{\longrightarrow} (c_0, c_1, \ldots, c_{k-1}) \overset{\text{Original Dims}}{\longrightarrow} \text{idx}(t) \overset{\text{Validity}}{\longrightarrow} \text{vc}(s, t)$$

1. **[Sequencer](#sequencer)**: The flat time index `t` is decomposed into counter values \\((c_0, c_1, \ldots, c_{k-1})\\) via mixed-radix decomposition.
2. **[Original Dimensions](#original-dimensions)**: Each counter is assigned to one of 4 dimensions (packet dim or gate dim 0-2). Per-dimension indices are computed as \\(\text{idx}(t) = \sum_{i} c_i \cdot \sigma _ i\\), where \\(\sigma_i\\) is the counter's stride.
3. **[Validity Decision](#validity-decision)**:
   - [Packet Dim](#packet-dim-packet-level-valid-count): produces a packet-level valid count \\(\text{packet\_vc}(t) = \min(\text{stride}_p,\; \max(0,\; V_p - \text{idx}_p(t)))\\).
   - [Gate Dims](#gate-dims-per-flit-binary-validity): each produces a binary gate \\(\text{gate}_d(s, t) \in \{0, 1\}\\) based on slice classification (below/boundary/above a threshold) and the per-dim index.

The final valid count combines these components:

$$\text{vc}(s, t) = \text{packet\_vc}(t) \times \text{gate}_0(s, t) \times \text{gate}_1(s, t) \times \text{gate}_2(s, t)$$

```text
vc(s,t) = packet_vc(t)  ×  gate_0(s,t)  ×  gate_1(s,t)  ×  gate_2(s,t)
          ───────────      ───────────      ───────────      ───────────
          packet dim       gate dim 0       gate dim 1       gate dim 2
          (count 0-8)      (gate 0/1)       (gate 0/1)       (gate 0/1)
```

- If **all** gates are open (= 1): the flit gets `packet_vc(t)` valid elements.
- If **any** gate is closed (= 0): `vc = 0` (entire flit is invalid, regardless of packet dim's count).

## VCG Configuration

Configuration is organized around two concepts: **counters** (which drive the sequencer) and **original dimensions** (which decide validity).
Counters produce a flit sequence; each dim uses its assigned counters to compute an index and decide validity.

The VCG is configured via the following parameters (each is explained in detail in subsequent sections):

| Field | Scope | Description |
|-------|-------|-------------|
| Counter limits \\(L_0 \ldots L_7\\) | per counter (up to 8) | Sequencer counter limits |
| Original dim assignment | per counter | Which dim (packet / gate 0-2) each counter belongs to |
| stride \\(\sigma_i\\) | per counter | stride for index computation |
| \\(\text{mask}\_{gd}\\) | per gate dim | Slice-id bitmask |
| \\(\text{match}\_{gd}\\) | per gate dim | Threshold for slice classification |
| \\(V_p\\) / \\(V\_{gd}\\) | packet dim / per gate dim | Valid count / threshold |
| \\(P\_{gd}\\) | per gate dim | Standard (0) vs transposed (1) |

Unassigned counters and disabled gate dims (\\(\text{mask}\_{gd} = 0, \text{match}\_{gd} = 1\\)) effectively pass through as "all valid."

## Sequencer

The sequencer interprets the flat time index `t` as a multi-dimensional counter.

### Counter Structure

Up to 8 nested counters iterate to produce the flit sequence:

$$t \to (c_0, c_1, \ldots, c_{k-1})$$

where \\(c_0\\) is the fastest (innermost) and \\(c_{k-1}\\) is the slowest (outermost).

Each counter \\(c_i\\) has a **limit** \\(L_i\\), cycling through \\(0, 1, \ldots, L_i - 1\\), and a **stride** \\(\sigma_i\\) that scales the counter's contribution to the dimension index (see [Original Dimensions](#original-dimensions)).
The total number of flits per slice is \\(L_0 \times L_1 \times \cdots \times L_{k-1}\\).

<details>
<summary>Example: 3 counters with limits [3, 2, 2]</summary>

This produces 3 * 2 * 2 = 12 flits per slice.
The counters cycle as:

```text
t=0:  (c_0=0, c_1=0, c_2=0)
t=1:  (c_0=1, c_1=0, c_2=0)
t=2:  (c_0=2, c_1=0, c_2=0)
t=3:  (c_0=0, c_1=1, c_2=0)   <- c_0 wraps, c_1 increments
t=4:  (c_0=1, c_1=1, c_2=0)
t=5:  (c_0=2, c_1=1, c_2=0)
t=6:  (c_0=0, c_1=0, c_2=1)   <- c_1 wraps, c_2 increments
t=7:  (c_0=1, c_1=0, c_2=1)
t=8:  (c_0=2, c_1=0, c_2=1)
t=9:  (c_0=0, c_1=1, c_2=1)
t=10: (c_0=1, c_1=1, c_2=1)
t=11: (c_0=2, c_1=1, c_2=1)
```

`c_0` changes every flit, `c_1` every 3 flits, `c_2` every 6 flits, just like digits in a mixed-radix number.

</details>

The sequencer produces counter values; the next step is mapping them to [original dimensions](#original-dimensions) and then to the [validity decision](#validity-decision).

### Original Dimensions

Each counter is assigned to one of 4 **original dimensions** (packet dim or gate dim 0-2), or left unassigned.

Let \\(D_d\\) be the set of counters assigned to original dimension `d`.
Each counter contributes to its assigned dimension's index by multiplying its current value by its stride.
The sum of all contributions gives the current position within that dimension's data:

$$\text{idx}_d(t) = \sum _ {i \in D_d} c_i(t) \cdot \sigma_i$$

This index tracks the position within that dimension's original data range.
Multiple counters can be assigned to the same dim; their contributions are simply summed.

<details>
<summary>Example: Counters mapped to original dimensions</summary>

Suppose 3 counters are configured as follows:

| Counter | Limit | stride | Assigned to |
|---------|-------|--------|-------------|
| c_0 | 3 | 8 | packet dim (W axis) |
| c_1 | 2 | 1 | gate dim 0 (C axis) |
| c_2 | 2 | 1 | gate dim 1 (H axis) |

At time step t=4, which gives (c_0=1, c_1=1, c_2=0):
- `idx_p = 1 * 8 = 8`, position 8 along W
- `idx_g0 = 1 * 1 = 1`, position 1 along C
- `idx_g1 = 0 * 1 = 0`, position 0 along H

Each dim uses its index independently to decide validity.

</details>

## Validity Decision

The following diagram shows the complete pipeline from time index to final valid count.
Each stage is explained in the subsections below.

```text
t (flat time index)
│
├─ mixed-radix decomposition (see Sequencer above)
▼
(c_0, c_1, ..., c_{k-1})              ← counter values
│
├─ each counter assigned to a dim, multiplied by stride σ_i
│  (see Original Dimensions above)
▼
idx_p(t), idx_g0(t), idx_g1(t), idx_g2(t)  ← per-dim indices
│
├─ packet dim: packet_vc = min(stride_p, max(0, V_p - idx_p))
├─ gate dim 0: gate_0 = f(masked_id(s), idx_g0, match_g0, V_g0)
├─ gate dim 1: gate_1 = f(masked_id(s), idx_g1, match_g1, V_g1)
├─ gate dim 2: gate_2 = f(masked_id(s), idx_g2, match_g2, V_g2)
│
▼
vc(s,t) = packet_vc(t) × gate_0(s,t) × gate_1(s,t) × gate_2(s,t)
```

Packet dim and gate dims make qualitatively different judgments:

- **Packet dim** answers: "**how many** elements in this flit are valid?" (a count, 0-8)
- **Gate dims** each answer: "is this flit valid **at all**?" (a binary gate, yes or no)

Gate dims act as gates: only when all three report "valid" does packet dim's count take effect.
If any gate reports "invalid", the entire flit gets valid count = 0.

### Packet Dim: Packet-Level Valid Count

Packet dim determines **how many elements** within a flit are valid.
Two parameters control the computation:

- \\(V_p\\): the **original valid count** for packet dim (the unpadded size of the data along this dimension).
- \\(\text{stride}_p\\): the **stride of the innermost counter assigned to packet dim**, representing how many flit elements belong to the axis tracked by packet dim.

The per-packet valid count is:

$$\text{packet\_vc}(t) = \min(\text{stride}_p, \max(0, V_p - \text{idx}_p(t)))$$

```text
packet_vc(t) = min( stride_p,         max(0, V_p - idx_p(t) ))
                    ─────────              ─────────────────
                    HW width cap           remaining valid data
```

When the axis fills all 8 flit positions, \\(\text{stride}_p = 8\\) and the formula is equivalent to \\(\min(8, \ldots)\\).
When the axis occupies only \\(k < 8\\) positions (with the remaining positions padded), \\(\text{stride}_p = k\\) caps the valid count so that only the axis's portion of the flit is counted as valid.

**Hardware constraints**:

- The innermost Packet counter must always be assigned to packet dim.
- Other counters may also be assigned to packet dim (e.g., a Time counter for the same axis).
- When no axis is assigned to packet dim, `packet_vc` is always 8 (full flit) or 0 (empty flit), effectively making packet dim a binary gate like gate dims.

As the sequencer advances, \\(\text{idx}_p\\) increases and `packet_vc` decreases.
This produces a repeating **sawtooth** pattern:

```text
Example 1: V_p = 19, stride_p = 8, counter stride=8, limit=3  (axis fills full 8-way)

flit 0: idx_p =  0 -> packet_vc = min(8, 19 -  0) = 8  (full)
flit 1: idx_p =  8 -> packet_vc = min(8, 19 -  8) = 8  (full)
flit 2: idx_p = 16 -> packet_vc = min(8, 19 - 16) = 3  (partial)

Example 2: V_p = 11, stride_p = 4, counter stride=4, limit=3  (axis fills 4 of 8 positions)

flit 0: idx_p =  0 -> packet_vc = min(4, 11 -  0) = 4  (full within stride)
flit 1: idx_p =  4 -> packet_vc = min(4, 11 -  4) = 4  (full within stride)
flit 2: idx_p =  8 -> packet_vc = min(4, 11 -  8) = 3  (partial)
```

In Example 2, positions 4-7 in each flit are padding and automatically excluded by the \\(\text{stride}_p = 4\\) cap.

**Key property**: `packet_vc` depends only on the sequencer state `t`, not on the slice `s`.
All slices receive the same packet valid count for the same time step.

<details>
<summary>Example: Why packet_vc is slice-independent</summary>

If \\(V_p = 19\\), \\(\text{stride}_p = 8\\), and the packet counter cycles [0, 8, 16], then:
- At t=2 (idx_p=16): packet_vc = 3 for **every** slice.
- Slice 0 gets vc=3, slice 5 gets vc=3, slice 15 gets vc=3, all the same.

This is because packet dim's formula \\(\min(\text{stride}_p, V_p - \text{idx}_p)\\) has no `s` term.
Gate dims can still make certain slices' final vc = 0 (by reporting invalid),
but they cannot change the packet_vc value itself.

</details>

### Gate Dims: Per-Flit Binary Validity

Gate dims 0, 1, 2 decide whether a **flit as a whole** is valid (1) or invalid (0), not a count.

Each gate dim classifies slices into groups by extracting a subset of the slice-id bits (via a bitmask) and comparing the result against a threshold.
The bitmask \\(\text{mask}\_{gd}\\) selects which bits of the slice-id this gate dim tracks:

$$\text{masked\_id} (s) = s \mathbin{\\&} \text{mask}\_{gd}$$

```text
Example: 16 slices (4-bit slice_id), mask_g0 = 0b1100

slice_id (4 bits):   [ b3  b2  b1  b0 ]
mask_g0 = 0b1100:    [  1   1   0   0 ]
                      ─────────────────
masked_id:           [ b3  b2   0   0 ]  → extracts the upper 2 bits
```

Slices fall into three groups based on comparing \\(\text{masked\_id}\\) with \\(\text{match}\_{gd}\\):

| Group | Condition | Meaning |
|-------|-----------|---------|
| **Below** | \\(\text{masked\_id}(s) < \text{match}\_{gd}\\) | All time steps valid |
| **Boundary** | \\(\text{masked\_id}(s) = \text{match}\_{gd}\\) | Valid when \\(\text{idx}\_{gd}(t) < V\_{gd}\\) |
| **Above** | \\(\text{masked\_id}(s) > \text{match}\_{gd}\\) | Depends on mode (see below) |

The \\(P\_{gd}\\) flag selects between two modes that differ only in the "above" group:

#### Standard mode (\\(P\_{gd} = 0\\))

$$\text{gate}_d(s, t) = \begin{cases} 1 & \text{masked\_id}(s) < \text{match}\_{gd} \\\\ [\text{idx}\_{gd}(t) < V\_{gd}] & \text{masked\_id}(s) = \text{match}\_{gd} \\\\ 0 & \text{masked\_id}(s) > \text{match}\_{gd} \end{cases}$$

Above-threshold slices are **entirely invalid**.
This is the common case: the Slice factor is laid out in ascending order, so slices beyond the boundary contain no valid data.

#### Transposed mode (\\(P\_{gd} = 1\\))

$$\text{gate}_d(s, t) = \begin{cases} 1 & \text{masked\_id}(s) < \text{match}\_{gd} \\\\ [\text{idx}\_{gd}(t) < V\_{gd}] & \text{masked\_id}(s) \ge \text{match}\_{gd} \end{cases}$$

Above-threshold slices get the **same** \\(V\_{gd}\\) check as the boundary: they are not entirely invalid.
This handles the transposed case where the slice ID encodes the _inner_ index: slices beyond the boundary still contain valid data at early time steps (the outer index is small enough), and only run out of valid data at the same point as the boundary slice.

To disable a gate dim (make it always valid), set \\(\text{mask}\_{gd} = 0, \text{match}\_{gd} = 1\\).
Then \\(\text{masked\_id} = 0 < 1\\) for all slices, so every slice is in the "below" group.

<details>
<summary>Example: Standard mode, H=5 split into Ho=4 (slice) × Hi=2 (time)</summary>

H=5 is split into `Ho × Hi = 4 × 2` (padded from 5 to 8).
`Ho` is the Slice factor (encoded in slice-id bits), `Hi` is the Time factor (sequencer counter).
Axis index = `Ho * 2 + Hi`. Valid when index < 5.

Gate dim 0 config: `mask=0b1100` (extracts 2 bits for Ho), `match=2`, `V_g0=1`, standard mode.

16 slices, where `masked_id = (slice_id & 0b1100) >> 2` gives Ho:

| Ho | masked_id | Group | Hi=0 | Hi=1 |
|----|-----------|-------|------|------|
| 0 | 0 | below (< 2) | valid | valid |
| 1 | 1 | below (< 2) | valid | valid |
| 2 | 2 | boundary (= 2) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |
| 3 | 3 | above (> 2) | invalid | invalid |

Ho=0,1: both time steps valid (index 0-3, all < 5).
Ho=2: only first time step (index 4 < 5), second invalid (index 5 >= 5).
Ho=3: fully invalid (index 6, 7 >= 5).

</details>

<details>
<summary>Example: Transposed mode, H=5 split into Ho=4 (slice, inner) × Hi=2 (time, outer)</summary>

H=5 is split into `Ho × Hi = 4 × 2` (padded from 5 to 8), but transposed: Ho is the inner factor, Hi is the outer factor.
Axis index = `Hi * 4 + Ho`. Valid when index < 5.

Gate dim 0 config: `match=1` (= 5 mod 4), `V_g0=1` (= floor(5/4)), transposed mode.

| Ho | masked_id | Group | Hi=0 | Hi=1 |
|----|-----------|-------|------|------|
| 0 | 0 | below (< 1) | valid | valid |
| 1 | 1 | boundary (= 1) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |
| 2 | 2 | above (> 1) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |
| 3 | 3 | above (> 1) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |

Verify: Ho=0, Hi=0: 0 < 5, Hi=1: 4 < 5, so 2 steps.
Ho=1, Hi=0: 1 < 5, Hi=1: 5 >= 5, so 1 step.
Ho=2, Hi=0: 2 < 5, Hi=1: 6 >= 5, so 1 step.
Ho=3, Hi=0: 3 < 5, Hi=1: 7 >= 5, so 1 step.

Key difference from standard: the "above" group (Ho=2,3) still gets V_g0=1 valid time steps, not zero.

</details>

<details>
<summary>Example: Full VCG computation for [H=5, C=5, W=19], step-by-step build-up</summary>

This example builds up from one axis to three, so each dimension's contribution is clear.

Original shape `[H, C, W] = [5, 5, 19]`.
Each axis is split into a slice part (slice_id) and a time part (sequencer):

```text
H = 5  ->  Ho(slice) * Hi(time) = 4 * 2    (padded from 5 to 8)
C = 5  ->  Co(slice) * Ci(time) = 4 * 2    (padded from 5 to 8)
W = 19 ->  Wi(packet)            = 3 * 8    (padded from 19 to 24)
```

#### Step 1: W=19 only (packet dim, no gates)

Ignore H and C for now. Disable gate dims 0 and 1.
Every slice processes 3 flits (Wi limit=3), and packet dim produces the sawtooth:

```text
packet_vc:  8, 8, 3
              ^     ^
            full   19 - 16 = 3 (partial)
```

Since there are no gates, **every slice gets this exact same pattern**:

```text
All slices, all flits:
flit 0: ████████  (vc=8)
flit 1: ████████  (vc=8)
flit 2: ███       (vc=3)
```

#### Step 2: Add C=5 (packet dim + gate dim 0)

Now enable the C-axis gate (gate dim 0).
C=5 is split into Co(slice, 4 values) * Ci(time, limit 2).
The C-gate uses: `mask=0b0011` (extracts Co from slice_id), `match=2`, `V_g0=1`, standard mode.

Each slice now runs 6 flits: Ci (limit 2) * Wi (limit 3).
The C-gate classifies slices by their Co value:

| Co | Group | Effect |
|----|-------|--------|
| 0 | below (< 2) | gate open: all 6 flits get packet dim's pattern |
| 1 | below (< 2) | gate open: same |
| 2 | boundary (= 2) | gate open for Ci=0, closed for Ci=1 |
| 3 | above (> 2) | gate closed: all 6 flits get vc=0 |

Result per slice (6 flits = 2 Ci groups * 3 Wi flits):

```text
Co=0:  [8,8,3, 8,8,3]    <- both Ci steps valid
Co=1:  [8,8,3, 8,8,3]    <- same
Co=2:  [8,8,3, 0,0,0]    <- Ci=0 valid, Ci=1 gated off
Co=3:  [0,0,0, 0,0,0]    <- entirely gated off
```

Notice the gate's effect: some slices go entirely to zero, and the boundary slice loses its second half.
But within the valid flits, the `[8,8,3]` pattern from packet dim is unchanged.

#### Step 3: Add H=5 (full 3-axis, packet dim + gate dim 0 + gate dim 1)

Now enable the H-axis gate (gate dim 1).
H=5 is split into Ho(slice, 4 values) * Hi(time, limit 2).
The H-gate uses: `mask=0b1100` (extracts Ho from slice_id), `match=0b1000`, `V_g1=1`, standard mode.

Slice ID encodes both slice factors: `slice_id = Ho * 4 + Co`, giving 16 slices.
Each slice now runs 12 flits: Hi (limit 2) * Ci (limit 2) * Wi (limit 3).

| Dim | Axis | What it tracks | VCG config |
|-----|------|----------------|------------|
| packet | W=19 | element count in packet | V_p=19, stride_p=8, counter stride=8, limit=3 |
| gate 0 | C=5 | gate: is Co within valid range? | mask=0b0011, match=2, V_g0=1, standard |
| gate 1 | H=5 | gate: is Ho within valid range? | mask=0b1100, match=0b1000, V_g1=1, standard |

The H-gate classifies slices by Ho, same logic as C-gate by Co:

| Ho | Group | Effect |
|----|-------|--------|
| 0 | below | H-gate open |
| 1 | below | H-gate open |
| 2 | boundary | H-gate open for Hi=0, closed for Hi=1 |
| 3 | above | H-gate closed |

The **final vc** for each flit is `packet_vc(t) * C_gate(s,t) * H_gate(s,t)`.
Both gates must be open for packet dim's count to survive.

The complete heatmap (16 slices * 12 flits).
Columns are slices grouped by Ho; rows are flits grouped by (Hi, Ci).
Right-side annotations show which gates are active for each row:

```text
                     Ho=0       |Ho=1       |Ho=2       |Ho=3
                Co:  0  1  2  3 | 0  1  2  3| 0  1  2  3| 0  1  2  3
  H-gate:            v  v  v  v | v  v  v  v| >  >  >  >| x  x  x  x
  C-gate:            v  v  >  x | v  v  >  x| v  v  >  x| v  v  >  x
--------------------------------------------------------------------------------
 t= 0  Hi=0,Ci=0  W  8  8  8  0 | 8  8  8  0| 8  8  8  0| 0  0  0  0  H:v C:v
 t= 1             |  8  8  8  0 | 8  8  8  0| 8  8  8  0| 0  0  0  0
 t= 2             |  3  3  3  0 | 3  3  3  0| 3  3  3  0| 0  0  0  0
                                |           |           |
 t= 3  Hi=0,Ci=1  W  8  8  0  0 | 8  8  0  0| 8  8  0  0| 0  0  0  0  H:v C:>
 t= 4             |  8  8  0  0 | 8  8  0  0| 8  8  0  0| 0  0  0  0
 t= 5             |  3  3  0  0 | 3  3  0  0| 3  3  0  0| 0  0  0  0
                                |           |           |
 t= 6  Hi=1,Ci=0  W  8  8  8  0 | 8  8  8  0| 0  0  0  0| 0  0  0  0  H:> C:v
 t= 7             |  8  8  8  0 | 8  8  8  0| 0  0  0  0| 0  0  0  0
 t= 8             |  3  3  3  0 | 3  3  3  0| 0  0  0  0| 0  0  0  0
                                |           |           |
 t= 9  Hi=1,Ci=1  W  8  8  0  0 | 8  8  0  0| 0  0  0  0| 0  0  0  0  H:> C:>
 t=10             |  8  8  0  0 | 8  8  0  0| 0  0  0  0| 0  0  0  0
 t=11             |  3  3  0  0 | 3  3  0  0| 0  0  0  0| 0  0  0  0

v = open (below threshold)   > = boundary (partial)   x = closed (above)
```

Reading the patterns:

- **Ho=3 columns** (rightmost 4): all 0. H-gate `x` (above threshold, always closed).
- **Co=3 columns** (every 4th): all 0. C-gate `x`.
- **Co=2 columns** (H:v C:`>`): C-gate is boundary; only rows with Ci=0 pass. Compare Co=1 vs Co=2 to see the gate's effect.
- **Ho=2 columns** (H:`>` C:v): H-gate is boundary; only rows with Hi=0 pass. Compare Ho=1 vs Ho=2.
- **Ho=2 * Co=2** (both `>`): only (Hi=0, Ci=0) rows pass, the intersection of both boundaries.
- **Within valid cells**: the `[8, 8, 3]` sawtooth from packet dim always appears, the same regardless of slice.

</details>

## What Patterns Are Expressible

The [valid count formula](#valid-count-formula) is a product of four independent terms.
This multiplicative structure determines which `vc(s, t)` functions the hardware can produce, and which it cannot.

### Why Limitations Arise

Each limitation traces back to a specific part of the formula.

**Packet dim cannot see slice-id.**
The [packet dim formula](#packet-dim-packet-level-valid-count) `packet_vc(t) = min(stride_p, max(0, V_p − idx_p(t)))` depends only on `t`.
If two slices need different partial counts at the same time step, packet dim cannot produce both:

```text
Suppose we need:  vc(s=0, t=0) = 8,  vc(s=1, t=0) = 3
                                       ───────────────
                                       packet dim would need to output
                                       both 8 and 3 at t=0, impossible
```

**Each gate classifies slices by a single threshold after masking.**
[Gate dims](#gate-dims-per-flit-binary-validity) first apply a bitmask to the slice-id (`masked_id = slice_id & mask`), then compare against one value `match`.
This produces three contiguous groups: below, boundary, above.
A gate **cannot** express "slices 0, 3, 7 are valid but 1, 2 are not"; it can represent only contiguous ranges of `masked_id`.
The mask selects which bits of the slice-id to inspect, allowing one gate to track a specific axis even when the slice-id encodes multiple axes.

**At most 4 independent checks.**
One packet count (packet dim) + three binary gates (gate dims) = 4 orthogonal dimensions total.

### Single-Axis Scenarios

A padded axis (original size `n`, padded to `n' > n`) occupies some combination of **Packet** ([packet dim](#packet-dim-packet-level-valid-count)), **Time** ([sequencer](#sequencer)), and **Slice** ([gate dims](#gate-dims-per-flit-binary-validity) via slice-id bits).

#### Single-position: Packet / Time / Slice

When an axis occupies only one position, validity tracking is straightforward:

- **Packet only** → packet dim handles the sawtooth (see [Packet Dim examples](#packet-dim-packet-level-valid-count)).
- **Time only** → [gate dim](#gate-dims-per-flit-binary-validity) with `mask=0, match=0`: all slices are boundary, binary validity by time step.
- **Slice only** → [gate dim](#gate-dims-per-flit-binary-validity) with appropriate mask/match: all time steps within a valid slice pass, invalid slices are fully gated.

All three are **always supported**.

#### Slice + Time

One axis split between slice-id bits and sequencer counters.
This is the VCG's most important use case, and it is how [gate dims](#gate-dims-per-flit-binary-validity) are typically used.

**Standard** (slice outer, time inner): axis index = `Ho × time_count + Hi`.

<details>
<summary>Example: H=14, Ho=8 (slice) × Hi=3 (time), standard mode</summary>

Gate dim config: `match = ⌊14/3⌋ = 4`, `V = 14 mod 3 = 2`.

```text
Ho  masked_id  group       Hi=0         Hi=1         Hi=2
──  ─────────  ─────────   ──────────   ──────────   ──────────
0   0          below       idx=0 < 2 ✅  idx=1 < 2 ✅  always ✅
1   1          below       always ✅     always ✅     always ✅
2   2          below       always ✅     always ✅     always ✅
3   3          below       always ✅     always ✅     always ✅
4   4          boundary    idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
5   5          above       ❌            ❌            ❌
6   6          above       ❌            ❌            ❌
7   7          above       ❌            ❌            ❌
```

Why "below" is genuinely all-valid: `Ho × 3 + Hi < 4 × 3 = 12 ≤ 14` for all `Hi ∈ [0,3)`.
Standard mode tolerates over-allocated `time_count`; the "below" interpretation remains correct.

</details>

**Transposed** (time outer, slice inner): axis index = `Hi × slice_count + Ho`.

<details>
<summary>Example: H=19, Ho=8 (slice, inner) × Hi=3 (time, outer), transposed mode</summary>

Gate dim config: `match = 19 mod 8 = 3`, `V = ⌊19/8⌋ = 2`, transposed mode.

```text
Ho  masked_id  group       Hi=0         Hi=1         Hi=2
──  ─────────  ─────────   ──────────   ──────────   ──────────
0   0          below       always ✅     always ✅     always ✅
1   1          below       always ✅     always ✅     always ✅
2   2          below       always ✅     always ✅     always ✅
3   3          boundary    idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
4   4          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
5   5          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
6   6          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
7   7          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
```

Verify against real data (axis index = `Hi × 8 + Ho`, valid when < 19):
- Ho=0, Hi=2: `2×8 + 0 = 16 < 19` ✅; "below" gives all-valid = 3 steps, need `V+1 = 3` steps ✅
- Ho=3, Hi=2: `2×8 + 3 = 19 ≥ 19` ❌; boundary gives 2 steps ✅
- Ho=7, Hi=1: `1×8 + 7 = 15 < 19` ✅; "above" gives V=2 steps, actual need is 2 steps ✅

**Constraint**: `time_count` must equal `⌈n / slice_count⌉` (= `V + 1`).
The "below" group gets `time_count` valid steps from the HW "all-valid" interpretation.
If `time_count > V + 1`, the "below" group receives more valid steps than the data actually has.

</details>

#### Packet + Time

Both packet and time factors assigned to **packet dim**, with multiple counters contributing to `idx_p(t)` (see [Original Dimensions](#original-dimensions)).

<details>
<summary>Example: n=50, two counters on packet dim, contiguous (`stride_outer = 8 × 3 = 24`)</summary>

```text
c_inner (limit=3, stride=8):  packet counter
c_outer (limit=3, stride=24): time counter     (24 = 8 × 3 ✅ contiguous)

idx_p = c_outer × 24 + c_inner × 8

          c_inner=0    c_inner=1    c_inner=2
          ─────────    ─────────    ─────────
c_outer=0  idx_p=0→8   idx_p=8→8    idx_p=16→8
c_outer=1  idx_p=24→8  idx_p=32→8   idx_p=40→8
c_outer=2  idx_p=48→2  idx_p=56→0   idx_p=64→0
                  ↑
                  min(8, 50-48)=2
```

Packet dim handles both the within-flit and across-flit boundaries.

</details>

#### Slice + Packet: not supported

One axis split between **slice** (gate dim) and **packet** (packet dim).
This directly violates the **slice-independent packet count** constraint (see [Packet Dim: Key property](#packet-dim-packet-level-valid-count)).

`packet_vc(t)` depends only on `t`, but the boundary slice needs a different partial count than all-valid slices.
The gate can multiply by 0 or 1, so it can fully close a flit but **cannot change the partial count**.

<details>
<summary>Example: n=10, stride_p=8, slice_count=2</summary>

```text
What we need:
  Ho=0: elements 0-7,   all valid     →  vc = 8
  Ho=1: elements 8-15,  first 2 valid →  vc = 2
                                           ─
                                           partial count, different from 8

Attempt 1: set `V_p = 10`:
  packet_vc = min(8, 10-0) = 8         for ALL slices
  Ho=0:  vc = 8 × 1 = 8  ✅
  Ho=1:  vc = 8 × 1 = 8  ❌  (need 2, not 8)
         vc = 8 × 0 = 0  ❌  (gate can close to 0, not to 2)

Attempt 2: set `V_p = 2`:
  packet_vc = min(8, 2-0) = 2          for ALL slices
  Ho=0:  vc = 2 × 1 = 2  ❌  (need 8, not 2)

No single V_p works.  Packet dim produces one value; the gate can only multiply by 0 or 1.
```

</details>

**Note on degenerate cases:**
When `n % stride_p = 0`, every packet is either fully valid or fully invalid, so packet dim produces no partial counts and a gate alone handles validity.
This is effectively **Slice only**, not a true Slice + Packet scenario.
Similarly, `n <= stride_p` means a single flit covers the entire axis, reducing to **Packet only**.

When the VCG cannot express the required pattern, [Padding Strategy](./intra-slice-reduce.md#padding-strategy) alternatives are available.

#### Slice + Time + Packet: not supported

The axis spans all three positions.
The Slice + Packet conflict carries over: the boundary slice still needs a different partial count than all-valid slices, and `packet_vc(t)` still cannot vary by slice.

The same degenerate exception applies: `n % stride_p = 0` eliminates partial counts, reducing to **Slice + Time** (packet dim unused).

### Multiple Axes

Each padded axis that needs validity tracking consumes one [original dimension](#original-dimensions) slot:

| Resource | Capacity | Notes |
|----------|----------|-------|
| **[Packet Dim](#packet-dim-packet-level-valid-count)** (packet count) | 1 slot | Innermost counter determines `stride_p` |
| **[Gate Dims](#gate-dims-per-flit-binary-validity)** (binary gates) | 3 slots | One gate per padded axis |
| **Unpadded axes** | free | No dim needed (`mask=0, match=1`) |

When the packet axis is fully aligned (`n % stride_p = 0`), `packet_vc` is constant and packet dim is effectively unused, so it can be repurposed as a gate for another axis.

### Summary

A valid count function `vc(s, t)` is VCG-expressible only if:

1. **Prefix property**: Valid elements form a contiguous prefix `[0, vc)` within each flit.
2. **Slice-independent packet count**: `packet_vc(t)` must be the same across all slices at the same `t`.
   Slices can be gated to `vc = 0`, but cannot receive a different partial count.
3. **Monotonic slice ordering**: Each gate dim classifies slices by a single threshold on `masked_id`.
4. **At most 4 orthogonal dimensions**: 1 packet count + 3 binary gates.

| Placement | Dim | Supported? | Key constraint |
|-----------|-----|------------|----------------|
| Packet only | packet | ✅ | none |
| Time only | gate | ✅ | none |
| Slice only | gate | ✅ | none |
| Slice + Time (standard) | gate | ✅ | none |
| Slice + Time (transposed) | gate | ✅ | `time_count = ⌈n / slice_count⌉` |
| Packet + Time | packet | ✅ | none |
| Slice + Packet | packet + gate | ❌ | `packet_vc(t)` cannot vary by slice |
| Slice + Time + Packet | packet + gate | ❌ | same as Slice + Packet |

For mapping-level code examples of each placement, see [Examples](./vcg-interface.md#examples).
For unsupported cases, see [Padding Strategy](./intra-slice-reduce.md#padding-strategy).

## Downstream: 4-Way Operations

VCG assigns valid counts per 8-way flit, but the VectorArithmeticUnit can operate on 4-way halves.

| Operation | Input | Output | Valid Count Transformation |
|-----------|-------|--------|---------------------------|
| **split_way4** | 8-way flit (vc = v) | two 4-way flits | `vc_low = min(v, 4)`, `vc_high = max(v - 4, 0)` |
| **trim_way4** | 8-way flit (vc = v) | one 4-way flit | `vc = v` (requires v <= 4) |
| **concat_way8** | two 4-way flits | 8-way flit | `vc = vc_low + vc_high` |
| **pad_way8** | 4-way flit | 8-way flit | `vc` unchanged |

The prefix property is preserved through split and concat.
For trim_way4, the constraint `v <= 4` must be statically guaranteed by the mapping; if the upper 4 elements could be valid, trimming them would lose data.
