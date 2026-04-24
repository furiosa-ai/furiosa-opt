# Chip/Cluster Reduce

When a previous operation has already mapped the reduce axis to the `Chip` or `Cluster` dimension, chip/cluster reduce is needed to combine partial results across physically separate processing units.
This section demonstrates how to perform those reduction operations when data is distributed across multiple chips or clusters.

<!-- > **TODO(youseok.yang)**: Slice/ChipShuffle operations do not yet have explicit Virtual ISA interfaces. -->
<!-- > Once those interfaces are available, replace the conceptual data flow descriptions below with working kernel code, following the style of `fetch-commit-engine.md`. -->
<!-- > Also fix the ReduceScatter example: Step 6 performs AllGather, turning the result into AllReduce and contradicting the stated Goal (`element: 1` per chip); either remove Step 6 or rename the example. -->

When possible, assigning reduce axes to Slice/Element (reduced by Inter-Slice Block/Vector Engine) is preferred because it avoids cross-chip communication overhead.

Two main operations implement chip/cluster reduce: AllReduce and ReduceScatter.
Both combine Switch Engine operations (for data redistribution across slices within a cluster) with Vector Engine binary operations (for actual reduction computation).


## `ReduceScatter`

`ReduceScatter` reduces data distributed across chip/cluster axes while distributing the result so each chip holds a portion.
This operation is useful when you need both reduction and result distribution in a single step.

### Example: 4-chip `ReduceScatter` with Add

This example demonstrates how to perform reduction across chips when data is partitioned by one dimension (`A`) but needs to be reduced along a different dimension (`B`).
The challenge is that each chip owns data for all `B` values of its assigned `A` value, but we need to sum across all `A` values for each `B`.

**Input:**
A 2D tensor `[A=4, B=4]` with 16 total elements, distributed across 4 chips:
- **Shape**: `[A=4, B=4]` - 16 elements total
- **Data type**: `i8` (8-bit signed integer)
- **Storage**: SRAM on each chip
- **Distribution**: `In = {chip: A, slice: 256, element: B}`
  - Chip 0 owns: `(A=0, B=0)`, `(A=0, B=1)`, `(A=0, B=2)`, `(A=0, B=3)` - all B values for A=0
  - Chip 1 owns: `(A=1, B=0)`, `(A=1, B=1)`, `(A=1, B=2)`, `(A=1, B=3)` - all B values for A=1
  - Chip 2 owns: `(A=2, B=0)`, `(A=2, B=1)`, `(A=2, B=2)`, `(A=2, B=3)` - all B values for A=2
  - Chip 3 owns: `(A=3, B=0)`, `(A=3, B=1)`, `(A=3, B=2)`, `(A=3, B=3)` - all B values for A=3

**Goal:**
Reduce along the `A` axis (summing across chips) while keeping results distributed by `B`:
- **Output shape**: `[B=4]` - 4 elements (A dimension eliminated by reduction)
- **Output distribution**: `Out = {chip: 4, slice: 256, element: 1}`
  - Chip 0 should hold: sum of `(A=0..3, B=0)` - the sum of all A values for B=0
  - Chip 1 should hold: sum of `(A=0..3, B=1)` - the sum of all A values for B=1
  - Chip 2 should hold: sum of `(A=0..3, B=2)` - the sum of all A values for B=2
  - Chip 3 should hold: sum of `(A=0..3, B=3)` - the sum of all A values for B=3

**Processing:**

`Slice` (also called asymmetric slice) is a sub-context operation that extracts a subset of elements from specific chip positions (see [Implementation Methods](#implementation-methods-for-each-operation)).
`ChipShuffle` is a DMA-based redistribution operation that moves data from one chip to another.

The algorithm works through six stages: create four intermediate tensors using diagonal `Slice` + `ChipShuffle` patterns, add them to reduce the `A` axis, then broadcast results to all chips.
The diagonal pattern ensures each chip receives the data it needs for its assigned `B` value.

#### Initial State

Each chip owns one value along the `A` axis:
- Chip 0: `(A=0, B=0)`, `(A=0, B=1)`, `(A=0, B=2)`, `(A=0, B=3)`
- Chip 1: `(A=1, B=0)`, `(A=1, B=1)`, `(A=1, B=2)`, `(A=1, B=3)`
- Chip 2: `(A=2, B=0)`, `(A=2, B=1)`, `(A=2, B=2)`, `(A=2, B=3)`
- Chip 3: `(A=3, B=0)`, `(A=3, B=1)`, `(A=3, B=2)`, `(A=3, B=3)`

#### Step 1: Create Tensor `T0` - `Slice(0,1,2,3)`

This step selects specific positions along the `B` axis from each chip using `Slice`, creating a diagonal selection pattern:
- Chip 0: select (0,0)
- Chip 1: select (1,1)
- Chip 2: select (2,2)
- Chip 3: select (3,3)

Result `T0`:
- Chip 0: (0,0)
- Chip 1: (1,1)
- Chip 2: (2,2)
- Chip 3: (3,3)

#### Step 2: Create Tensor `T1` - `Slice(3,0,1,2)` + `ChipShuffle(1,2,3,0)`

This step combines `Slice` with `ChipShuffle` to create a rotated diagonal pattern.
First, `Slice` selects elements:
- Chip 0: (0,3)
- Chip 1: (1,0)
- Chip 2: (2,1)
- Chip 3: (3,2)

Then `ChipShuffle(1,2,3,0)` redistributes the data so each chip receives data from another chip:
- Data from Chip 1 moves to Chip 0: (1,0)
- Data from Chip 2 moves to Chip 1: (2,1)
- Data from Chip 3 moves to Chip 2: (3,2)
- Data from Chip 0 moves to Chip 3: (0,3)

#### Step 3: Create Tensor `T2` - Slice(2,3,0,1) + ChipShuffle(2,3,0,1)

This step creates another rotated diagonal pattern.
First, Slice selects positions:
- Chip 0: select (0,2)
- Chip 1: select (1,3)
- Chip 2: select (2,0)
- Chip 3: select (3,1)

Then ChipShuffle(2,3,0,1) redistributes the data, yielding `T2`:
- Chip 0: (2,0)
- Chip 1: (3,1)
- Chip 2: (0,2)
- Chip 3: (1,3)

#### Step 4: Create Tensor `T3` - ChipSlice(1,2,3,0) + ChipShuffle(3,0,1,2)

This step creates the final rotated diagonal pattern.
First, Slice selects positions:
- Chip 0: select (0,1)
- Chip 1: select (1,2)
- Chip 2: select (2,3)
- Chip 3: select (3,0)

Then ChipShuffle(3,0,1,2) redistributes the data, yielding `T3`:
- Chip 0: (3,0)
- Chip 1: (0,1)
- Chip 2: (1,2)
- Chip 3: (2,3)

#### Step 5: Vector Engine Add - A Axis Reduction

This step performs the actual reduction by adding all 4 tensors element-wise:
- Chip 0: (0,0) + (1,0) + (2,0) + (3,0)
- Chip 1: (1,1) + (2,1) + (3,1) + (0,1)
- Chip 2: (2,2) + (3,2) + (0,2) + (1,2)
- Chip 3: (3,3) + (0,3) + (1,3) + (2,3)

After this addition, each chip holds only one value because the A axis has been reduced:
- `Intermediate = { chip: B, slice: 256, element: 1 }`

#### Step 6: AllGather

<!-- > **TODO(jeongmin.park)**: After Step 5, each chip already holds exactly 1 element (the correct ReduceScatter output, matching the Goal above which says `element: 1`). Step 6 (AllGather) broadcasts to all chips, producing `element: 4` — that is AllReduce, not ReduceScatter. Either remove Step 6 (to match the ReduceScatter definition/goal), or rename this example to AllReduce. -->

This final step broadcasts the result so all chips hold the complete reduction output.
Each chip gathers data from Chip 0 through Chip 3:

- `Intermediate = { chip: 4, slice: 256, element: B }`

**Output:**

<!-- > **TODO(jeongmin.park)**: The Goal above says `Out = {chip: 4, slice: 256, element: 1}` (each chip holds 1 element), but this says `element: 4` (each chip holds 4 elements). One of them is wrong. -->

After all six steps complete, each chip holds a portion of the reduced result:
- **Final distribution**: `Out = {chip: A, slice: 256, element: 4}`
- **Chip 0**: Holds sum of all `(A=*, B=0)` values
- **Chip 1**: Holds sum of all `(A=*, B=1)` values
- **Chip 2**: Holds sum of all `(A=*, B=2)` values
- **Chip 3**: Holds sum of all `(A=*, B=3)` values

The `A` axis has been reduced (summed across all 4 chips), and the results are scattered across chips based on the `B` value.
Each chip now owns one element representing the sum of all `A` values for its assigned `B` coordinate.

**Why this example is useful:**

`ReduceScatter` combines two operations that frequently occur together in distributed computing:
- **Reduction across processors**: Summing/aggregating data distributed across multiple chips
- **Result distribution**: Each chip gets a portion of the result rather than duplicating it everywhere

This pattern is essential for:
- **Distributed matrix multiplication**: Reduce partial products from different chips while distributing the result
- **Gradient aggregation in data parallelism**: Sum gradients across workers, with each worker holding a portion
- **Memory efficiency**: Avoids storing the full reduced result on every chip (unlike AllReduce)
- **Pipeline parallelism**: Enables efficient communication patterns between pipeline stages

The diagonal slicing pattern is key: it ensures that data needed for each output element is gathered from all chips before reduction, minimizing communication rounds.


## `AllReduce`

`AllReduce` reduces data distributed across the chip axis so that all chips have identical reduction results.
Unlike `ReduceScatter`, `AllReduce` ensures every chip ends up with the complete result rather than a portion.

### Example: 4-chip `AllReduce` with Add

This example demonstrates the most common collective operation in distributed deep learning: reducing values across all processors so every processor has the identical complete result.
This is essential for operations like averaging gradients across data-parallel training workers.

**Input:**
A 2D tensor `[A=4, B=4]` distributed across 4 chips by the `A` dimension:
- **Shape**: `[A=4, B=4]` - 16 elements total
- **Data type**: `i8` (8-bit signed integer)
- **Storage**: SRAM on each chip
- **Distribution**: `In = {chip: A, slice: 256, element: B}`
  - Chip 0 owns: `(A=0, B=0-3)` - all 4 B values for A=0
  - Chip 1 owns: `(A=1, B=0-3)` - all 4 B values for A=1
  - Chip 2 owns: `(A=2, B=0-3)` - all 4 B values for A=2
  - Chip 3 owns: `(A=3, B=0-3)` - all 4 B values for A=3

**Goal:**
Reduce along the `A` axis and replicate the complete result to all chips:
- **Output shape**: `[B=4]` - 4 elements (A dimension eliminated by summation)
- **Output distribution**: `Out = {chip: 4, slice: 256, element: B}`
  - **Every chip** holds: sum of all `(A=0..3, B=0)`, sum of all `(A=0..3, B=1)`, sum of all `(A=0..3, B=2)`, sum of all `(A=0..3, B=3)`
  - All chips have identical data after AllReduce completes

**Processing:**

The algorithm creates 4 versions of the input tensor through rotation, then adds them all together:
1. Use 3 `ChipShuffle` operations on the original tensor `T0` to create 3 rotated versions (`T1`, `T2`, `T3`)
2. Add all 4 tensors element-wise using Vector Engine
3. Every chip performs the same additions on its local data, producing identical results everywhere

#### Initial State (`T0`)

Each chip owns one value along the A axis:
- Chip 0: (A=0, B=0), (A=0, B=1), (A=0, B=2), (A=0, B=3)
- Chip 1: (A=1, B=0), (A=1, B=1), (A=1, B=2), (A=1, B=3)
- Chip 2: (A=2, B=0), (A=2, B=1), (A=2, B=2), (A=2, B=3)
- Chip 3: (A=3, B=0), (A=3, B=1), (A=3, B=2), (A=3, B=3)

#### Step 1: Create Tensor `T1` - ChipShuffle(1,2,3,0)

This step rotates the data by one chip position.
ChipShuffle(1,2,3,0) is applied to the original `T0`:
- Data from Chip 1 moves to Chip 0
- Data from Chip 2 moves to Chip 1
- Data from Chip 3 moves to Chip 2
- Data from Chip 0 moves to Chip 3

The resulting `T1`:
- Chip 0: (1,0), (1,1), (1,2), (1,3)
- Chip 1: (2,0), (2,1), (2,2), (2,3)
- Chip 2: (3,0), (3,1), (3,2), (3,3)
- Chip 3: (0,0), (0,1), (0,2), (0,3)

#### Step 2: Create Tensor `T2` - ChipShuffle(2,3,0,1)

This step rotates the data by two chip positions.
ChipShuffle(2,3,0,1) is applied to the original `T0`:
- Data from Chip 2 moves to Chip 0
- Data from Chip 3 moves to Chip 1
- Data from Chip 0 moves to Chip 2
- Data from Chip 1 moves to Chip 3

The resulting `T2`:
- Chip 0: (2,0), (2,1), (2,2), (2,3)
- Chip 1: (3,0), (3,1), (3,2), (3,3)
- Chip 2: (0,0), (0,1), (0,2), (0,3)
- Chip 3: (1,0), (1,1), (1,2), (1,3)

#### Step 3: Create Tensor `T3` - ChipShuffle(3,0,1,2)

This step rotates the data by three chip positions.
ChipShuffle(3,0,1,2) is applied to the original `T0`:
- Data from Chip 3 moves to Chip 0
- Data from Chip 0 moves to Chip 1
- Data from Chip 1 moves to Chip 2
- Data from Chip 2 moves to Chip 3

The resulting `T3`:
- Chip 0: (3,0), (3,1), (3,2), (3,3)
- Chip 1: (0,0), (0,1), (0,2), (0,3)
- Chip 2: (1,0), (1,1), (1,2), (1,3)
- Chip 3: (2,0), (2,1), (2,2), (2,3)

#### Step 4: Vector Engine Add - A Axis Reduction

This step performs the actual reduction by adding all 4 tensors `T0`, `T1`, `T2`, `T3`:
- Chip 0: (0,0)+(1,0)+(2,0)+(3,0), (0,1)+(1,1)+(2,1)+(3,1), (0,2)+(1,2)+(2,2)+(3,2), (0,3)+(1,3)+(2,3)+(3,3)
- Chip 1: (1,0)+(2,0)+(3,0)+(0,0), (1,1)+(2,1)+(3,1)+(0,1), (1,2)+(2,2)+(3,2)+(0,2), (1,3)+(2,3)+(3,3)+(0,3)
- Chip 2: (2,0)+(3,0)+(0,0)+(1,0), (2,1)+(3,1)+(0,1)+(1,1), (2,2)+(3,2)+(0,2)+(1,2), (2,3)+(3,3)+(0,3)+(1,3)
- Chip 3: (3,0)+(0,0)+(1,0)+(2,0), (3,1)+(0,1)+(1,1)+(2,1), (3,2)+(0,2)+(1,2)+(2,2), (3,3)+(0,3)+(1,3)+(2,3)

Notice that each chip computes the same mathematical result, just with operands in different orders (addition is commutative, so order doesn't matter).
After this step, all chips have identical data.

**Output:**

After the AllReduce completes, every chip holds the complete reduced result:
- **Final distribution**: `Out = {chip: 4, slice: 256, element: B}`
- **Every chip holds identical data**: The sum of all A values for each B position
  - All chips have: `[sum(A=0..3, B=0), sum(A=0..3, B=1), sum(A=0..3, B=2), sum(A=0..3, B=3)]`

This can be viewed as transforming `[A=4] | [B=4]` to `[Broadcast=4] | [B=4]`:
- The `A` axis has been reduced (eliminated through summation)
- The result is broadcast to all chips (every chip has the complete result)

**Why this example is useful:**

`AllReduce` is the workhorse operation for distributed machine learning:
- **Data parallel training**: Average gradients computed across multiple batches on different chips
- **Model averaging**: Combine parameter updates from multiple workers
- **Synchronization primitive**: Ensure all chips have identical state before proceeding
- **Global statistics**: Compute metrics like mean/max/min across the entire distributed dataset

Key characteristics:
- **Bandwidth efficient**: Each chip only receives data from 3 shuffle operations (not 3 full tensor transfers)
- **Symmetric**: All chips perform the same computation, simplifying implementation
- **Complete replication**: Every chip ends with full result, enabling independent downstream operations
- **Foundation for collectives**: More complex distributed operations build on AllReduce

The rotation-based algorithm shown here scales to any power-of-2 number of chips: for 8 chips, use 7 rotations; for 16 chips, use 15 rotations, etc.


## Implementation Methods for Each Operation

Each operation in chip/cluster reduce maps to specific hardware primitives. Understanding these mappings helps predict performance and resource usage patterns.

### Asymmetric Slice

Chip/cluster asymmetric slice operations extract a subset of data from specific positions in the chip or cluster dimension. The `ParallelCopy` operation implements this by running in the sub-context using the `stos` (Store to SRAM) command. This approach enables selective data extraction without full tensor movement, copying only the elements at positions specified by the slice indices. The sub-context execution ensures that slice operations can overlap with main-context computation, maintaining pipeline efficiency.

### Shuffle

Chip/cluster shuffle redistributes data across chips using DMA operations through HBM. The `DmaCommand` handles intra-chip shuffles by moving data between HBM regions associated with different chips, while `PCIeDmaCommand` extends this capability to inter-chip communication when needed. The HBM-to-HBM transfer pattern avoids unnecessary round-trips through chip-local memory, directly routing data to its destination. Shuffle operations are the primary cost factor in chip/cluster reduce because they involve cross-chip data movement over the interconnect fabric, typically requiring hundreds to thousands of cycles depending on data volume.

### Tensor Addition

Tensor addition combines multiple input tensors element-wise to perform the actual reduction computation. This operation runs in the main context using a two-stage approach: interleaved fetch brings data from multiple tensor instances into the pipeline, and the Vector Engine's binary add operation performs the element-wise summation. The interleaved fetch pattern enables the Vector Engine to process additions efficiently by presenting operands in alternating time steps, avoiding the need for separate accumulation buffers. This main-context execution provides maximum throughput for the arithmetic-intensive reduction phase after data has been properly arranged through slice and shuffle operations.
