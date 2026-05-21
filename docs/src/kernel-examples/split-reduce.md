# Split Reduce


Split reduce handles reductions when a logical reduction axis cannot be mapped to a single continuous hardware dimension.
The axis is split into multiple separate tensor instances that must be fetched independently and then combined.
The fetch uses interleaved fetch and the combination uses Vector Engine binary operations.

## When to Use Split Reduce

Split reduce applies when:

- **You need to split**: A reduction axis is too large to fit in VRF (8KB per slice) as a single tensor, requiring the logical axis to be split into multiple physical tensor instances.
- **Data is already split**: Multiple tensor instances independently hold different portions of the same logical reduction axis (e.g., from different model layers, experts, or temporal segments).
- **Avoiding cross-chip communication**: Data resides on the same chip/cluster but in separate memory allocations, making interleaved fetch more efficient than DMA-based approaches.

As a multi-instance fetch-and-combine pattern, split reduce fits into TCP's reduction hierarchy between slice-level and chip-level reductions:

- **Packet reduce**: Within a single packet (Packet Reducer)
- **Time reduce**: Across time dimension (Time Reducer)
- **Slice reduce**: Across slices within a cluster (Inter-Slice Reducer)
- **Split reduce**: Across multiple independent tensor instances, using *interleaved fetch* (alternating loads from separate tensor instances) combined with Vector Engine binary ops
- **Chip/Cluster reduce**: Across chips or clusters (DMA + interleaved fetch + Vector Engine binary op)


## Implementation: Interleaved Fetch

The fetch pattern introduces an interleave dimension `I` that indexes the separate tensor instances, creating a time-interleaved stream that the Vector Engine reduces:

```rust,ignore
// Two tensor instances to be reduced together
let tensor_0: DmTensor<bf16, m![1], m![1], m![1], m![A, B]> = ...;
let tensor_1: DmTensor<bf16, m![1], m![1], m![1], m![A, B]> = ...;

// Interleaved fetch creates alternating time stream: I=2 dimension
let interleaved: TuTensor<bf16, m![1], m![1], m![1],
    m![I: 2, A], m![B]
> = ctx.main.begin_interleaved().fetch(&tensor_0, &tensor_1);

// Vector Engine reduction combines the I dimension
let reduced: TuTensor<bf16, m![1], m![1], m![1],
    m![A], m![B]
> = interleaved.reduce_add(axis: I);
```

The interleaved fetch alternates between tensor instances in the time dimension: `time[0]` holds data from `tensor_0`, `time[1]` from `tensor_1`, `time[2]` from `tensor_0` again, and so on. The Vector Engine performs binary operations (add, max, min) across the interleave dimension to complete the reduction.

## Example 1: Layer Normalization Split Reduction

Layer normalization drives a split reduce when the Hidden dimension exceeds VRF capacity.
The feature dimension must be split into multiple chunks that are processed separately and then combined.
Layer normalization computes statistics (mean, variance) over the entire feature dimension, so the full Hidden axis must be reduced.

**The Problem:**
Layer normalization requires computing the mean and variance of all features for each token.
The formula is:
```text
output = (input - mean) / sqrt(variance + epsilon)
```
where mean and variance are computed over the entire `Hidden` dimension.

When `Hidden` is very large (like 8,192 elements), the tensor won't fit in the 8KB VRF, so we cannot reduce it in a single operation.

**Input:**
A 3D tensor representing transformer activations:
- **Shape**: `[Batch=32, SeqLen=128, Hidden=8192]`
- **Data type**: `bf16` (2 bytes per element)
- **Total size**: 32 × 128 × 8192 × 2 bytes = 64 MB
- **Per-token slice**: For each of 4,096 tokens (32 × 128), we have 8,192 features = 16 KB per token
- **VRF constraint**: Only 8KB per slice ≈ 4,096 `bf16` elements
- **Problem**: Cannot load all 8,192 features for a token simultaneously

**Solution Strategy:**
Split the `Hidden` dimension into two 4,096-element chunks:
- **Chunk 0**: `[Batch=32, SeqLen=128, Hidden_0=4096]` - first half of features
- **Chunk 1**: `[Batch=32, SeqLen=128, Hidden_1=4096]` - second half of features
- Each chunk = 4,096 elements × 2 bytes = 8KB, fits in VRF

### Step-by-Step Execution

#### Step 1: Compute Partial Statistics

First, compute statistics for each chunk independently:

```rust,ignore
// Chunk 0: Hidden dimensions 0..4096
let chunk_0: DmTensor<bf16, m![1], m![1], m![1], m![Batch, SeqLen, Hidden_0: 4096]> = ...;

// Chunk 1: Hidden dimensions 4096..8192
let chunk_1: DmTensor<bf16, m![1], m![1], m![1], m![Batch, SeqLen, Hidden_1: 4096]> = ...;

// Compute sum for each chunk (using Packet Reducer + Inter-Slice Reducer)
let sum_0: DmTensor<f32, m![1], m![1], m![1], m![Batch, SeqLen]> = chunk_0.reduce_sum(axis: Hidden_0);
let sum_1: DmTensor<f32, m![1], m![1], m![1], m![Batch, SeqLen]> = chunk_1.reduce_sum(axis: Hidden_1);
```

#### Step 2: Interleaved Fetch and Combine

Use split reduce to combine the partial sums:

```rust,ignore
// Fetch both chunks in interleaved pattern
let interleaved_sums: TuTensor<f32, m![1], m![1], m![1],
    m![I: 2, Batch, SeqLen], m![1]
> = ctx.main.begin_interleaved().fetch(&sum_0, &sum_1);

// Vector Engine adds across I dimension to get total sum
let total_sum: TuTensor<f32, m![1], m![1], m![1],
    m![Batch, SeqLen], m![1]
> = interleaved_sums.reduce_add(axis: I);

// Compute mean: total_sum / Hidden
let mean = total_sum * (1.0 / 8192.0);  // Vector Engine scalar multiply
```

#### Step 3: Compute Variance

Using the `mean` computed in Step 2, compute and combine partial variance calculations:

```rust,ignore
// Compute squared differences for each chunk
let sq_diff_0 = (chunk_0 - mean).square().reduce_sum(axis: Hidden_0);
let sq_diff_1 = (chunk_1 - mean).square().reduce_sum(axis: Hidden_1);

// Split reduce to combine variance contributions
let interleaved_vars: TuTensor<f32, m![1], m![1], m![1],
    m![I: 2, Batch, SeqLen], m![1]
> = ctx.main.begin_interleaved().fetch(&sq_diff_0, &sq_diff_1);

let total_variance = interleaved_vars.reduce_add(axis: I);
let std = total_variance.sqrt();
```

**Output:**

The three steps produce the statistics needed for layer normalization:
- **Mean**: `[Batch=32, SeqLen=128]` - one mean value per token, representing the average of all 8,192 features
- **Standard deviation**: `[Batch=32, SeqLen=128]` - one std value per token
- **Result**: Use these statistics to normalize each token's 8,192 features:
  ```text
  normalized_chunk_0 = (chunk_0 - mean) / std
  normalized_chunk_1 = (chunk_1 - mean) / std
  ```

Computing statistics in two separate chunks produces the same mathematical result as computing over all 8,192 features at once:
- **Mathematically**: `mean([a,b,c,d,e,f]) = (sum(a,b,c) + sum(d,e,f)) / 6`
- **In practice**: `mean([Hidden_0, Hidden_1]) = (sum(Hidden_0) + sum(Hidden_1)) / 8192`

Split reduce computes global statistics despite VRF capacity limits.

### Hardware Mapping

The split reduce operation maps to hardware as follows:

| Operation | Hardware Component | Cycles |
|-----------|-------------------|--------|
| Fetch chunk_0 | Fetch Engine | ~1 cycle per 32-byte flit |
| Fetch chunk_1 | Fetch Engine (interleaved) | ~1 cycle per 32-byte flit |
| Interleave dimension creation | Fetch Sequencer | 0 (structural transformation) |
| Binary add across I | Vector Engine | 1 cycle per packet |


### Performance Analysis

**Total cycles for split reduce**:
- Fetch both tensors: `2 * (Batch * SeqLen * ceil(Hidden / flit_elements))` cycles
- Vector Engine reduction: `(Batch * SeqLen)` cycles
- Total: Dominated by fetch time, ~8K cycles for this example

**Bottleneck**: Memory bandwidth for fetching both tensor instances sequentially.

**Optimization**: Restructure the computation to avoid splitting the reduction axis when possible. If the axis must be split, minimize the number of split instances.

## Example 2: Batch Normalization Across Split Batches

When the batch dimension is split across two independent allocations, split reduce combines the per-allocation statistics to produce global batch normalization results.
Batch normalization computes statistics across the entire batch dimension, so all allocations must be reduced together.

### Problem Setup

- **Input**: `[Batch_0 = 256, ...], [Batch_1 = 256, ...]` (two separate batch tensors)
- **Reduction goal**: Compute mean and variance across all 512 examples
- **Constraint**: Cannot allocate single tensor for 512 batches due to memory limits

### Execution Pattern

```rust,ignore
// Two batch allocations
let batch_0: DmTensor<bf16, m![1], m![1], m![1], m![Batch_0: 256, C, H, W]> = ...;
let batch_1: DmTensor<bf16, m![1], m![1], m![1], m![Batch_1: 256, C, H, W]> = ...;

// Compute per-batch statistics (reduce over H, W)
let batch_stats_0 = batch_0.reduce_mean(axis: [H, W]);  // [Batch_0=256, C]
let batch_stats_1 = batch_1.reduce_mean(axis: [H, W]);  // [Batch_1=256, C]

// Split reduce to combine batch statistics
let interleaved: TuTensor<f32, m![1], m![1], m![1],
    m![I: 2, Batch: 256, C], m![1]
> = ctx.main.begin_interleaved().fetch(&batch_stats_0, &batch_stats_1);

// Compute global statistics across all batches
let global_mean = interleaved.reduce_mean(axis: I);  // Average the two batch means
```

This pattern extends naturally to more than two splits by increasing the interleave dimension: `I: 4` for four splits, etc.

## Example 3: Mixture of Experts Partial Reduction


### Problem Setup

- **Expert outputs**: Multiple tensors from different expert evaluations
- **Routing weights**: Weights determining how much each expert contributes
- **Goal**: Weighted sum across expert outputs

### Execution Pattern

```rust,ignore
// Expert outputs from separate evaluations (simplified: 2 experts)
let expert_0_output: DmTensor<bf16, m![1], m![1], m![1], m![Tokens, Hidden]> = ...;
let expert_1_output: DmTensor<bf16, m![1], m![1], m![1], m![Tokens, Hidden]> = ...;

let routing_weights: [f32; 2] = [0.7, 0.3];  // Per-expert weights

// Apply routing weights during fetch using zero-point arithmetic or scaling
let weighted_0 = expert_0_output * routing_weights[0];
let weighted_1 = expert_1_output * routing_weights[1];

// Split reduce to combine weighted expert contributions
let interleaved: TuTensor<bf16, m![1], m![1], m![1],
    m![I: 2, Tokens], m![Hidden]
> = ctx.main.begin_interleaved().fetch(&weighted_0, &weighted_1);

let combined_output = interleaved.reduce_add(axis: I);
```


## Example 4: Temporal Reduction Across Windows


### Problem Setup

- **Input**: Video frames or sequence tokens split into temporal chunks
- **Goal**: Compute global statistics across all chunks
- **Constraint**: Cannot load all chunks simultaneously due to memory limits

### Execution Pattern

```rust,ignore
// Temporal chunks
let chunk_t0: DmTensor<bf16, m![1], m![1], m![1], m![Time_0: 128, Features]> = ...;
let chunk_t1: DmTensor<bf16, m![1], m![1], m![1], m![Time_1: 128, Features]> = ...;
let chunk_t2: DmTensor<bf16, m![1], m![1], m![1], m![Time_2: 128, Features]> = ...;
let chunk_t3: DmTensor<bf16, m![1], m![1], m![1], m![Time_3: 128, Features]> = ...;

// Compute per-chunk max (e.g., for max pooling over time)
let max_t0 = chunk_t0.reduce_max(axis: Time_0);  // [Features]
let max_t1 = chunk_t1.reduce_max(axis: Time_1);  // [Features]
let max_t2 = chunk_t2.reduce_max(axis: Time_2);  // [Features]
let max_t3 = chunk_t3.reduce_max(axis: Time_3);  // [Features]

// Split reduce with I=4 to find global maximum
let interleaved: TuTensor<bf16, m![1], m![1], m![1],
    m![I: 4], m![Features]
> = ctx.main.begin_interleaved().fetch(&max_t0, &max_t1, &max_t2, &max_t3);

let global_max = interleaved.reduce_max(axis: I);
```

## Comparison with Other Reduction Methods

Split reduce has two primary alternatives: slice reduce/Inter-Slice Reducer for same-tensor distributions, and chip/cluster reduce for cross-chip data.
The choice among them depends on data location, tensor shape, and whether data can be merged into a single allocation.

### Split Reduce vs. Slice Reduce (Inter-Slice Reducer)

| Aspect | Split Reduce | Slice Reduce (Inter-Slice Reducer) |
|--------|--------------|-------------------|
| Data layout | Multiple independent tensors | Single tensor across slices |
| Fetch pattern | Interleaved fetch from multiple sources | Single contiguous fetch |
| Reduction hardware | Vector Engine binary ops | Inter-Slice Reducer |
| Typical cycles | ~2× fetch time | ~256 cycles (slice reduction) |
| Use case | Data cannot fit in single tensor | Data distributed across hardware |

**Prefer split reduce**: Multiple tensor instances that cannot be merged into a single tensor due to memory allocation constraints, but all reside on the same chip/cluster.

**Prefer slice reduce**: Allocate a single tensor that spans slices, allowing the hardware to handle distribution automatically.

### Split Reduce vs. Chip/Cluster Reduce

| Aspect | Split Reduce | Chip/Cluster Reduce |
|--------|--------------|---------------------|
| Data location | Same chip/cluster | Across chips/clusters |
| Communication | Local memory fetch | DMA over chip interconnect |
| Overhead | Minimal (interleaved fetch) | Significant (DMA + synchronization) |
| Bandwidth | SRAM bandwidth | Chip interconnect bandwidth |

**Prefer split reduce**: All data resides on the same chip, even if in separate allocations.

**Prefer chip/cluster reduce**: Data is distributed across physically separate processing units requiring cross-chip communication.

## Implementation Methods

The split reduce operation maps to the following hardware primitives:

- **Interleaved fetch**: Fetch Engine with `begin_interleaved()` mode, creating the `I` interleave dimension
- **Reduction across I**: Vector Engine binary operations (add, max, min) configured to reduce the interleave axis
- **Alternative for 2-way split**: Can use binary operation directly without explicit interleave dimension

### Two-Instance Optimization

For the common case of splitting into exactly two instances, the Vector Engine can perform the reduction without creating an explicit interleave dimension:

```rust,ignore
// Direct binary operation for 2-way split
let sum_0: TuTensor<f32, m![1], m![1], m![1], m![A], m![B]> = ...;
let sum_1: TuTensor<f32, m![1], m![1], m![1], m![A], m![B]> = ...;

// Fetch both and add in one operation
let total = sum_0.binary_add(sum_1);  // No interleave dimension needed
```

This optimization reduces overhead by combining fetch and reduction into a single pipelined operation.

## Performance Considerations

### Cycle Analysis

Split reduce cycle count is dominated by fetch time, with Vector Engine cycles and pipeline overlap as secondary factors:

- **Fetch cycles**: `N_splits * fetch_cycles_per_tensor`
- **Vector Engine cycles**: `Time_dim_size * cycles_per_packet` (typically 1 cycle per packet)
- **Pipeline overlap**: Fetch and VE operations can overlap when possible

**Total cycles** ≈ `N_splits * fetch_cycles + max(0, VE_cycles - pipeline_overlap)`

### Memory Bandwidth

Split reduce consumes memory bandwidth proportionally to the number of splits:

- **2-way split**: 2x memory bandwidth vs. single tensor
- **4-way split**: 4x memory bandwidth vs. single tensor

**Optimization**: Minimize the number of splits by maximizing individual tensor size within VRF capacity.

### Comparison to Alternatives

For a reduction requiring combining N tensor instances:

| Method | Cycles | Memory BW | Complexity |
|--------|--------|-----------|------------|
| Split reduce (interleaved) | ~N * fetch + VE | N * tensor_size | Low |
| Sequential fetch + accumulate | ~N * (fetch + VE) | N * tensor_size | Medium |
| DMA to single buffer + reduce | DMA + single_reduce | N * tensor_size | High |

Split reduce with interleaved fetch provides the best balance of performance and implementation simplicity for same-chip reductions.

## Constraints and Limitations

### Hardware Constraints

- **Interleave dimension size**: Limited by Fetch Engine capabilities
- **Tensor alignment**: All tensor instances must have compatible shapes for interleaving
- **VRF capacity**: After interleaving, the combined tensor must fit in VRF (8KB per slice)


### When Split Reduce Is Not Optimal

- **Single tensor possible**: Data fits in one tensor allocation, use slice reduce (Inter-Slice Reducer) instead
- **Cross-chip reduction needed**: Data spans chips, use chip/cluster reduce with DMA
- **Very large split count**: Beyond ~8 splits, consider alternative memory management strategies

### Best Practices

- **Minimize splits**: Design tensor allocations to minimize the number of splits required
- **Power-of-2 splits**: Use 2, 4, or 8 splits when possible for optimal hardware utilization
- **Reuse reduction results**: Cache split reduce results when the same combination is needed multiple times
- **Consider memory layout**: Organize tensor allocations to enable efficient interleaved fetch patterns
