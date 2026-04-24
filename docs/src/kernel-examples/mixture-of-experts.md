# Mixture of Experts

<!-- > **TODO(youseok.yang)**: The Gating/Router (Section 1), Sparse Expert Computation (Section 3), Combine (Section 4), and Blockwise Execution (Section 3.3–3.6) sections are written in algorithmic prose with no Virtual ISA code. -->
<!-- > Add kernel code for at least one complete execution cycle — for example, the sparse MLP step (Section 3.5) showing how a block of tokens is fed through one expert's up/down projection using the Contraction Engine. -->

Mixture of Experts (MoE) scales model capacity by routing each token to only K of E experts rather than all of them; this sparse activation allows many parameters while keeping inference cost manageable.
This example shows how to implement MoE on TCP hardware, focusing on two key challenges: replacing control-flow-based TopK routing with branchless matrix operations, and executing sparse expert computations blockwise.

## Background: Basic FFN

To understand MoE, first consider the basic FFN (Feed-Forward Network) in transformer blocks.
The following describes FFN with only up/down projection, without gate projection:

- **Input**
    - `x_ffn_norm: T x D`
- **Weights**
    - `W_up: D x F` (up projection)
    - `W_down: F x D` (down projection)
- **Output**
    - `ffn_out: T x D`
- **Operations**
    - **Up projection**:
        - `up = einsum(x_ffn_norm, W_up)`
        - `(T x D), (D x F) -> T x F`
    - **Down projection**:
        - `ffn_out = einsum(up, W_down)`
        - `(T x F), (F x D) -> T x D`

## MoE Structure

MoE replaces a single FFN with `E` independent FFNs called experts.
Each expert has its own weights:
- `W_up[0], W_up[1], ..., W_up[E-1]`
- `W_down[0], W_down[1], ..., W_down[E-1]`

Computing all experts would increase computation by `E` times.
To avoid this, MoE uses a router to select only the `Top-K` most suitable experts per token, enabling sparse computation.

## Model Parameters

The following arguments define an MoE layer:
- `T`: number of tokens
    - prefill: `T = B * S_in`
    - decode: `T = B`
- `D`: hidden dimension
- `F`: intermediate dimension of ffn up projection result
- `E`: number of total experts (typically 128)
- `K`: number of experts to apply ffn
    - `llama4`: 1, `gpt-oss`: 4, `qwen3`: 8

## MoE Processing Steps

MoE processing consists of three main stages: routing (selecting which experts to use), sparse expert computation (applying the selected experts), and combining (merging expert outputs with routing weights).

### 1. Gating (Router) & Top-K Selection

The router calculates a score for each expert for every token, determining which experts should process each token:
- **Input**
    - `x_norm: T x D`
- **Weight**
    - `W_router: D x E` (Gating network weights)
- **Output**
    - `scores: T x E`
- **Operation**
    - `scores = einsum(x_norm, W_router)`
    - `(T x D), (D x E) -> T x E`
    - Calculate the score (Logit) for `E` Experts per token

### 2. `Top-K` Selection

This step selects the `Top-K` Experts based on router scores and calculates the weight for each selected Expert:

- **Input**
    - `scores: T x E`
- **Outputs**
    - `topk_indices: T x K` (selected Expert ID per token)
    - `routing_weights: T x K` (weight of selected Expert per token)
- **Operations**
    - **`Top-K` Selection**:
        - `raw_weights, topk_indices = topk(scores, K)`
        - Extract the `K` Expert indices and scores with the highest scores per token
    - **Softmax Normalization**:
        - `routing_weights = softmax(raw_weights)`
        - Convert the selected `K` scores to probability values (sum is 1 per token)
        - `softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for j in 0..K)`

The output for each token `t` consists of:
- `topk_indices[t, :]`: `K` Expert IDs (`0 <= e < E`)
- `routing_weights[t, :]`: weights of those Experts (sum is 1)

### 3. Sparse Expert Computation

Only selected Experts perform computation, making this stage sparse.
A total of `T * K` Expert calls occur, but each Expert only computes for the tokens that selected it.

For each token `t in [0, T-1]` and selected Expert `k in [0, K-1]`:

- **Selected Expert ID**: `e = topk_indices[t, k]`
- **Input**
    - `x_norm[t]: D` (input of token `t`)
- **Weights** (weights of Expert `e`)
    - `W_up[e]: D x F`
    - `W_down[e]: F x D`
- **Output**
    - `y[t, k]: D` (`k`-th Expert output of token `t`)
- **Operations**
    - **Up projection**:
        - `up = einsum(x_norm[t], W_up[e])`
        - `D, (D x F) -> F`
    - **Down projection**:
        - `y[t, k] = einsum(up, W_down[e])`
        - `F, (F x D) -> D`

The results for all `(t, k)` pairs are collected into `y_experts: T x K x D`.

### 4. Weighted Sum (Combine)

The final step combines the `K` Expert outputs using the routing weights calculated earlier:

- **Inputs**
    - `y_experts: T x K x D`
    - `routing_weights: T x K` (weight of each Expert)
- **Output**
    - `ffn_out: T x D`
- **Operations**
    - `ffn_out = einsum(y_experts, routing_weights)`

The result is that each token receives the weighted average output of its selected `K` Experts.

## MoE Implementation on TCP

Implementing MoE efficiently on TCP requires bridging the gap between the model's logical structure and hardware constraints.
This section describes the techniques needed to achieve high performance.

### 1. Overview and Design Philosophy

#### 1.1. Bridging Logical and Physical Execution

Two fundamental challenges arise when implementing MoE on TCP:

- **Challenge 1: Conflict between control flow and parallel structure**
    - Problem: General `Top-K` algorithms use branch statements where the execution path varies depending on data values.
    Such branch statements cause performance degradation in SIMT-based accelerators that process thousands of elements with a single instruction.
    - Solution: Completely removing control flow and using Branchless `Top-K` technique with matrix operations and bit manipulation is essential.
- **Challenge 2: Gap between logical Routing and physical execution**
    - Problem: Logically, MoE is a process where each token finds the Expert that suits it (Token-centric).
    However, if implemented as is, memory access becomes irregular and the number of tokens to process per Expert changes dynamically, reducing TCP compiler efficiency.
    - Solution: The perspective must be shifted to a method where the Expert becomes the subject and collects tokens (Expert-centric).

#### 1.2. Core Techniques for TCP Implementation

Two core techniques address these challenges:

- **Branchless `TopK`**: Performs routing via matrix operations only, eliminating all control flow
- **Blockwise execution**: Processes only selected Experts with data packed in fixed-size `Block` units

The following sections describe each technique in detail.

### 2. Branchless `TopK`

Branchless `TopK` replaces control-flow-based sorting with pure matrix operations.
This approach consists of three stages: bit packing to combine score and index, parallel ranking to determine order, and filtering to extract the top `K` results.

#### 2.1. Bit Packing (Combining Score and Index)

The Vector Engine pipeline operates on all 256 slices in lockstep, so any operation whose address or control path depends on runtime data values must be replaced with a fixed sequence of matrix operations.
Bit packing bundles score and index into a single value so the Expert ID is preserved when scores are reordered during sorting:

- **Inputs**
    - `scores: T x E`
    - `Index_expert: E`
        - `Index_expert(e) = e where e = 0, 1, 2, ..., E - 1`
- **Output**
    - `Packed_Value: T x E`
        - Tensor with (score, index) packed.
    - `Packed_Value_cmp: T x E`
        - Tensor with (score, index) packed, preprocessed to enable comparison of score magnitude using integer comparison.
- **Operation**
    - **Packing**
        - Place Expert Score (e.g., `bf16`) in the upper bits and Expert Index (e.g., `int16`) in the lower bits to create a single 32-bit integer (or float).
        - `Packed_Value_unprocessed = (Score << 16) | Index`
        - Processed in Vector Engine.
    - **Comparison Trick**
        - This preprocessing enables magnitude comparison of score values using simple integer comparison.
        - Bit Flipping preprocessing solves the problem of negative magnitude relationships being reversed when comparing float values as integer values.
This enables accurate Top-K selection with only integer comparators.
        - ```rust,ignore
          Packed_Value_cmp = if Packed_Value >= 0 {
              Packed_Value
          } else {
              Packed_Value ^ 0x7fff0000
          }
          ```

#### 2.2. Parallel Ranking (All-to-All Comparison)

Parallel ranking determines the order of all experts simultaneously instead of sequential sorting.
Although this requires `E x E` comparisons, TCP efficiency remains high because only matrix operations are used without control flow:

- **Input**
    - `Packed_Value_cmp: T x E`
        - 32-bit Packed Tensor with Comparison Trick applied.
- **Output**
    - `Rank: T x E`
        - Rank of each Expert (0-based rank). Higher scores are closer to 0.
- **Operations**
    - **Broadcast & Compare**
        - Replicate (Tile) `Packed_Value_cmp` along the `E` axis to expand to `T x E x E` shape.
Compare magnitude relationships for all Expert pairs `(i, j)`.
        - `Compare[t, i, j] = 1 if Packed_Value_cmp[t, j] > Packed_Value_cmp[t, i] else 0`
        - Meaning: "Is Expert `j`'s score higher than Expert `i`'s?"
    - **Rank Calculation (ReduceSum)**
        - Sum along the `E` (comparison target) axis to calculate rank.
        - `Rank[t, i] = sum(Compare[t, i, j] for j in 0..E)`
        - Meaning: "The total number of Experts with higher scores than me" becomes my rank.

#### 2.3. Filtering & Unpacking

Filtering extracts the top `K` entries based on rank, then unpacking separates the packed scores and indices:

- **Inputs**
    - `Rank: T x E`
    - `Packed_Value: T x E`
        - Note: The original Packed Value before Comparison Trick was applied must be used to restore accurate Score/Index later.
- **Outputs**
    - `TopK_Indices: T x K`
    - `TopK_Scores: T x K`
    - `routing_weights: T x K` (weights for K selected experts per Token)
- **Operations**
    - **Filtering (`FilterCompaction`)**
        - Only elements satisfying the `Top-K` condition (`Rank < K`) are kept.
        - `Mask[t, i] = 1 if Rank[t, i] < K else 0`
        - Only `Packed_Value` at positions where Mask is True are collected and compressed to `T x K` size.
        - Result: `Selected_Packed: T x K`
        - Uses the filter function of Vector Engine.
    - **Unpacking**
        - Restore scores and indices through bit operations from the selected 32-bit values.
        - Score Extraction: `TopK_Scores = Selected_Packed >> 16` (then reinterpreted as `bf16` type)
        - Index Extraction: `TopK_Indices = Selected_Packed & 0xffff`
    - **Softmax Normalization**
        - Softmax is applied to the extracted `Top-K` Scores to calculate final weights. This is used in the Combine stage later.
        - `routing_weights[t, k] = exp(TopK_Scores[t, k]) / sum(exp(TopK_Scores[t, j]) for j in 0..K)`

### 3. Blockwise Execution

Blockwise execution physically rearranges data based on `Top-K` routing decisions while satisfying TCP's static shape constraints. This section describes how to handle dynamic token-to-expert assignments efficiently.

#### 3.1. Problem: Dynamic Shape & Memory Explosion

The core challenge is that the number of tokens `L_e` assigned per Expert varies dynamically depending on the input.
In the worst case, if all tokens are concentrated on a specific Expert, `L_e ~ T`.

Two approaches address this challenge:

- **Naive Solution**: Allocating a buffer of maximum size `T` for all Experts requires memory of `E x T x D` size, most of which is wasted as padding.
- **Blockwise Solution**: Instead of variable length `L_e`, manage data in fixed-size `Block` (`B`) units to optimize memory usage to approximately `T x K` level.

#### 3.2. Grid Size Calculation

Grid size determines how many blocks are needed to process all tokens.
Tokens for the same Expert are grouped into blocks of `B` tokens, enabling blockwise computation with a single expert loaded.

The total number of blocks needed (`Grid Size`, `G`) is calculated as the sum of blocks required per expert:
- Number of blocks allocated to Expert `e`
    - Number of tokens allocated to `e`: `Count_e`
    - Number of blocks: `ceil(Count_e / B)`
- `G = sum(ceil(Count_e / B) for e in 0..E)`

The compiler calculates the worst-case `G` value and allocates memory space.
At runtime, sparse operations skip execution for empty Grids.

In the worst case where all Experts include a grid containing only one token, `(T*K - E) / B + E` Grids are required.

#### 3.3. Index Generation (`Cumsum`-based Address Calculation)

Index generation computes the destination address for each token using `cumsum`-based parallel address calculation.
(`Cumsum` is implemented in the Vector Engine using branch logging; see [Section 4](#4-cumsum-implementation-on-npu) for the hardware implementation.)
This approach avoids loops and enables efficient parallel execution:
- **Inputs**
    - `TopK_Indices: T x K`
    - `Expert_Indices: E = [0, 1, ..., E-1]`
    - `Block_Range: G = [0, 1, ..., G-1]` (sequence of maximum block count, e.g., 32)
- **Outputs**
    - `Scatter_Idx: T x K` (final 1D address where each token will move)
    - `Expert_IDs: G` (Expert number each Block is responsible for)
- **Operations**
    - **Mask Generation (One-Hot)**
        - Convert indices to computable mask form.
        - `Expert_Mask: T x K x E = one_hot(TopK_Indices, depth=E)`
    - **Histogram**
        - Sum the masks to count the number of tokens allocated per Expert.
        - `Count: E = reduce_sum(Expert_Mask, axis: (T, K))`
    - **Block calculation**
        - Calculate the number of Blocks needed for each Expert.
            - `Num_Blocks: E = ceil(Count / B)`
    - **Global offset Calculation**
        - Through Cumsum, obtain the Block Start Index where each Expert starts in the entire Grid (`G`).
        - `Global_Offset: E = cumsum(Num_Blocks) - Num_Blocks`
    - **Local Offset Calculation**
        - Using Mask and Cumsum, calculate what position each token is in the Expert's queue.
        - `Cumsum_Mask: T x K x E = cumsum(Expert_Mask, axis: (T, K))`
        - `Token_Rank: T x K = gather(Cumsum_Mask, index: TopK_Indices)`
        - `Local_Offset: T x K = Token_Rank - 1`
    - **Expert ID expansion**
        - `Diff: E x G = Num_Blocks - Block_Range`
        - `Grid: E x G`
            - ```rust,ignore
              Grid(e, i) = if Diff(e, i) > 0 {
                  Expert_Indices(e)
              } else {
                  -1
              }
              ```
        - `Expert_IDs: G = filter_compaction(Grid, condition=(Grid >= 0))`
        - e.g.)
            - expert 0: 2 blocks, expert 1: 3 blocks, expert 3: 3 blocks
            - Diff[0] = [2, 1, 0, -1, -2, ...], Diff[1] = [3, 2, 1, 0, -1, ...]: has positive terms equal to the number of allocated blocks per expert.
            - Grid[0] = [0, 0, -1,-1, ...], Grid[1] = [1, 1, 1, -1, -1, ...]: has expert id equal to the number of allocated blocks per expert.
            - Expert_IDs = [0, 0, 1, 1, 1, 3, 3, 3]: Filter only values >= 0 (expert id) from Grid.

    - **Address Synthesis**
        - `Scatter_Idx = (Global_Offset * B) + Local_Offset`
        - Calculate which block and which position within the block each of the `T` tokens corresponds to. `Scatter_Idx in [0, G * B)`

#### 3.4. Dispatch (Blockwise Scatter)

Dispatch physically rearranges tokens using the computed addresses, placing each token in its designated block position:

- **Input**
    - `x_norm: T x D` (Input after Attention and norm)
    - `Scatter_Idx: T x K` (Final 1D address where each token will move)
- **Output**
    - `x_blocked: G x B x D` (Rearranged Blocked Tensor)
- **Operation**
    - **Scatter**
        - Place tokens `x_norm` at `Scatter_Idx` positions.
    
#### 3.5. Sparse Computation (Weight Gather)

Sparse computation applies Expert weights to the sorted Blocks.
The key insight is that weights are gathered only for Experts that have assigned tokens:

- **Inputs**
    - `x_blocked: G x B x D`
    - `Expert_IDs: G` (Expert number each Block is responsible for)
- **Output**
    - `y_blocked: G x B x D`
- **Operations**
    - **Weight Gather**
        - Using `Expert_IDs` as indices, only the necessary weights are fetched.
        - `W_gathered_up: G x D x F = gather(W_up, index: Expert_IDs)`
        - `W_gathered_down: G x F x D = gather(W_down, index: Expert_IDs)`
    - **Sparse MLP**
        - Operations are performed only for valid Blocks (`G`).
        - `up: G x B x F = einsum(x_blocked, W_gathered_up)`
        - `y_blocked: G x B x D = einsum(up, W_gathered_down)`

#### 3.6. Combine (Weighted Sum)

Combine restores results to original token order and applies Routing probabilities.
This is the final step that produces the MoE layer output:

- **Inputs**
    - `y_blocked: G x B x D`
    - `Scatter_Idx: T x K`
    - `routing_weights: T x K`
- **Output**
    - `moe_out: T x D` (Final MoE layer output)
- **Operations**
    - **Gather**
        - Using `Scatter_Idx` in reverse, results are fetched from `y_blocked` in the original token order.
        - `y_restored: T x K x D = gather(y_blocked, index: Scatter_Idx)`
    - **Weighted Sum**
        - The final output is summed by multiplying with `routing_weights` obtained from the Top-K process.
        - `y_weighted: T x K x D = einsum(y_restored, routing_weights)`
        - `moe_out: T x D = reduce_sum(y_weighted, axis: K)`

### 4. `Cumsum` Implementation on TCP

`Cumsum` is a key primitive used in index generation.
On TCP, it is implemented in Vector Engine using the following approach:

1. Create a static branch logger: For the axis (of size n) over which the sum is computed,

```rust,ignore
branch(i) = if i == 0 {
    0
} else if i < n - 1 {
    1
} else {
    2  // i == n - 1
}
```

2. Configure the Vector Engine as follows:
```rust,ignore
add %mainstream, OperandRead(branch = 1, 2)
WriteOperand(branch = 0, 1)
```
