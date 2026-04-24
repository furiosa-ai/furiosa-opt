# Transformer Architecture

<!-- > **TODO(youseok.yang)**: This page describes each transformer operation in terms of mathematical shapes and hardware engine annotations, but shows no Virtual ISA code. -->
<!-- > Add kernel code for representative operations following the style of `fetch-commit-engine.md` — at minimum: one einsum (e.g., QKV projection showing the Contraction Engine axis mapping) and one Vector Engine operation (e.g., RMSNorm showing the elementwise sequence). -->

This page uses Llama 3 70B as a concrete example to show how each transformer operation maps to specific TCP hardware components.
Llama 3 70B implements a decoder-only transformer architecture with two main phases: prefill (input encoding) and decode (token generation).

## Model Parameters

The following parameters define the Llama 3 70B architecture, grouped by category:

**Sequence dimensions** (control input/output length):
- `B`: batch size
- `s_in`: input sequence length
- `s_max`: maximum sequence length/context length
- `s`: total sequence length processed so far (prefill + decode)

**Model size** (vocabulary and layer counts):
- `V = 128256`: vocab size
- `D = 8192`: hidden dimension/size of embedding
- `F = 28672`: intermediate dimension for FFN up projection
- `L = 80`: num layers

**Attention head dimensions** (how attention is partitioned):
- `h_q = 64`: number of query heads
- `h_kv = 8`: number of key/value heads
- `G = 8`: number of attention groups (`= h_q / h_kv`)
- `d_k = 128`: head dimension (equal to `D / h_q`)
- `d_k_prime = 64`: split head dimension for RoPE computation
- `f = 2`: frequency dimension for adjacent heads (`d_k = d_k_prime * f`)

## Prefill Phase

The prefill phase processes the entire input sequence in parallel, outputting the first token while storing computed Key/Value pairs as KV cache.
The transformer block executes on all input tokens provided by the user.
The following subsections describe each step in order.

### 1. Embedding Lookup

Embedding lookup converts input tokens to vector space representations.

- **Input**
    - `input: shape![B, s_in]`
    - Token indices of input text (which vocabulary entry each token corresponds to)
- **Weight**
    - `w_emb: shape![V, D]`
    - Pre-trained embedding value table for each vocabulary entry
- **Output**
    - `x_0: shape![B, s_in, D]`
- **Operation**
    - `x_0 = gather(index: input, table: w_emb)`
    - gather: Operation that reads values from the table using index values specified in the index tensor.
        - Processed by TensorDMA.

### 2. Transformer Layers (repeated L times)

Each transformer layer applies attention and feed-forward operations sequentially.
For each layer `l = 1, ..., L`, perform the following:

#### 2.1. Input Layer Normalization

Input layer normalization stabilizes training by normalizing activations before attention.

- **Input**
    - `x_prev: shape![B, s_in, D]` (layer input from previous layer)
- **Output**
    - `x_norm: shape![B, s_in, D]`
- **Operation**
    - Apply RMSNorm
    - `x_norm = RMSNorm(x_prev)`
    - RMSNorm: Root Mean Square Layer Normalization
        - Processed by Vector Engine.


#### 2.2. Multi-Head Grouped Query Attention (GQA)

Grouped Query Attention (GQA) improves memory efficiency by sharing key/value heads across multiple query heads, reducing KV cache size.

##### 2.2.1. QKV Projection

QKV projection transforms the normalized input into Query, Key, and Value tensors.
- **Input**
    - `x_norm: shape![B, s_in, D]`
- **Weights**
    - `w_q: shape![D, h_q, d_k]`
    - `w_k: shape![D, h_kv, d_k]`
    - `w_v: shape![D, h_kv, d_k]`
- **Outputs**
    - `Q: shape![B, s_in, h_q, d_k]`
    - `K: shape![B, s_in, h_kv, d_k]`
    - `V: shape![B, s_in, h_kv, d_k]`
- **Operations**
    - `Q = einsum(x_norm, w_q)`
    - `K = einsum(x_norm, w_k)`
    - `V = einsum(x_norm, w_v)`
    - matmul corresponds to einsum (= broadcast + elementwise mul + reduce add).
    - elementwise mul: Contraction Engine
    - reduce add
        - packet reduce: Reducer
        - time reduce: Reducer
        - slice reduce: global adder tree
        - split reduce: interleaved fetch + Vector Engine binary op
        - cluster/chip reduce: DMA + interleaved fetch + Vector Engine binary op

##### 2.2.2. Rotary Position Embedding (RoPE)

Rotary Position Embedding (RoPE) applies positional information to Query and Key tensors through rotation transformations.
- **Inputs**
    - `Q: shape![B, s_in, h_q, d_k]`
    - `K: shape![B, s_in, h_kv, d_k]`
    - `d_k = d_k_prime * f`
        - Split the `d_k` axis to apply RoPE rotation in a TCP-friendly manner.
- **RoPE table**
    - `w_rope: shape![s_max, d_k_prime, 2, 2]`
    - Pre-computed table of cos/sin values based on sequence position and head position.
    - RoPE operation groups consecutive pairs among `d_k` values and applies rotation transformation using cos/sin.
    - Store the 2 × 2 matrix representing the cos/sin rotation transformation for TCP-friendly execution.
- **Position**
    - `position: shape![s_in]`
    - `position(i) = i`
- **Outputs**
    - `Q_rope: shape![B, h_q, s_in, d_k]`
    - `K_rope: shape![B, h_kv, s_in, d_k]`
- **Operations**
    - **RoPE table lookup**
        - `t_rope: shape![s_in, d_k_prime, 2, 2] = gather(index: position, table: w_rope)`
    - **Apply RoPE**
        - RoPE computation reduces to a simple einsum operation given the prepared rotation transformation matrix values.
        - **Reshape (noop)**
            - `Q: shape![B, s_in, h_q, d_k] == shape![B, s_in, h_q, d_k_prime, f]`
            - `K: shape![B, s_in, h_kv, d_k] == shape![B, s_in, h_kv, d_k_prime, f]`
            - `t_rope: shape![s_in, d_k_prime, 2, 2] == shape![s_in, d_k_prime, f, 2]`
        - **einsum**
            - `Q_rope = einsum(Q, t_rope)`
                - `(shape![B, s_in, h_q, d_k_prime, f], shape![s_in, d_k_prime, f, 2]) -> shape![B, h_q, s_in, d_k_prime, 2] == shape![B, h_q, s_in, d_k]`
            - `K_rope = einsum(K, t_rope)`
                - `(shape![B, s_in, h_kv, d_k_prime, f], shape![s_in, d_k_prime, f, 2]) -> shape![B, h_kv, s_in, d_k_prime, 2] == shape![B, h_kv, s_in, d_k]`

As a result of RoPE, Q/K values encode relative positional information.

##### 2.2.3. Store in KV Cache

KV cache stores the current layer's Key and Value for reuse during the decode phase, avoiding redundant computation.

- **Inputs**
    - `K_rope: shape![B, h_kv, s_in, d_k]`
    - `V: shape![B, s_in, h_kv, d_k]`
- **KV Cache** (for layer `l`)
    - `kv_cache_l_K: shape![B, h_kv, s_in, d_k]`
    - `kv_cache_l_V: shape![B, h_kv, s_in, d_k]`
- **Operations**
    - `kv_cache_l_K = K_rope`
    - `kv_cache_l_V = V`
    - Cache storage: Stores einsum computation results from DM to HBM, processed by TensorDMA.

##### 2.2.4. Grouped Query Attention Computation

Grouped Query Attention shares each key/value head across multiple query heads.
Each of the 8 KV heads is shared with 8 Query heads (`G = h_q / h_kv = 64 / 8 = 8`).

**2.2.4.1. Attention Scores Computation**

Attention scores measure the relevance between query and key positions using dot product similarity.

- **Inputs**
    - `Q_rope: shape![B, h_q, s_in, d_k]`
    - `K_rope: shape![B, h_kv, s_in, d_k]`
- **Output**
    - `scores: shape![B, h_q, s_in, s_in]`
- **Operations**
    - `scores = (Q_rope @ K_rope.T) / sqrt(d_k)`
    - **Reshape (noop)**
        - The dot product operation can be expressed as einsum.
        Each tensor's shape axes must be precisely distinguished from the output shape perspective to accurately represent the einsum operation semantics.
        - `Q_rope: shape![B, h_q, s_in, d_k] == shape![B, G, h_kv, s_in_q, d_k]`
        - `K_rope: shape![B, h_kv, s_in, d_k] == shape![B, h_kv, s_in_k, d_k]`
    - **einsum**
        - `scores_before_normalize = einsum(Q_rope, K_rope)`
        - `(shape![B, G, h_kv, s_in_q, d_k], shape![B, h_kv, s_in_k, d_k]) -> shape![B, G, h_kv, s_in_q, s_in_k] == shape![B, h_q, s_in, s_in]`
        - The einsum expression shows that `G` was broadcast from `K_rope`, and `d_k` was reduced.
    - **Normalize**
        - `scores = scores_before_normalize / sqrt(d_k)`
        - Division by `sqrt(d_k)` can be computed as multiplication by `1/sqrt(d_k)`. The value `1/sqrt(d_k)` is pre-computed, and the Vector Engine performs simple constant multiplication.


**2.2.4.2. Causal Mask Application**

Causal masking prevents tokens from attending to future positions, preserving autoregressive semantics.
In the prefill phase, `s_in` tokens are processed in parallel, but the `i`-th token must not reference tokens after position `i` to maintain the autoregressive model's semantics.

- **Input**
    - `scores: shape![B, h_q, s_in, s_in]`
    - `attention_mask: shape![s_in, s_in]`
    - `attention_mask(i, j) = true if j <= i, false if j > i`
- **Output**
    - `scores_masked: shape![B, h_q, s_in, s_in]`
- **Operation**
    - `scores_masked(b, h, i, j) = scores(b, h, i, j) if j <= i, -inf if j > i`
    - In the Vector Engine, the `attention_mask` tensor is written to the branch log, then processed through branched operations.

**2.2.4.3. Softmax Application**

Softmax normalizes attention scores into a probability distribution over key positions.

- **Input**
    - `scores_masked: shape![B, h_q, s_in, s_in]`
- **Output**
    - `attn_weights: shape![B, h_q, s_in, s_in]`
- **Operation**
    - `attn_weights = softmax(scores_masked)`
    - Softmax computes the ratio at which each query should reference each token to combine values.
    - Reduces the key-corresponding axis among the two `s_in` dimensions.
    - `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`
    - Processed by Vector Engine

**2.2.4.4. Weighted Sum (Attention Output)**

Weighted sum computes the attention output by combining Value vectors according to attention weights.

- **Inputs**
    - `attn_weights: shape![B, h_q, s_in, s_in]`
    - `V: shape![B, s_in, h_kv, d_k]`
- **Output**
    - `attn_output: shape![B, h_q, s_in, d_k]`
- **Operations**
    - **Reshape (noop)**
        - `attn_weights: shape![B, h_q, s_in, s_in] == shape![B, G, h_kv, s_in_q, s_in_kv]`
        - `V: shape![B, s_in, h_kv, d_k] == shape![B, h_kv, s_in_kv, d_k]`
    - **einsum**
        - `attn_output = einsum(attn_weights, V)`
        - `(shape![B, G, h_kv, s_in_q, s_in_kv], shape![B, h_kv, s_in_kv, d_k]) -> shape![B, G, h_kv, s_in_q, d_k] == shape![B, h_q, s_in, d_k]`
        - The einsum expression shows that `G` was broadcast from `V`, and `s_in_kv` was reduced.

##### 2.2.5. Output Projection

Output projection combines the multi-head attention results into a single hidden state vector.

- **Input**
    - `attn_output: shape![B, h_q, s_in, d_k]`
- **Weight**
    - `w_o: shape![h_q, d_k, D]`
- **Output**
    - `attn_out: shape![B, s_in, D]`
- **Operations**
    - `attn_out = einsum(attn_output, w_o)`
    - `(shape![B, h_q, s_in, d_k], shape![h_q, d_k, D]) -> shape![B, s_in, D]`

##### 2.2.6. Residual Connection

Residual connection adds the attention output to the layer input, improving gradient flow during training.

- **Inputs**
    - `x_prev: shape![B, s_in, D]` (layer input)
    - `attn_out: shape![B, s_in, D]` (attention output)
- **Output**
    - `x_attn: shape![B, s_in, D]`
- **Operation**
    - `x_attn = x_prev + attn_out`
    - elementwise addition: Processed by Vector Engine

#### 2.3. Feed-Forward Network (FFN)

The Feed-Forward Network applies non-linear transformations to each token independently after attention.

##### 2.3.1. Post-Attention Layer Normalization

Post-attention normalization stabilizes activations before the FFN computation.

- **Input**
    - `x_attn: shape![B, s_in, D]`
- **Output**
    - `x_ffn_norm: shape![B, s_in, D]`
- **Operation**
    - `x_ffn_norm = RMSNorm(x_attn)`
    - RMSNorm: Processed by Vector Engine

##### 2.3.2. SwiGLU FFN

SwiGLU (Swish-Gated Linear Unit) is Llama 3's activation function, combining gating with the Swish non-linearity.

- **Input**
    - `x_ffn_norm: shape![B, s_in, D]`
- **Weights**
    - `w_gate: shape![D, F]` (gate projection)
    - `w_up: shape![D, F]` (up projection)
    - `w_down: shape![F, D]` (down projection)
- **Output**
    - `ffn_out: shape![B, s_in, D]`
- **Operations**
    - **Gate projection**:
        - `gate = einsum(x_ffn_norm, w_gate)`
        - `(shape![B, s_in, D], shape![D, F]) -> shape![B, s_in, F]`
    - **Up projection**:
        - `up = einsum(x_ffn_norm, w_up)`
        - `(shape![B, s_in, D], shape![D, F]) -> shape![B, s_in, F]`
    - **SwiGLU activation**:
        - `activated = SiLU(gate) * up`
        - SiLU (Swish): `SiLU(x) = x * sigmoid(x)`
        - `*`: element-wise multiplication
        - Processed by Vector Engine
    - **Down projection**:
        - `ffn_out = einsum(activated, w_down)`
        - `(shape![B, s_in, F], shape![F, D]) -> shape![B, s_in, D]`

##### 2.3.3. Residual Connection

FFN residual connection adds the FFN output to the post-attention output.

- **Inputs**
    - `x_attn: shape![B, s_in, D]` (post-attention output)
    - `ffn_out: shape![B, s_in, D]` (FFN output)
- **Output**
    - `x_l: shape![B, s_in, D]` (final output of layer `l`)
- **Operation**
    - `x_l = x_attn + ffn_out`
    - elementwise addition: Processed by Vector Engine

### 3. Final Layer Normalization

Final layer normalization is applied after passing through all 80 transformer layers.

- **Input**
    - `x_L: shape![B, s_in, D]` (last layer output)
- **Output**
    - `x_final: shape![B, s_in, D]`
- **Operation**
    - `x_final = RMSNorm(x_L)`
    - RMSNorm: Processed by Vector Engine

### 4. Language Model Head (Output Layer)

The language model head converts the hidden state at the last token position into vocabulary logits for next-token prediction.

- **Input**
    - `x_final: shape![B, s_in, D]`
- **Weight**
    - `w_lm_head: shape![D, V]`
    - Typically `w_lm_head = w_emb.T` (weight tying)
- **Output**
    - `logits: shape![B, V]`
- **Operations**
    - **Slice**: In prefill phase, only the last token is used
        - `x_last: shape![B, D] = x_final[:, -1, :]`
        - Extract only the hidden state of the last token to predict the next token
        - Process the slice operation as a simple view operation depending on shape, or use parallel copy to directly read and move a portion of data.
    - **einsum**: Logit computation for vocabulary
        - `logits = einsum(x_last, w_lm_head)`
        - `(shape![B, D], shape![D, V]) -> shape![B, V]`

### 5. Sampling

Sampling converts logit values into a probability distribution and selects the next token.
This process occurs on the Host, not the TCP.

- **Input**
    - `logits: shape![B, V]`
    - `temperature: scalar` (sampling temperature parameter, typically 0.7~1.0)
- **Output**
    - `next_token: shape![B]` (next token index for each batch)
- **Operations**
    - **Temperature scaling**:
        - `logits_scaled = logits / temperature`
        - Higher temperature leads to more diverse token selection, lower temperature leads to more deterministic selection
        - The value `1/temperature` is pre-computed, then processed as constant multiplication in Vector Engine
    - **Softmax**:
        - `probs: shape![B, V] = softmax(logits_scaled)`
        - `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`
        - Apply softmax over the Vocabulary axis (`V`)
    - **Token sampling**:
        - Sample the next token index from the probability distribution `probs`
        - Sampling strategies:
            - Greedy: `next_token = argmax_i(probs_i)`
            - Top-k sampling: Sample only from the top k tokens by probability
            - Top-p (nucleus) sampling: Sample from the smallest token set whose cumulative probability exceeds p

## Decode Phase

The decode phase reuses the same operation sequence as prefill (embedding, transformer layers, LM head, sampling), but operates on a single token at a time, reusing cached KV pairs instead of recomputing them.
The decode phase generates tokens one at a time autoregressively, continuing until an end token (EOS) is produced or the maximum length is reached.
Unlike prefill, decode processes only one token per iteration.

Three characteristics distinguish decode from prefill:

- **Single-token input**: `s_in = 1` (only the most recent output token is used as query)
- **KV cache reuse**: Previously computed Key and Value tensors are reused rather than recomputed
- **Autoregressive generation**: Each token prediction references all previous tokens via the cache

For each decoding step `s = s_prefill + 1, ..., s_max`:

### 1. Embedding Lookup

Embedding lookup converts the previously generated token to its vector representation.

- **Input**
    - `input: shape![B, 1]`
    - Token index sampled in the previous step
- **Weight**
    - `w_emb: shape![V, D]`
- **Output**
    - `x_0: shape![B, 1, D]`
- **Operation**
    - `x_0 = gather(index: input, table: w_emb)`
    - Processed by TensorDMA

### 2. Transformer Layers (repeated L times)

Each transformer layer processes the single token through attention and FFN, reusing cached KV pairs.
For each layer `l = 1, ..., L`, perform the following:

#### 2.1. Input Layer Normalization

Input layer normalization prepares the token for attention computation.

- **Input**
    - `x_prev: shape![B, 1, D]` (layer input from previous layer)
- **Output**
    - `x_norm: shape![B, 1, D]`
- **Operation**
    - `x_norm = RMSNorm(x_prev)`
    - Processed by Vector Engine

#### 2.2. Multi-Head Grouped Query Attention (GQA)

Attention in decode phase computes attention between the current token (query) and all cached tokens (keys/values).

##### 2.2.1. QKV Projection

QKV projection computes Query, Key, and Value for the current token only.

- **Input**
    - `x_norm: shape![B, 1, D]`
- **Weights**
    - `w_q: shape![D, h_q, d_k]`
    - `w_k: shape![D, h_kv, d_k]`
    - `w_v: shape![D, h_kv, d_k]`
- **Outputs**
    - `Q: shape![B, 1, h_q, d_k]`
    - `K_new: shape![B, 1, h_kv, d_k]`
    - `V_new: shape![B, 1, h_kv, d_k]`
- **Operations**
    - `Q = einsum(x_norm, w_q)`
    - `K_new = einsum(x_norm, w_k)`
    - `V_new = einsum(x_norm, w_v)`
    - `(shape![B, 1, D], shape![D, h_q/kv, d_k]) -> shape![B, 1, h_q/kv, d_k]`

##### 2.2.2. Rotary Position Embedding (RoPE)

RoPE applies positional encoding corresponding to the current sequence position.

- **Inputs**
    - `Q: shape![B, 1, h_q, d_k]`
    - `K_new: shape![B, 1, h_kv, d_k]`
- **RoPE table**
    - `w_rope: shape![s_max, d_k_prime, 2, 2]`
- **Position**
    - `position: shape![1]`
    - `position(0) = s` (total sequence length processed so far)
- **Outputs**
    - `Q_rope: shape![B, h_q, 1, d_k]`
    - `K_rope: shape![B, h_kv, 1, d_k]`
- **Operations**
    - **RoPE table lookup**
        - `t_rope: shape![1, d_k_prime, 2, 2] = gather(index: position, table: w_rope)`
    - **Apply RoPE**
        - **Reshape (noop)**
            - `Q: shape![B, 1, h_q, d_k] == shape![B, 1, h_q, d_k_prime, f]`
            - `K_new: shape![B, 1, h_kv, d_k] == shape![B, 1, h_kv, d_k_prime, f]`
            - `t_rope: shape![1, d_k_prime, 2, 2] == shape![1, d_k_prime, f, 2]`
        - **einsum**
            - `Q_rope = einsum(Q, t_rope)`
                - `(shape![B, 1, h_q, d_k_prime, f], shape![1, d_k_prime, f, 2]) -> shape![B, h_q, 1, d_k_prime, 2] == shape![B, h_q, 1, d_k]`
            - `K_rope = einsum(K_new, t_rope)`
                - `(shape![B, 1, h_kv, d_k_prime, f], shape![1, d_k_prime, f, 2]) -> shape![B, h_kv, 1, d_k_prime, 2] == shape![B, h_kv, 1, d_k]`

##### 2.2.3. KV Cache Update

KV cache update appends the new Key and Value to the existing cache for future token generation.

- **Inputs**
    - `kv_cache_l_K: shape![B, h_kv, s-1, d_k]` (existing cache)
    - `kv_cache_l_V: shape![B, h_kv, s-1, d_k]` (existing cache)
    - `K_rope: shape![B, h_kv, 1, d_k]` (new Key)
    - `V_new: shape![B, 1, h_kv, d_k]` (new Value)
    > **TODO** (youseok.yang): `V_new` has shape `[B, 1, h_kv, d_k]` but the cache expects `[B, h_kv, s, d_k]`. Either correct `V_new`'s shape to `[B, h_kv, 1, d_k]` (consistent with `K_rope` and the cache) or add an explicit reshape/transpose step before the cache update.
- **Outputs**
    - `kv_cache_l_K: shape![B, h_kv, s, d_k]` (updated cache)
    - `kv_cache_l_V: shape![B, h_kv, s, d_k]` (updated cache)
- **Operations**
    - **Concatenate**: Add new K, V to existing cache
        - `kv_cache_l_K[s-1] = K_rope`
        - `kv_cache_l_V[s-1] = V_new`
        - Processing differs depending on concat axis allocation.
        Data movement between slices: use RoutingEngine/parallel copy; data movement between elements: use parallel copy.
        - Concat on HBM using DMA is also possible.

##### 2.2.4. Grouped Query Attention Computation

Attention computation uses the current Query against the entire KV cache to determine which past tokens are relevant.

**2.2.4.1. Attention Scores Computation**

Attention scores measure similarity between the current Query and all cached Keys.

- **Inputs**
    - `Q_rope: shape![B, h_q, 1, d_k]`
    - `kv_cache_l_K: shape![B, h_kv, s, d_k]`
- **Output**
    - `scores: shape![B, h_q, 1, s]`
- **Operations**
    - `scores = (Q_rope @ kv_cache_l_K.T) / sqrt(d_k)`
    - **Reshape (noop)**
        - `Q_rope: shape![B, h_q, 1, d_k] == shape![B, G, h_kv, 1, d_k]`
        - `kv_cache_l_K: shape![B, h_kv, s, d_k] == shape![B, h_kv, s, d_k]`
    - **einsum**
        - `scores_before_normalize = einsum(Q_rope, kv_cache_l_K)`
        - `(shape![B, G, h_kv, 1, d_k], shape![B, h_kv, s, d_k]) -> shape![B, G, h_kv, 1, s] == shape![B, h_q, 1, s]`
        - The einsum expression shows that `G` was broadcast from `kv_cache_l_K`, and `d_k` was reduced.
    - **Normalize**
        - `scores = scores_before_normalize / sqrt(d_k)`
        - Processed as constant multiplication in Vector Engine

**2.2.4.2. Softmax Application**

Softmax converts scores to attention weights.
Causal mask is unnecessary in decode because the current token only references past tokens.

- **Input**
    - `scores: shape![B, h_q, 1, s]`
- **Output**
    - `attn_weights: shape![B, h_q, 1, s]`
- **Operation**
    - `attn_weights = softmax(scores)`
    - Softmax is applied over the last axis (`s`, i.e., all past tokens)
    - `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`
    - Processed by Vector Engine

**2.2.4.3. Weighted Sum (Attention Output)**

Weighted sum combines cached Values according to attention weights to produce the attention output.

- **Inputs**
    - `attn_weights: shape![B, h_q, 1, s]`
    - `kv_cache_l_V: shape![B, h_kv, s, d_k]`
- **Output**
    - `attn_output: shape![B, h_q, 1, d_k]`
- **Operations**
    - **Reshape (noop)**
        - `attn_weights: shape![B, h_q, 1, s] == shape![B, G, h_kv, 1, s]`
        - `kv_cache_l_V: shape![B, h_kv, s, d_k] == shape![B, h_kv, s, d_k]`
    - **einsum**
        - `attn_output = einsum(attn_weights, kv_cache_l_V)`
        - `(shape![B, G, h_kv, 1, s], shape![B, h_kv, s, d_k]) -> shape![B, G, h_kv, 1, d_k] == shape![B, h_q, 1, d_k]`
        - The einsum expression shows that `G` was broadcast from `kv_cache_l_V`, and `s` was reduced.

##### 2.2.5. Output Projection

Output projection transforms the attention result back to the hidden dimension.

- **Input**
    - `attn_output: shape![B, h_q, 1, d_k]`
- **Weight**
    - `w_o: shape![h_q, d_k, D]`
- **Output**
    - `attn_out: shape![B, 1, D]`
- **Operations**
    - `attn_out = einsum(attn_output, w_o)`
    - `(shape![B, h_q, 1, d_k], shape![h_q, d_k, D]) -> shape![B, 1, D]`

##### 2.2.6. Residual Connection

Residual connection combines attention output with layer input.

- **Inputs**
    - `x_prev: shape![B, 1, D]` (layer input)
    - `attn_out: shape![B, 1, D]` (attention output)
- **Output**
    - `x_attn: shape![B, 1, D]`
- **Operation**
    - `x_attn = x_prev + attn_out`
    - elementwise addition: Processed by Vector Engine

#### 2.3. Feed-Forward Network (FFN)

FFN in decode phase is identical to prefill, but processes only a single token (sequence length = 1).

##### 2.3.1. Post-Attention Layer Normalization

Post-attention normalization prepares the token for FFN processing.

- **Input**
    - `x_attn: shape![B, 1, D]`
- **Output**
    - `x_ffn_norm: shape![B, 1, D]`
- **Operation**
    - `x_ffn_norm = RMSNorm(x_attn)`
    - Processed by Vector Engine

##### 2.3.2. SwiGLU FFN

SwiGLU applies the gated activation function with three projections.

- **Input**
    - `x_ffn_norm: shape![B, 1, D]`
- **Weights**
    - `w_gate: shape![D, F]`
    - `w_up: shape![D, F]`
    - `w_down: shape![F, D]`
- **Output**
    - `ffn_out: shape![B, 1, D]`
- **Operations**
    - **Gate projection**:
        - `gate = einsum(x_ffn_norm, w_gate)`
        - `(shape![B, 1, D], shape![D, F]) -> shape![B, 1, F]`
    - **Up projection**:
        - `up = einsum(x_ffn_norm, w_up)`
        - `(shape![B, 1, D], shape![D, F]) -> shape![B, 1, F]`
    - **SwiGLU activation**:
        - `activated = SiLU(gate) * up`
        - Processed by Vector Engine
    - **Down projection**:
        - `ffn_out = einsum(activated, w_down)`
        - `(shape![B, 1, F], shape![F, D]) -> shape![B, 1, D]`

##### 2.3.3. Residual Connection

FFN residual connection produces the final layer output.

- **Inputs**
    - `x_attn: shape![B, 1, D]`
    - `ffn_out: shape![B, 1, D]`
- **Output**
    - `x_l: shape![B, 1, D]`
- **Operation**
    - `x_l = x_attn + ffn_out`
    - elementwise addition: Processed by Vector Engine

### 3. Final Layer Normalization

Final layer normalization prepares the output for the language model head.

- **Input**
    - `x_L: shape![B, 1, D]`
- **Output**
    - `x_final: shape![B, 1, D]`
- **Operation**
    - `x_final = RMSNorm(x_L)`
    - Processed by Vector Engine

### 4. Language Model Head

The language model head projects the hidden state to vocabulary logits.
Unlike prefill, no slice operation is needed since there is only a single token.

- **Input**
    - `x_final: shape![B, 1, D]`
- **Weight**
    - `w_lm_head: shape![D, V]`
- **Output**
    - `logits: shape![B, V]`
- **Operations**
    - **Reshape/Squeeze**: Remove sequence dimension
        - `x_squeezed: shape![B, D] = squeeze(x_final)`
    - **einsum**: Logit computation for vocabulary
        - `logits = einsum(x_squeezed, w_lm_head)`
        - `(shape![B, D], shape![D, V]) -> shape![B, V]`

### 5. Sampling

Sampling is identical to [Prefill Sampling](#5-sampling): temperature scaling, softmax, and token selection, performed on the Host.

### 6. Termination Conditions

Generation terminates when any of three conditions is met:

- **EOS token generated**: Sampled token is the End-of-Sequence token
- **Maximum length reached**: `s >= s_max`
- **User-defined termination conditions**: When specific patterns or conditions are met

If generation continues, update `s <- s + 1` and return to the next decoding step.

## Prefill vs Decode Phase Comparison

The following table summarizes the key differences between prefill and decode phases:

| Characteristic | Prefill Phase | Decode Phase |
|------|---------------|--------------|
| Input sequence length | `s_in` (variable) | 1 (fixed) |
| Parallel processing | `s_in` tokens processed in parallel | Only 1 token processed |
| KV Cache | Create and store | Read and update |
| Attention computation | Causal mask required | Causal mask not required |
| Attention shape | `shape![B, h_q, s_in, s_in]` | `shape![B, h_q, 1, s]` |
| Computation characteristics | Compute-bound (large-scale computation) | Memory-bound (KV cache access) |
| Throughput | High (parallel processing) | Low (sequential processing) |
| Latency | Relatively high | Low (per token) |
