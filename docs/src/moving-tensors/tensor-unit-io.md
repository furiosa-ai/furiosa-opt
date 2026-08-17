# Case Study: Tensor Unit I/O

This end-to-end Tensor Unit I/O pattern uses Fetch and Commit to stage a tensor through DM while preserving every logical index.
It is a composed movement pattern, distinct from the individual Fetch Engine and Commit Engine reference pages.

The canonical public case is `fetch_commit_simple`.
It accepts an `i8` HBM tensor with mapping `[A, B]`, widens values to `i32` through the Tensor Unit, and returns an HBM tensor with mapping `[B, A]`.
The final `to_hbm` DMA call selects the output HBM mapping, so the test proves the complete HBM-to-DM, Fetch, Collect, Commit, and DM-to-HBM pipeline.
It intentionally does not claim a streaming permutation: the public example's `[A, B]` to `[B, A]` relayout occurs at the final DMA boundary.
The device source is included below.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/fetch_commit.rs}}
```

The host oracle and test are included in the examples test target.

```rust,ignore
{{#include ../../../furiosa-opt-examples/tests/fetch_commit_tests.rs}}
```

Run `cargo furiosa-opt test --test fetch_commit_tests` to execute the canonical case.
The host oracle checks every `output[b, a]` value against the widened `input[a, b]` value.
The streaming path is input DM, Fetch, Cast, Collect, Commit, and output DM with the original `[A, B]` DM layout.
The [Fetch Engine](./fetch-engine.md), [Collect Engine](../computing-tensors/collect-engine.md), and [Commit Engine](./commit-engine.md) document the APIs used here.

