# Tensor Functions
The preceding pages showed that the same tensor can live in different memory tiers with different mapping expressions.
To reason about operations independently of physical layout, TCP models hardware operations as abstract functions on mathematical tensors, focusing on what data they produce rather than how it is physically arranged.

The function `elementwise_add` implements the mathematical operation \\(f(T_1, T_2) = T_1 + T_2\\):
- For every tensor \\(T_1\\) and \\(T_2\\),
- if `lhs` holds \\(T_1\\) and `rhs` holds \\(T_2\\),
- then the return value holds \\(T_1 + T_2\\).

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, B = 512];

fn elementwise_add(
    lhs: &HbmTensor<bf16, m![A], m![B]>,
    rhs: &HbmTensor<bf16, m![A], m![B]>,
) -> HbmTensor<bf16, m![A], m![B]> {
    // ... computes elementwise add ...
    # todo!("elementwise add lhs and rhs")
}
```

The same reasoning applies to data movement: moving a tensor from one memory tier to another is also a tensor function, one that preserves the mathematical content while changing the physical representation.
The `.to_dm()` method implements the identity function on the mathematical tensor (not the physical representation), copying tensor \\(T\\) from HBM to on-chip Data Memory:

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, B = 512];

fn hbm_to_dm(
    ctx: &mut Context,
    hbm: &HbmTensor<bf16, m![A], m![B]>,
) -> DmTensor<bf16, m![A], m![1], m![B / 2], m![B % 2]> {
    hbm.to_dm(&mut ctx.tdma, 1 << 16) // 64KB offset
}
```

Both the input `HbmTensor` and output `DmTensor` hold the same mathematical tensor \\(T\\), but in different memory tiers and with different mapping expressions.
This means correctness is defined at the tensor level: a function is correct if its output holds the right mathematical tensor, regardless of which mapping or memory tier is used.
Treating data movement as a function on tensors rather than as a copy of bytes makes it composable with compute operations in the same pipeline.

