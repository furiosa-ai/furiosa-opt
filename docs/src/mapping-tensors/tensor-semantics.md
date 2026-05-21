# Tensor Semantics

Tensors reside in HBM, on-chip DM, or the pipeline stream, and operations transform them.
This chapter defines their mathematical meaning: what it means for a tensor variable to *hold* a mathematical tensor, and what it means for an operation to *specify* a mathematical function.
These definitions enable tensor-level reasoning about vISA programs: a function is correct when its output holds the right mathematical tensor, regardless of which mapping or memory tier is used.

## Holding a Tensor

A tensor variable *holds* mathematical tensor \\(T\\) when each element stores the value of \\(T\\) at the tensor index formed by summing the partial indices produced by each dimension's mapping.

`HostTensor<D, E>` is the simplest case: a single mapping `E` fully determines the correspondence between buffer positions and tensor indices.
`HostTensor<bf16, m![A, B]>` with `A = 8` and `B = 512`, for instance, stores 4,096 `bf16` elements in A-major, B-minor order.
It holds tensor \\(T\\) when:
- for every buffer index `i` where `E::map(i) = Some(ti)`,
- the `i`-th element stores the value of \\(T\\) at `ti`.

`HbmTensor<D, Chip, Element>` extends this by splitting the single mapping into two: `Chip` maps chip indices to partial tensor indices, and `Element` maps per-chip element indices to the remaining partial indices, with each covering a disjoint subset of axes so their sum recovers the full tensor index.
It holds \\(T\\) when:
- for every chip index `i` and element index `j` where `Chip::map(i) = Some(ti)` and `Element::map(j) = Some(tj)`,
- the `i`-th chip's `j`-th element stores \\(T\\) at the index `ti + tj`.

All other tensor types apply the same rule to more dimensions: each element stores \\(T\\) at the sum of the partial indices returned by all its mapping parameters.

## Specifying a Function

Specifying a function means declaring what its output holds in terms of its inputs.
For example, the function `elementwise_add` specifies the mathematical operation \\(f(T_1, T_2) = T_1 + T_2\\) in that:
- For every tensor \\(T_1\\) and \\(T_2\\),
- if `lhs` holds \\(T_1\\) and `rhs` holds \\(T_2\\),
- then the return value holds \\(T_1 + T_2\\).

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];

fn elementwise_add(
    lhs: &HbmTensor<bf16, m![A], m![B]>,
    rhs: &HbmTensor<bf16, m![A], m![B]>,
) -> HbmTensor<bf16, m![A], m![B]> {
    // ... computes elementwise add ...
    # todo!("elementwise add lhs and rhs")
}
```

<a id="mathematical-tensor-move"></a>
A *mathematical tensor move* specifies \\(f(T) = T\\): the output holds the same mathematical tensor as the input, regardless of representation.
`.to_dm()` is a mathematical tensor move.
The `.to_dm()` method, for instance, specifies \\(f(T) = T\\) in that:
- if `hbm` holds \\(T\\),
- the return value holds \\(T\\).

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];

fn hbm_to_dm(
    ctx: &mut Context,
    hbm: &HbmTensor<bf16, m![A], m![B]>,
) -> DmTensor<bf16, m![A], m![1], m![B / 2], m![B % 2]> {
    hbm.to_dm(&mut ctx.tdma, 1 << 16) // 64KB offset
}
```
