# Tensor Semantics

Tensors reside in HBM, on-chip DM, or the pipeline stream, and operations transform them.
This chapter defines their mathematical meaning: what it means for a tensor variable to *hold* a mathematical tensor, and what it means for an operation to *specify* a mathematical function.
These definitions enable tensor-level reasoning about vISA programs: a function is correct when its output holds the right mathematical tensor, regardless of which mapping or memory tier is used.

## Tensor Holding Semantics

A tensor variable *holds* mathematical tensor \\(T\\) when each element stores the value of \\(T\\) at the tensor index formed by summing the partial indices produced by each dimension's mapping.

`HostTensor<D, E>` is the simplest case: a single mapping `E` fully determines the correspondence between buffer positions and tensor indices.
`HostTensor<bf16, m![A, B]>` with `A = 8` and `B = 512`, for instance, stores 4,096 `bf16` elements in A-major, B-minor order.
It holds tensor \\(T\\) when:
- for every buffer index `i` where `E::map(i) = Cell::Index(ti)`,
- the `i`-th element stores the value of \\(T\\) at `ti`.

`HbmTensor<D, Chip, Element>` extends this by splitting the single mapping into two: `Chip` maps chip indices to partial tensor indices, and `Element` maps per-chip element indices to the remaining partial indices, with each covering a disjoint subset of axes so their sum recovers the full tensor index.
It holds \\(T\\) when:
- for every chip index `i` and element index `j` where `Chip::map(i) = Cell::Index(ti)` and `Element::map(j) = Cell::Index(tj)`,
- the `i`-th chip's `j`-th element stores \\(T\\) at the index `ti + tj`.

All other tensor types apply the same rule to more dimensions: each element stores \\(T\\) at the sum of the partial indices returned by all its mapping parameters.

## Linear Combination Semantics

Linear combination expressions `$(e1:n1, ..., ed:nd)` combine multiple dimensions with specified strides.
Their size is `size_S($(e1:n1, ..., ed:nd)) = 1 + sum_k((size_S(ek) - 1) * nk)`.
The mapping `S, $(e1:n1, ..., ed:nd) |- si ~ ti` is valid if there exist `si1...sid, ti1...tid` such that for every `k`, `S, ek |- sik ~ tik`, `si = sum_k(sik * nk)`, and `ti = sum_k(tik * nk)`.

Linear combinations can encode outer sum: `e1 * e2` is equivalent to `$(e1 : size_S(e2), e2 : 1)`.
Outer sum is preferred when axis reordering matters because changing `e1 * e2` to `e2 * e1` does not require manual stride updates.

Sliding operations access overlapping data blocks.
Consider a buffer of 9 elements representing a tensor with shape \\(\\{N=5, F=3\\}\\), where each row is a 3-element slice that slides one element at a time.
The tensor element at \\(N, F\\) maps to buffer index \\(N + 2F\\):

$$
\\begin{array}{c|ccc}
  & F=0 & F=1 & F=2 \\\\
\\hline
N=0 & 0 & 2 & 4 \\\\
N=1 & 1 & 3 & 5 \\\\
N=2 & 2 & 4 & 6 \\\\
N=3 & 3 & 5 & 7 \\\\
N=4 & 4 & 6 & 8 \\\\
\\end{array}
$$

In this sliding pattern, a single space index can map to multiple tensor indices.
For example, space index `4` maps to `{4_N}`, `{2_N, 1_F}`, and `{2_F}` simultaneously, illustrating the non-one-to-one nature of `(S, e).maps(si, ti)`.
The linear combination uses stride `1` for `N` and stride `2` for `F`, yielding `1 + (5-1)*1 + (3-1)*2 = 9`.

## Function Specification

Specifying a function means declaring what its output holds in terms of its inputs.
For example, the function `elementwise_add` specifies the mathematical operation \\(f(T_1, T_2) = T_1 + T_2\\) in that:
- For every tensor \\(T_1\\) and \\(T_2\\),
- if `lhs` holds \\(T_1\\) and `rhs` holds \\(T_2\\),
- then the return value holds \\(T_1 + T_2\\).

```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];

// This signature specifies the mathematical contract; the engine pipeline
// implementation is shown in the Computing Tensors chapter.
fn elementwise_add(
    lhs: &HbmTensor<bf16, m![A], m![B]>,
    rhs: &HbmTensor<bf16, m![A], m![B]>,
) -> HbmTensor<bf16, m![A], m![B]> {
    // The implementation is intentionally omitted; this example states only
    // the function's tensor-level contract.
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
    hbm.to_dm(&mut ctx.tdma)
}
```
