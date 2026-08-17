# Tensor and Contraction

A tensor maps each named index in its shape to a value.
A shape is an unordered set of named axes, so `{ N = 4, C = 3 }` and `{ C = 3, N = 4 }` identify the same tensor.
An ordered representation of that shape behaves like a familiar multidimensional array.
A tensor index supplies one value for every axis, so `{ N = 4, C = 3 }` includes indices such as `{ N: 0, C: 0 }` and `{ N: 0, C: 1 }`.

| Tensor | Dimension | Example | Named shape |
|--------|-----------|---------|-------------|
| Scalar | 0D | `5.2` | `{}` |
| Vector | 1D | `[1, 2, 3]` | `{ I = 3 }` |
| Matrix | 2D | a `2 × 4` grid | `{ I = 2, J = 4 }` |
| Image batch | 4D | four RGB images | `{ N = 4, C = 3, H = 256, W = 512 }` |

Tensor contraction multiplies two inputs elementwise and reduces every shared axis that is absent from the output.
Every contraction consists of broadcast, multiply, and reduce steps.

| Operation | Einsum | Broadcast | Multiply | Reduce |
|-----------|--------|-----------|----------|--------|
| Dot product | \(I, I \rightarrow 1\) | None. | \(x_i y_i\) | \(\sum_i x_i y_i\) |
| GEMV | \(IJ, J \rightarrow I\) | \(x\) across \(I\). | \(A_{ij} x_j\) | \(y_i = \sum_j A_{ij} x_j\) |
| GEMM | \(IK, KJ \rightarrow IJ\) | \(A\) across \(J\) and \(B\) across \(I\). | \(A_{ik} B_{kj}\) | \(C_{ij} = \sum_k A_{ik} B_{kj}\) |

This page owns the introductory math only.
Mapping, movement, and engine contracts remain in their respective reference chapters.
