# Quick Start

This index bootstraps a runnable base-template vector kernel.
The five runnable patterns in [Kernel Design](./quick-start/kernel-design.md) add one hardware decision at a time.
That single chapter owns the source-backed device and host-oracle walkthroughs for constant addition, elementwise multiplication, dot product, GEMV, and GEMM.

Validation meanings and caveats are defined only in [Kernel Validation](./quick-start/kernel-validation.md).

HBM is high-bandwidth memory used for host transfers, and DM is on-chip data memory used by the Tensor Unit.

After these basic patterns, continue through Mapping, Moving, Computing, End-to-End Cases, Scheduling, and Tools.
The subsystem chapters now own the matching advanced examples, while End-to-End Cases collects composed model tutorials.

## Read the ordered child pages

Read [Setup and Tooling](./quick-start/setup-and-tooling.md) for installation and the first project.
Read [Tensor and Contraction](./quick-start/tensor-and-contraction.md) for tensor shapes and contraction math.
Read [Kernel Design](./quick-start/kernel-design.md) for the five runnable pattern sections.
Read [Kernel Validation](./quick-start/kernel-validation.md) for compile, CPU, NPU, and schedule evidence.

## Continue to Reference Chapters

The runnable patterns process tensors that fit in one hardware pass and within the 512 KB per-slice DM capacity.
Use temporal partitioning when tiles run sequentially over time and spatial partitioning when tiles distribute across parallel hardware units.
Read [Mapping Tensors](./mapping-tensors/index.md), then [Moving Tensors](./moving-tensors/index.md), then [Computing Tensors](./computing-tensors/index.md).
After those reference chapters, choose a matching advanced pattern in the relevant Mapping, Moving, or Computing chapter.

## Choose the next pattern

The three reference chapters establish the mapping, movement, and compute contracts before you choose an example.
Use the simplest example route that matches the kernel's next constraint.

- **Fits one pass:** No escalation; the preceding basic patterns apply.
- **Layout permutation or packet handling:** Read [Case Study: Tensor Unit I/O](./moving-tensors/tensor-unit-io.md) under Moving Tensors.
- **Composed model patterns:** Read the [Case Study: Transformer](./kernel-examples/transformer.md) under End-to-End Cases.
