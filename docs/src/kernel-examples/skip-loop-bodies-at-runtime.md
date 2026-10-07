# Case Study: Skip Loop Bodies at Runtime

A kernel can use a runtime scalar to execute only the useful part of a statically bounded workload.
The host stores the value in `HbmScalar<T>`, the kernel loads it into SPM with `to_spm`, and a normal `if` guards the operations that may be skipped.
The tensor shapes and the loop's maximum trip count remain compile-time constants.

This pattern fits MoE expert blocks, variable-length batches, and other kernels with a known maximum amount of work and a runtime valid count.

## Guard a Static Loop

The following tested kernel copies the first `active_experts` groups and leaves the remaining output groups zero.

```rust
# #![feature(adt_const_params, proc_macro_hygiene, register_tool)]
# #![register_tool(furiosa_opt)]
# extern crate furiosa_opt_std;
# extern crate tokio;
# use furiosa_opt_std::prelude::*;
axes![Slice = 64, Group = 4, Width = 8];
type Chip = m![1];
type Cluster = m![1];
{{#include ../../../furiosa-opt-examples/src/moe.rs:runtime_work_skipping}}
# #[tokio::main]
# async fn main() -> Result<(), Error> {
#     let mut device = Device::new(bounded_dynamic_expert_loop.topology())?;
#     let input = HostTensor::<i32, m![Slice, Group, Width]>::from_vec(
#         (0..<m![Slice, Group, Width]>::SIZE).map(|value| value as i32),
#     ).to_hbm(&mut device.pdma).await?;
#     let active = HbmScalar::from_host(2usize, &mut device.pdma).await?;
#     let output = launch(bounded_dynamic_expert_loop, (&mut device, &input, &active)).await?;
#     let output = output.to_host::<m![Slice, Group, Width]>(&mut device.pdma).await?;
#     assert_eq!(output.into_vec().len(), <m![Slice, Group, Width]>::SIZE);
#     Ok(())
# }
```

The loop always has the static bound `Group::SIZE`.
Each iteration compares its `usize` induction variable with the `usize` value loaded from HBM.
If `active_experts` is zero, every body is skipped; if it is greater than `Group::SIZE`, every body runs.
Initialize any output region that skipped iterations must leave with a defined value, as the example does before the loop.

The same guard can surround Tensor Unit computation.
This excerpt from the tested MoE example runs each expert contraction only when its index is below the runtime count:

```rust
# #![feature(adt_const_params, proc_macro_hygiene, register_tool)]
# #![register_tool(furiosa_opt)]
# extern crate furiosa_opt_std;
# extern crate tokio;
# use furiosa_opt_std::prelude::*;
axes![Slice = 64, Expert = 4, MatmulOut = 8, MatmulRed = 16];
type Chip = m![1];
type Cluster = m![1];
{{#include ../../../furiosa-opt-examples/src/moe.rs:runtime_matmul_guard}}
# #[tokio::main]
# async fn main() -> Result<(), Error> {
#     let mut device = Device::new(bounded_dynamic_matmul.topology())?;
#     let activation = HostTensor::<bf16, m![Slice, MatmulRed]>::from_vec(
#         vec![bf16::from_f32(1.0); <m![Slice, MatmulRed]>::SIZE],
#     ).to_hbm(&mut device.pdma).await?;
#     let weight = HostTensor::<bf16, m![Slice, Expert, MatmulOut, MatmulRed]>::from_vec(
#         vec![bf16::from_f32(1.0); <m![Slice, Expert, MatmulOut, MatmulRed]>::SIZE],
#     ).to_hbm(&mut device.pdma).await?;
#     let active = HbmScalar::from_host(2usize, &mut device.pdma).await?;
#     let output = launch(bounded_dynamic_matmul, (&mut device, &activation, &weight, &active)).await?;
#     let output = output.to_host::<m![Slice, Expert, MatmulOut]>(&mut device.pdma).await?;
#     assert_eq!(output.into_vec().len(), <m![Slice, Expert, MatmulOut]>::SIZE);
#     Ok(())
# }
```

## Pass and Update a Runtime Scalar

Create the scalar through the launch context's PDMA queue, then pass it like another kernel argument:

```rust
# #![feature(adt_const_params, proc_macro_hygiene, register_tool)]
# #![register_tool(furiosa_opt)]
# extern crate furiosa_opt_std;
# extern crate tokio;
# use furiosa_opt_std::prelude::*;
# #[device]
# fn scalar_consumer(device: &mut Device, active: &HbmScalar<usize>) {
#     let _active = active.to_spm(&mut device.tdma);
# }
# #[tokio::main]
# async fn main() -> Result<(), Error> {
# let mut device = Device::new(scalar_consumer.topology())?;
let mut active = HbmScalar::<usize>::from_host(0, &mut device.pdma).await?;

active.write(3, &mut device.pdma).await?;
launch(scalar_consumer, (&mut device, &active)).await?;
# Ok(())
# }
```

`from_host` allocates one scalar value in every chip's local HBM and writes the same value to each copy.
`write` reuses that allocation for a later launch.
Inside the kernel, `to_spm(&mut device.tdma)` stages the value into an SPM scalar visible to device scalar expressions.

`HbmScalar` supports `bool`, `i32`, `u32`, `i64`, `u64`, and `usize`.
Use `bool` for a direct on/off guard and `usize` for loop bounds and tensor indices.
The current compiler accepts `usize` runtime scalars only for a 64-bit Rust frontend target and stores them as unsigned 64-bit values.

## Use `if`, Not a Runtime Loop Bound

Runtime `if` statements are supported, including statement blocks that return no value.
General `break` statements and runtime loop bounds are not supported in device functions.
Write a static maximum loop with a runtime guard:

```rust
# struct Expert;
# impl Expert { const SIZE: usize = 4; }
# fn run_expert(_: usize) {}
# let active_experts = 2;
for expert in 0..Expert::SIZE {
    if expert < active_experts {
        run_expert(expert);
    }
}
```

The following forms are not supported:

```text
for expert in 0..active_experts {
    run_expert(expert);
}

for expert in 0..Expert::SIZE {
    if expert >= active_experts {
        break;
    }
    run_expert(expert);
}
```

The supported form schedules the static maximum number of iterations.
The guarded Tensor Unit or DMA operations do not run when the condition is false, but evaluating the guard and advancing the outer loop still have a runtime cost.

## Unroll a Small Maximum

`#[unroll]` may be applied when the maximum trip count is a small compile-time constant:

```rust
# #![feature(adt_const_params, proc_macro_hygiene, register_tool, stmt_expr_attributes)]
# #![register_tool(furiosa_opt)]
# extern crate furiosa_opt_std;
# extern crate tokio;
# use furiosa_opt_std::prelude::*;
axes![Slice = 64, Group = 4, Width = 8];
type Chip = m![1];
type Cluster = m![1];
{{#include ../../../furiosa-opt-examples/src/moe.rs:unrolled_runtime_work_skipping}}
# #[tokio::main]
# async fn main() -> Result<(), Error> {
#     let mut device = Device::new(unrolled_bounded_dynamic_expert_loop.topology())?;
#     let input = HostTensor::<i32, m![Slice, Group, Width]>::from_vec(
#         (0..<m![Slice, Group, Width]>::SIZE).map(|value| value as i32),
#     ).to_hbm(&mut device.pdma).await?;
#     let active = HbmScalar::from_host(2usize, &mut device.pdma).await?;
#     let output = launch(unrolled_bounded_dynamic_expert_loop, (&mut device, &input, &active)).await?;
#     let output = output.to_host::<m![Slice, Group, Width]>(&mut device.pdma).await?;
#     assert_eq!(output.into_vec().len(), <m![Slice, Group, Width]>::SIZE);
#     Ok(())
# }
```

With `Group::SIZE = 4`, `#[unroll]` replaces the loop with four copies of its guarded body. The
result behaves like this:

```text
if 0 < active_experts { run_group(0); }
if 1 < active_experts { run_group(1); }
if 2 < active_experts { run_group(2); }
if 3 < active_experts { run_group(3); }
```

`active_experts` is still read at runtime. Unrolling removes the loop control and lets the scheduler
consider operations from different groups together; it does not remove the four runtime checks.

The range must start at zero and end at a nonzero compile-time constant. Add `#[unroll]` to each
level that should be expanded in a nested loop.

There is no compiler threshold for a “small” loop. Generated code grows with the static bound, so a
four-group loop is a reasonable candidate while a 1,024-expert loop should normally remain rolled.
Compare both forms when the trade-off is unclear.
See [Unroll a loop](../scheduling/tuning.md#unroll-a-loop) for the general scheduling trade-offs.
