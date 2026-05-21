# Scheduler


vISA translates programs into hardware schedules using two programmer-visible inputs: the textual order of operations and explicit memory address assignments.
The scheduler respects the written order as the authoritative sequence: it does not reorder operations.


## Operation Order

The execution order follows the written order exactly, as the example below illustrates, where `load_from_host()` loads a tensor from host memory and `.op()` represents any pipeline operation:

```rust,ignore
let t0 = load_from_host();  // O0
let t1 = load_from_host();  // O1
let t2 = t0.op();           // O2
let t3 = t1.op();           // O3
let t4 = t2.op();           // O4
let t5 = t4.op();           // O5
```

The final execution order respects the written order: `O0 → O1 → O2 → O3 → O4 → O5`.


## Memory Allocation

Each tensor requires a specific memory address for precise scheduling.
Currently, tensor addresses must be specified explicitly by the programmer.



## Hardware Resources

The hardware exposes three execution contexts — **main**, **sub**, and **DMA** — that can run in parallel.
See [Execution Contexts](./computing-tensors/index.md#scheduling-execution-contexts) for which engines each context can drive.
The scheduler treats each context independently: operations on different contexts run in parallel, operations on the same context serialize.

Two factors cause operations to serialize: resource conflicts and memory dependencies.

Resource conflicts occur when two operations require the same context, forcing the later one to wait.
For example, two matrix multiplications both requiring the main context must execute sequentially.
However, a matrix multiplication (main context) can run in parallel with a DMA transfer (DMA context) because they use different contexts.

Memory dependencies arise from data hazards on shared addresses:

- **RAW** (read-after-write): a read must see the result of a preceding write.
- **WAR** (write-after-read): a write must not overwrite data still being read.
- **WAW** (write-after-write): writes to the same address must execute in order.

The scheduler detects these hazards by analyzing the memory addresses specified in the program.

The scheduler manages these constraints automatically.
It analyzes each operation's resource usage and memory addresses to determine parallelism opportunities while respecting program order, inserting implicit waits where necessary.
This dependency resolution frees programmers from manually inserting synchronization barriers, though memory addresses must still be specified explicitly (see [Memory Allocation](#memory-allocation)).

## Double-Buffering Pattern

Double-buffering splits the TRF into two halves so the sub context fills one half while the main context reads the other, and the kernel alternates which half each context targets across iterations.
This works because the TRF storage splits each bank into a `FirstHalf` and a `SecondHalf` (see [Register Files](./computing-tensors/register-files.md#double-buffering)), letting main and sub target different halves without contention.

The VRF does not enforce a halved split: each slice's 8 KB of VRF can be freely partitioned among multiple tensors, and double-buffering, when desired, is arranged by the kernel writer allocating disjoint regions rather than by hardware-enforced halves.

The kernel pattern is two passes per iteration, swapping `FirstHalf` and `SecondHalf` between them:

```rust,ignore
// Prime the first half before the loop.
let mut trf = ctx.sub
    .begin(weights[0].view())
    .fetch::<...>()
    .collect::<...>()
    .to_trf(TrfAddress::FirstHalf);

for i in 0..N {
    // While main reads the current half, sub preloads the next batch into the other half.
    let other_half = if i % 2 == 0 { TrfAddress::SecondHalf } else { TrfAddress::FirstHalf };
    let next_trf = (i + 1 < N).then(|| {
        ctx.sub
            .begin(weights[i + 1].view())
            .fetch::<...>()
            .collect::<...>()
            .to_trf(other_half)
    });

    ctx.main.begin(input[i].view()).contract_outer::<...>(&trf)...;

    if let Some(t) = next_trf {
        trf = t;
    }
}
```

The scheduler overlaps sub and main automatically because they hit different TRF halves (no WAR hazard) and different hardware resources (no resource conflict).



