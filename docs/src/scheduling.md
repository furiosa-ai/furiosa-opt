# Scheduling

<!-- > **TODO** (youseok.yang): Add practical examples for kernel writers: -->
<!-- > - Show Main + DMA parallel execution example (the win from overlapping) -->
<!-- > - Add "double-buffering" walkthrough: Sub prefetches while Main computes -->
<!-- > - Complete resource occupation table: which operations block which resources -->
<!-- > - Current content is conceptual; needs concrete before/after schedule comparisons -->

Scheduling determines how operations execute on hardware resources.
This chapter explains how the Virtual ISA translates programs into executable schedules. Two programmer-visible inputs determine the schedule: the textual order of operations and explicit memory address assignments.

Programs implicitly define their schedule through the textual order of operations and explicit memory address assignments. The scheduler respects the written order as the authoritative sequence — it does not reorder operations.
The scheduler then analyzes resource dependencies to determine which operations can run in parallel.


## Operation Order

Operation order is how the program communicates sequencing intent to the scheduler: the textual order of operations defines their execution order.
The following example shows this, where `load_from_host()` loads a tensor from host memory and `.op()` represents any pipeline operation:

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

Each tensor requires a specific memory address for precise scheduling. Currently, tensor addresses must be specified explicitly by the programmer.

<!-- > **TODO**: In practice, manually specifying all addresses is not feasible. A scheduler can handle address assignment and detailed scheduling from an intermediate representation where addresses are not yet fixed. -->


## Hardware Resources

The hardware provides three allocatable resources that can execute in parallel:

| Resource | Description |
|----------|-------------|
| Main context | Primary Tensor Unit execution context |
| Sub context | Secondary context for data prefetching |
| Direct Memory Access (DMA) Engine | Memory-to-memory data transfer |

**Main context** handles compute-intensive operations through the complete Tensor Unit pipeline — Fetch, Switching, Collect, Contraction, Vector, Cast, Transpose, and Commit — but can only execute one operation at a time.

**Sub context** runs data movement operations (SRAM-to-TRF transfers, SRAM-to-SRAM copies) concurrently with the main context, enabling double-buffering where the next operation's data is prepared while the current one computes.

**DMA Engine** moves large tensors between HBM and SRAM independently of both Tensor Unit contexts, enabling overlapped data transfer and computation.

Two factors cause operations to serialize: resource conflicts and memory dependencies.

Resource conflicts occur when two operations require the same resource, forcing the later one to wait. For example, two matrix multiplications both requiring the main context must execute sequentially. However, a matrix multiplication (main context) can run in parallel with a DMA transfer (DMA engine) because they use different resources.

Memory dependencies arise from data hazards on shared addresses. Read-after-write (RAW) hazards require a read to see the result of a preceding write. Write-after-read (WAR) hazards prevent a write from overwriting data still being read. Write-after-write (WAW) hazards require writes to the same address to execute in order. The scheduler detects these hazards by analyzing the memory addresses specified in the program.

The scheduler manages these constraints automatically. It analyzes each operation's resource usage and memory addresses to determine parallelism opportunities while respecting program order, inserting implicit waits where necessary. This dependency resolution frees programmers from manually inserting synchronization barriers, though memory addresses must still be specified explicitly (see [Memory Allocation](#memory-allocation)).

<!-- > **TODO**: Document which resources each operation type occupies. -->

<!-- > **TODO**: Provide tooling for users to visualize the final schedule (LIR plot) from their program. -->
