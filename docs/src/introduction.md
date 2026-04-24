# Introduction


FuriosaAI's Tensor Contraction Processor (TCP) is a massively parallel AI accelerator targeting inference workloads.
High-level frameworks such as PyTorch and XLA abstract away memory layouts and hardware scheduling, but give programmers no control over either.
Low-level kernel APIs give fine-grained control, but require reasoning in bytes and hardware addresses rather than tensors.
TCP's **Virtual Instruction Set Architecture (Virtual ISA)** bridges this gap: it lets programmers think in terms of tensors while directly managing memory allocation and tensor unit scheduling.
This manual explains TCP programming through the Virtual ISA.

The manual walks through concrete examples, targeting two audiences: programmers writing Virtual ISA directly and compiler developers generating it.
Basic Rust familiarity is assumed; see [the language manual](https://doc.rust-lang.org/book/) if needed.
> [!WARNING]
> **Alpha Test Build: Experimental Software**
>
> This software is an early, experimental, and incomplete build intended strictly for technical evaluation and internal testing.
>
> Before using this software for any production work, critical tasks, or for important data, you must consult with Furiosa engineers.
>
> Your feedback is vital to our development. Please provide it.

## Installation

Install two dependencies:

- **Rust**: Follow the [official guide](https://doc.rust-lang.org/book/ch01-01-installation.html#installation).
- **Furiosa SDK**: Follow the [SDK documentation](https://developer.furiosa.ai/latest/en/).

## Your First Program

Create a new project:

```bash
cargo new --bin tcp-my-project
cd tcp-my-project
cargo add furiosa-visa-std tokio
```

Add `rust-toolchain.toml`:
```toml
{{#include ../../rust-toolchain.toml}}
```

Write `main.rs`:

```rust
# #![feature(register_tool)]
# #![register_tool(tcp)]
# extern crate furiosa_visa_std;
# extern crate tokio;
# extern crate rand;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use furiosa_visa_std::prelude::*;  // provided by the Furiosa SDK

// Declare axis sizes
axes![A = 8, B = 512];

/// The main function running in host
#[tokio::main]
async fn main() {
    // Acquire exclusive access to the TCP device
    let mut ctx = Context::acquire();

    // TCP has three memory levels:
    // - Host: system memory
    // - HBM (High-Bandwidth Memory): device's main memory
    // - SRAM (on-chip scratchpad): the primary SRAM tier is called DM (Data Memory)
    //
    // Data flows: Host → HBM → DM → compute → DM → HBM → Host.
    //
    // Two DMA engines move data between these levels:
    // - `ctx.pdma` (PCIe DMA): transfers between Host and HBM
    // - `ctx.tdma` (Tensor DMA): transfers between HBM and DM

    // Create tensor on host
    // Tensors are parameterized by element type and mapping
    // The mapping `m![A, B]` specifies `A` as the major axis and `B` as the minor axis
    let mut rng = SmallRng::seed_from_u64(42);
    let host: HostTensor<i8, m![A, B]> = HostTensor::rand(&mut rng);

    // Transfer to device HBM using PCIe DMA engine
    // HBM tensor has two dimensions: m![A] for chip and m![B] for intra-chip address
    let hbm: HbmTensor<i8, m![A], m![B]> = host.to_hbm(&mut ctx.pdma, 0x1000).await;

    // Launch kernel on device
    // Host continues while kernel runs asynchronously, but the kernel synchronously occupies the device
    launch(kernel, (&mut ctx, &hbm))
    // Host waits for the asynchronous execution of the kernel to finish
        .await;
}

#[device(chip = 1)] // Running on a single chip
fn kernel(ctx: &mut Context, hbm: &HbmTensor<i8, m![A], m![B]>) {
    // Move to DM (Data Memory) in on-chip SRAM using Tensor DMA engine
    let dm = hbm.to_dm::<m![1], m![A], m![B]>(&mut ctx.tdma, 0);

    // ... perform computations ...
}
```

## Build and Test

TCP supports two execution environments, ordered from fastest iteration to production use:

```bash
# 1. CPUs (standalone Rust)
cargo build  # Add --release for optimized builds, same below
cargo test

# 2. Real TCP devices
cargo furiosa-opt build
cargo furiosa-opt test
```

## Development Tools

<!-- > **TODO**: This page needs a comprehensive rewrite to serve as a practical guide for the TCP Software Toolchain. -->
<!-- > -->
<!-- > **What to Cover:** -->
<!-- > - Introduction to the TCP Software Toolchain (cargo tcp) and its purpose -->
<!-- > - Each of the four main tools: Compiler, Interpreter, Language Server, and Schedule Viewer -->
<!-- > - Command-line interface and available options for each tool -->
<!-- > - Common workflows and usage patterns -->
<!-- > -->
<!-- > **Level of Detail:** -->
<!-- > - Provide command-line examples at the abstraction level of `cargo tcp --help` -->
<!-- > - Show real-world usage scenarios for each tool -->
<!-- > - Include both basic and advanced usage examples -->
<!-- > - Explain when and why to use each tool -->
<!-- > -->
<!-- > **Structure:** -->
<!-- > 1. Brief overview of the toolchain (already exists) -->
<!-- > 2. Individual sections for each tool with: -->
<!-- >    - Purpose and use cases -->
<!-- >    - Command syntax and key options -->
<!-- >    - Practical examples -->
<!-- >    - Integration with other tools in the workflow -->
<!-- > 3. Common workflows section showing how tools work together -->
<!-- > 4. Troubleshooting tips (optional) -->
<!-- > -->
<!-- > **Examples Needed:** -->
<!-- > - Basic compilation: cargo tcp compile [options] -->
<!-- > - Running interpreter with different flags -->
<!-- > - Setting up language server in IDE -->
<!-- > - Generating and viewing schedule visualizations -->
<!-- > - End-to-end workflow from Virtual ISA to optimized execution -->

The TCP Software Toolchain (`cargo furiosa-opt`) provides utilities for developing, testing, and optimizing Virtual ISA programs on Furiosa chips.
It complements the [Furiosa SDK's compiler](https://developer.furiosa.ai/latest/en/overview/software_stack.html#furiosa-compiler) by giving developers fine-grained control over program behavior, whether the programmer writes Virtual ISA by hand or a compiler generates it.

The toolchain consists of four components:

- **Compiler**: Translates Virtual ISA into executable code for the chip.
- **Interpreter**: Executes Virtual ISA as native Rust programs for software simulation and debugging.
- **Language Server**: Enables IDE features (autocompletion, diagnostics, navigation) via Rust's language server infrastructure.
- **Schedule Viewer**: Visualizes the execution timeline to help identify performance bottlenecks.

## Book Organization

The rest of this book is organized in the following chapters:

- **[Hello, TCP!](./hello-tcp.md)**: How TCP programming works, introduced through worked examples covering element-wise operations and tensor contractions.
- **[Mapping Tensors](./mapping-tensors/index.md)**: How logical tensors map to physical memory: axis layout, stride, padding, and tiling.
- **[Moving Tensors](./moving-tensors/index.md)**: How data moves between memory tiers (HBM, DM) and the Tensor Unit via Fetch, Commit, and DMA engines.
- **[Computing Tensors](./computing-tensors/index.md)**: How the Tensor Unit pipeline (Switching, Collect, Contraction, Vector, Cast, Transpose) transforms data each cycle.
- **[Scheduling](./scheduling.md)**: How to control the order and concurrency of operations across contexts.
- **[Kernel Examples](./kernel-examples/index.md)**: End-to-end examples showing how mapping, movement, computation, and scheduling combine into real kernels.

<!-- > **TODO**: Add an API reference chapter pointing to the generated rustdoc for `furiosa-visa-std`. -->

## License

This documentation and the entire `furiosa-opt` repository are licensed under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
