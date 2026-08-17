# Kernel Optimizer

The Kernel Optimizer provides build, test, run, and direct-compile workflows for Furiosa Optimizer kernels.
The `cargo-furiosa-opt` command is the entry point for these workflows.

`cargo-furiosa-opt` is a thin wrapper around cargo for the Furiosa NPU compiler toolchain.
`cargo furiosa-opt` is the NPU build; a plain `cargo` invocation builds and runs on the CPU.
Every cargo argument passes through verbatim, and the wrapper compiles the kernels your build needs.

For installation, see [Installation and First Program](../quick-start/setup-and-tooling.md#installation-and-first-program).

## First Commands

Run a binary on the CPU, then check mappings, compare values with an oracle, and export schedule data.

```bash
cargo run --release --bin gemm
cargo furiosa-opt compile
cargo test --release --bin gemm
cargo furiosa-opt compile gemm_kernel --dump-schedule schedule.json
```

Quick Start introduces these commands in context.
This section gives the complete `cargo-furiosa-opt` command reference.
Use [Kernel Validation](../quick-start/kernel-validation.md) to interpret compile, CPU, NPU, and schedule results.

## Usage

```text
cargo furiosa-opt <command> [args]
cargo furiosa-opt compile [FILTER]... [options]
```

The first form is the cargo passthrough: `<command>` is any cargo subcommand (`build`, `test`, `run`, …) and `[args]` are forwarded to cargo unchanged, compiled for the NPU.
The second form compiles kernels directly.
See [Direct compilation](#direct-compilation-cargo-furiosa-opt-compile).

A plain `cargo build`/`cargo test`/`cargo run` is the CPU build: kernels run on the host, and no NPU hardware or SDK is required.

## What the wrapper adds

Compared to plain cargo, `cargo furiosa-opt` does two things around the cargo invocation:

- **Targets the NPU.**
  Kernels dispatch to real hardware.
  It requires a physical NPU and the Furiosa SDK.
- **Builds the kernels you need automatically.**
  It compiles necessary `#[device]` functions before the cargo build so the resulting binary can load them at runtime.
  See [Automatic kernel builds](#automatic-kernel-builds).

See [Quick Start: Kernel Optimizer](../quick-start/setup-and-tooling.md#kernel-optimizer) for a task-oriented overview.

## Automatic kernel builds

`cargo furiosa-opt` runs a kernel pre-compilation step before handing off to cargo.
The pre-step runs only when it matters.
It is skipped unless **both** of the following hold:

- The cargo subcommand builds or executes code: `build`, `check`, `test`, `run`, `bench`, or `doc`.
- The invocation is not a `-h` / `--help` query.

When it runs, the pre-step compiles only the kernels the build actually needs:

- It reads cargo's unit graph, honoring your `-p` / `--package` and workspace selection, to find kernel packages.
  A crate is a kernel package if its `Cargo.toml` declares `[package.metadata.furiosa-opt]`.
- When the command resolves to specific runnable targets, such as a test, example, or binary, the compiler scans each target and compiles only the `#[device]` functions reachable from it.
- Otherwise, it falls back to compiling every kernel in the selected packages.

Compilation is cached per kernel so unchanged kernels are not recompiled on the next run.
Artifacts are written under the output directory (see [`FURIOSA_OPT_OUT_DIR`](#environment-variables)).

## Direct compilation: `cargo furiosa-opt compile`

`cargo furiosa-opt compile` compiles `#[device]` functions directly, without the cargo passthrough.

```bash
# Compile every #[device] function in every kernel package.
cargo furiosa-opt compile

# Compile only the functions matching a filter, in one package.
cargo furiosa-opt compile transpose_simple -p my_kernels

# Compile a single function and dump its schedule for the Schedule Viewer.
cargo furiosa-opt compile transpose::transpose_simple \
  --dump-schedule schedule.json
```

### Compilation pipeline and dump options

The dump flags below each capture a different point of the kernel compilation pipeline, which runs in three stages:

1. **Stage 1 — MIR → vISA.**
2. **Stage 2 — vISA → LIR.**
3. **Stage 3 — LIR → EDF (the kernel binary).**

Each public dump flag targets a stage and emits an artifact for a specific consumer.
All dump flags are single-kernel options.

- `--dump-visa`: Stage 1 vISA text for compiler/backend inspection.
- `--dump-ir`: Stage 2 LIR as a bincode binary for compiler tooling.
- `--dump-dfg`: Stage 1 vISA data-flow graph serialized as compiler IR binary data to the requested file for graph inspection.
- `--dump-graph`: Stage 2 LIR graph as JSON for the IR graph viewer.
- `--dump-summary`: Stage 3 LIR-to-EDF compilation summary written to the requested directory.
- `--dump-schedule`: Stage 3 scheduling after LIR is lowered to `ResourceLir`, emitted as JSON for the [Schedule Viewer](./schedule-viewer.md).

### `[FILTER]...`

Specifies the set of `#[device]` functions to compile.
Filters are matched as a substring against `#[device]` function names as a full path (`abc::def::foo`).
A function is compiled if it matches any filter.
When omitted, all device functions are compiled.

### `-p`, `--package` _name_

Restrict compilation to the named kernel package.
The option may be repeated to select multiple kernel packages.
When omitted, all kernel packages are compiled.

### `--message-format` _format_

Diagnostic format forwarded to the compiler (e.g.
`json`), so tools can machine-parse kernel-compile failure diagnostics.

### `--dump-visa` _file_

Dump the intermediate vISA to a file.
Should only be used when compiling a single kernel.

### `--dump-ir` _file_

Dump the intermediate LIR (Low-Level IR) as a bincode binary to a file.
Should only be used when compiling a single kernel.

### `--dump-schedule` _file_

Lower LIR to `ResourceLir` and dump the resulting schedule as JSON for the [Schedule Viewer](./schedule-viewer.md).
Should only be used when compiling a single kernel.

### `--dump-dfg` _file_

Dump the Stage 1 vISA data-flow graph to a file.
Should only be used when compiling a single kernel.

### `--dump-graph` _file_

Dump the Stage 2 LIR graph as JSON for the IR graph viewer.
Should only be used when compiling a single kernel.

### `--dump-summary` _dir_

Dump the Stage 3 LIR-to-EDF compilation summary to a directory for diagnostic tooling.
Should only be used when compiling a single kernel.

## Environment variables

### `FURIOSA_OPT_OUT_DIR`

Kernel output directory.
Defaults to `<workspace target>/furiosa-opt/kernel`.

