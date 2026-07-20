# `cargo-furiosa-opt`

`cargo-furiosa-opt` is a thin wrapper around cargo for the Furiosa NPU compiler toolchain.
Anywhere you would run `cargo`, run `cargo furiosa-opt` instead: every cargo argument passes through verbatim, and the wrapper selects a backend and compiles the kernels your build needs.

For installation, see [Installation](../introduction.md#installation).

## Usage

```text
cargo furiosa-opt [--backend <backend>] <command> [args]
cargo furiosa-opt compile [FILTER]... [options]
```

The first form is the cargo passthrough: `<command>` is any cargo subcommand (`build`, `test`, `run`, …) and `[args]` are forwarded to cargo unchanged.
The second form compiles kernels directly; see [Direct compilation](#direct-compilation-cargo-furiosa-opt-compile).

## What the wrapper adds

Compared to plain cargo, `cargo furiosa-opt` does a few things around the cargo invocation:

- **Selects a backend.** It builds/tests/runs with your specified backend. See [`--backend`](#--backend-backend).
- **Builds the kernels you need, automatically.** Under `--backend npu`, `cargo furiosa-opt` compiles the necessary `#[device]` functions into kernels before the cargo build so the resulting binary can load them at runtime. See [Automatic kernel builds](#automatic-kernel-builds).

## `--backend` _backend_

Selects the backend that evaluates each kernel.
Because every argument after it passes through to cargo verbatim, `--backend` must appear **before** the cargo subcommand:

```bash
cargo furiosa-opt --backend typecheck run
cargo furiosa-opt --backend npu test my_test
```

**Default:** `simulation`

**Possible values:**

- `typecheck`: Type-level mapping/shape validation; value-level loops are short-circuited with phantom tensors. No NPU hardware required.
- `simulation`: Host-side interpretation of tensor ops. No NPU hardware required.
- `emulation`: Host-side simulation of the NPU device path. No NPU hardware required.
- `npu`: Real NPU dispatch via compiled kernels. Triggers automatic kernel compilation. Requires a physical NPU and the Furiosa SDK.

See [Backends](../introduction.md#backends) for a task-oriented overview.

## Automatic kernel builds

Under `--backend npu`, `cargo furiosa-opt` runs a kernel pre-compilation step before handing off to cargo.
The pre-step runs only when it will matter. It is skipped unless **both** of the following hold:

- The cargo subcommand builds or executes code: `build`, `check`, `test`, `run`, `bench`, or `doc`.
- The invocation is not a `-h` / `--help` query.

When it runs, the pre-step compiles only the kernels the build actually needs:

- It reads cargo's unit graph — honoring your `-p` / `--package` and workspace selection — to find which kernel packages the command builds. A crate is a kernel package if its `Cargo.toml` declares `[package.metadata.furiosa-opt]` (see [Layout](../introduction.md#layout)).
- When the command resolves to specific runnable targets, such as a test, example, or binary, the compiler scans each target and compiles only the `#[device]` functions reachable from it.
- Otherwise, it falls back to compiling every kernel in the selected packages.

Compilation is cached per kernel so unchanged kernels are not recompiled on the next run.
Artifacts are written under the output directory (see [`FURIOSA_OPT_OUT_DIR`](#environment-variables)).

## Direct compilation: `cargo furiosa-opt compile`

`cargo furiosa-opt compile` compiles `#[device]` functions directly, without the cargo passthrough.
It always builds NPU kernels, so it takes no `--backend`.

```bash
# Compile every #[device] function in every kernel package.
cargo furiosa-opt compile

# Compile only the functions matching a filter, in one package.
cargo furiosa-opt compile transpose_simple -p my_kernels

# Compile a single function and dump its schedule for the Schedule Viewer.
cargo furiosa-opt compile transpose::transpose_simple \
  --dump-schedule schedule.json
```


### `[FILTER]...`

Specifies the set of `#[device]` functions to compile.
Filters are matched as a substring against `#[device]` function names as a full path (`abc::def::foo`); a function is compiled if it matches any filter.
When omitted, all device functions are compiled.

### `-p`, `--package` _name_

Restrict compilation to the named kernel package.
The option may be repeated to select multiple kernel packages.
When omitted, all kernel packages are compiled.

### `--message-format` _format_

Diagnostic format forwarded to the compiler (e.g. `json`), so tools can machine-parse kernel-compile failure diagnostics.

### `--dump-visa` _file_

Dump the intermediate vISA to a file.
Should only be used when compiling a single kernel.


### `--dump-schedule` _file_

Dump the schedule as JSON for the [Schedule Viewer](./schedule-viewer.md).
Should only be used when compiling a single kernel.


## Environment variables

### `FURIOSA_OPT_OUT_DIR`

Kernel output directory.
Defaults to `<workspace target>/furiosa-opt/kernel`.

