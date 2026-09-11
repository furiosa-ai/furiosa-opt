# Changelog

All notable changes to `furiosa-opt` are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [v0.8.1]

### Fixed

- Ship the production-signed bootloader in runtime packages so production devices can initialize.
- Align `furiosa-opt-rt`, `furiosa-opt-abi`, and `furiosa-opt-ipc` with the release version.

## [v0.8.0]

### Added

- `HbmTensor::hbm_chip_shuffle` and `HbmTensorView::hbm_chip_shuffle` compile; only the view API
  existed before, and it reached MIR only.
- `DmTensor` and `DmTensorView` provide `chip_shuffle`, `cluster_swap`,
  `chip_slice::<Axis, Output>`, and `cluster_slice::<Axis, Output>`. Each slice call declares its
  resulting `Element` mapping, and these methods build a redistribution that `to_dm` or
  `to_dm_view` fuses into one SRAM DMA. Repeated slice calls can remove independently located axes.
- `DmTensor::{asymmetric_chip_slice,asymmetric_cluster_slice}` select one local slice per dimension index
  through the sub-context.
- `furiosa-opt-examples` provides separate chip- and cluster-reduction examples for ReduceScatter,
  AllGather, ring-style AllReduce, and Butterfly AllReduce.
- The fetch-size rule is published as `config_fetch_volume` / `FetchContext` in `furiosa-opt-lower`.
  A fetch narrower than one SRAM access word is legal when the cast widens the payload back to
  whole words, which the compiler used to reject.
- `furiosa-opt-rt`: `Device::open(topology)` for a set of chips and
  `Device::builder(topology)` to set the per-launch `timeout`; `Function::load` from a device
  function's image; `Function::launch(inputs, outputs)` awaiting a `Launch`; and
  `Function::profiled(level)` returning the spans of one launch. Host memory goes through
  `Pinned<[T]>` and `Device::write`/`read` over `(bytes, view)` pairs, where a `Buffer` view is
  a `slice`, the buffer `on` one chip, or `on_all`.
- `Device::launch(function, args)` launches with the context as the function's first parameter:
  `device.launch(f, (&a, &b))` is `launch(f, (&mut device, &a, &b))`.
- A task panic reports its message: the firmware sends it with the failing code, and the host error
  shows it.
- `HostTensor::pinned()` moves a tensor into page-locked memory, and `to_hbm(..).output(&mut hbm)` /
  `to_host(..).output(&mut host)` transfer into an existing tensor. A transfer to or from pinned
  memory is the DMA's direct source or destination; pageable memory has its pages pinned on every
  call, which costs a few microseconds per page.
- The firmware on each cluster logs through `log`, at the level the host process's logger allows
  for the `furiosa_opt_firmware` target; the lines appear in the driver's `pe_log` debugfs files.
  At `debug` every launch reports its setup time and its tasks' time.
- Device parameters support fixed-size arrays and borrowed, nested structs.
  Runtime loop indices can select tensor references such as `model.layers[index].weight.x`.
- Mapping factors accept generic axes and parenthesized counts, such as `m![Width / Factor]`
  and `m![Width / (Width / 256)]`.
- `#[unroll]` fully expands a static `for` loop before scheduling. See [Unroll a loop](https://furiosa-ai.github.io/furiosa-opt/scheduling/tuning.html#unroll-a-loop).
- `DmTensorViewMut::slice_tile` creates mutable slice views.
  Indexed writes through these views are not yet supported by NPU compilation.

### Changed

Breaking changes come first, each with what to write instead.

- A chip is a chip and a device is a device. `#[device(chip, pe)]` declares chips of PEs, so the
  runtime counts `Topology { chips, pes }` where it said `devices`, ranks a `ChipRank` where it said
  `DeviceRank`, holds `CHIP_PES` where it held `DEVICE_PES`, and reads `FURIOSA_VISIBLE_CHIPS` where
  it read `FURIOSA_VISIBLE_DEVICES`.
- `Error::Chips` is what a failure to open the chips is called, where it was `Error::Group`.
- `FunctionError::WrongTopology` names its `image_chips` the chips its message already called
  chips, where the field said `image_devices`.
- `Device` is what a process opens and what a device function runs on, replacing both `Context` and
  the `DeviceGroup` that only existed because `Device` named a chip. `Device::new(topology)` opens
  chips of its own, so two devices of one topology hold different PEs, and a device function's first
  parameter is `&mut Device`. `Context` survives where it is one: the tensor-unit and DMA contexts a
  device carries, `main`, `sub`, `tdma` and `pdma`.
- One `Topology`, in `furiosa-opt-rt` and re-exported by `furiosa-opt-std`, which held a second
  copy of the same two fields. Naming exact chips is `Builder::among`'s, so the variant that
  carried a chip list is gone with the conversion that was its only constructor.
- `TuContext::{parallel_copy_chip_slice,parallel_copy_cluster_slice}` are replaced by
  `DmTensor::{asymmetric_chip_slice,asymmetric_cluster_slice}`. The input tensor is now the receiver,
  the sub-context is the first argument, and the input scalar and dimension types plus the number of
  slice indices are inferred instead of supplied as generic arguments.
- `DmTensorView::{dm_chip_shuffle,dm_cluster_swap}` are replaced by the staged
  `chip_shuffle(...).to_dm(...)` and `cluster_swap().to_dm(...)` forms. Use `to_dm_view` as the
  terminal operation to write into an existing view, and start the same operations directly from a
  `DmTensor` when no explicit view is needed.
- `VectorInitTensor::vector_intra_slice_unzip` infers the group-tiled input `Time` mapping. Calls now
  provide only the axis name and output `Time` mapping, for example
  `vector_intra_slice_unzip::<I, m![R]>()` instead of
  `vector_intra_slice_unzip::<I, m![R, 1 # 2], m![R]>()`.
- `DmTensor::to_dm` can relayout every DM dimension. Its generic arguments are now
  `<Chip2, Cluster2, Slice2, Element2>` instead of `<Slice2, Element2>`; unchanged dimensions can be
  repeated explicitly or inferred with `_` from the expected output type.
- The device-call ABI now uses `Binding`, structural `Bindings`, and `DeviceOutput`; `Buffer`,
  `BufferList`, `FixedBufferList`, `KernelOutput`, and `KernelOutputDestination` are gone. Derived
  `DeviceSend` implementations update automatically. Manual implementations must return
  `impl __private::Bindings` from `bind()`, and manual `DeviceFn` output types must implement both
  `DeviceSend` and `__private::DeviceOutput`.
- A DM-producing slice now requires an 8-byte-aligned `AxisToSlice` stride. An asymmetric slice
  additionally requires that axis to be outermost in `Element`, and a chip shuffle requires a
  permutation of the tensor's chips. Redistribution source and slice indices must be compile-time
  constants. Invalid mappings no longer reach hardware as a different operation than the kernel
  declared.
- A launch and a transfer return `Result`: `launch(f, args).await`, `device.launch`, `to_hbm`,
  `to_host`, their `.output(..)` forms and `Device::acquire` are `Result<_, furiosa_opt_std::Error>`
  on every backend. A queue that is full, a device that fails or times out, memory that runs out,
  an HBM tensor never placed by `to_hbm`, or a device function the binary has no image for is an
  error the caller handles, not a panic. `#[derive(DeviceSend)]` follows: `bind` returns `Result`.
- `Device::new(function.topology())` names the device group it opens, where
  `Device::new()` returned a process-wide context and left the device to `Acquired::bind`.
  `#[device]` gives every function a `topology()`, and `Acquired` is gone with the singleton it
  guarded. Each acquisition opens a group of its own, so two contexts of one topology hold
  different PEs.
- A device function's image records the topology it was compiled for and what each argument slot
  must hold; a runtime rejects an image for another topology at load and a smaller buffer at
  launch.
- Device functions run on the standalone `furiosa-opt-rt` runtime instead of the vendored device
  runtime. Each NPU boots a signed bootloader and the runtime's own firmware, so a launch no longer
  crosses a runtime process; a launch from an idle device takes tens of microseconds.
  `furiosa-opt-rt` is a dependency of `furiosa-opt-std` on Linux NPU builds. The public runtime
  embeds the vendor-built firmware image; its source is not part of the public distribution.
- Profiling turns on with `FURIOSA_OPT_PROFILE`, not `TUC_PROFILE_LEVEL`; the `span::npu` tracing
  target is unchanged. An invalid level or one the image cannot provide emits a warning and runs
  the function unprofiled instead of panicking.
- The device function image format is ABI 1 of the new runtime: chunked tasks aligned to 256 bytes
  with a 1 MiB chunk limit, and a profile section sized by depth. `cargo furiosa-opt` lowers the
  limit with `FURIOSA_OPT_CHUNK_LIMIT` for tests that need several chunks.
- Primitive scalars no longer implement `DeviceSend`. Pass runtime data through HBM tensors
  and use const generics for compile-time values.
- `Kernel::alloc` is renamed to `alloc_binding`; use `launch` in place of `Kernel::run`.
- `Stride`, `Modulo`, `Resize`, and `Padding` take a mapping type as their factor: replace `N`
  with `Broadcast<N>`. Existing literal and escaped-constant `m!` / `i!` syntax is unchanged.
- `TableLookup` is renamed to `TableLookupCast`; update imports and trait bounds.
  Calls to `fetch_table_lookup()` are unchanged.
- Replace `commit_view::<Element>(dst)` with `commit_view::<_, Element>(dst)` or `commit_view(dst)`.
  Compatible destination slice mappings may now use different padding kinds.
- `Mapping` debug output uses the same mapping notation as `Display`.

### Fixed

- Fixed incorrect rejection of custom broadcasts with padded destination slices.
- Fixed a compilation failure during cycle estimation for some contraction kernels.
- Unsupported scalar conversions report the supported types at the call site.
- Empty or malformed `m!` and `i!` expressions report errors at the macro invocation.

## [v0.7.0]

### Changed

- `launch(function, args)` now returns a `Launch` builder that runs when awaited. Existing
  `launch(function, args).await` calls are unchanged; code that passes the un-awaited value to an
  API requiring `Future` must call `.into_future()`. `DeviceFn` implementations generated by
  `#[device]` also support `launch(function, args).output(&mut output).await`, which binds existing
  output HBM in NPU builds instead of allocating new output buffers for each launch. Manual
  `DeviceFn` implementations keep working through the default allocate-and-assign implementation.
- `vector_intra_slice_reduce` is no longer available in pair mode, before or after the `_zip`. The
  reducer takes no group condition, so it cannot fold only the group a `_zip` wrote. The call no
  longer compiles on a zipped tensor, and a stream that went through `vector_intra_slice_unzip` is
  refused when it is compiled.
- `vector_intra_slice_reduce` checks the accumulator slots it needs. The non-reduce `Time` positions
  inner to the outermost reduced axis each hold a live accumulator, and the reducer has 8; a mapping
  that asks for more is refused with the count it asked for. This is the bound the Time Reducer
  already checked, now shared between the two engines.
- `vector_intra_slice_unzip` bounds the interleaved group it forms. The volume inside the group axis
  becomes the branch group size, and one group waits in the VE register-file cache (1 KiB, 256 32-bit
  lanes) while the other streams in, so a larger group is refused with the volume it asked for. Read
  the group axis further inside the fetch/collect time to shrink it.

### Added

- Generic `#[device]` functions compile. Only a launch fixes their generic arguments, and the
  binary being linked is what knows every launch, so that is where their kernels now compile:
  a `cargo furiosa-opt` build embeds the kernels for each binary's own launches into that
  binary, and a launch finds its kernel in the running binary itself — no kernel files ship
  next to it. This covers every linked shape (a bin, an example, a test harness, and a cdylib
  such as a Python wheel's extension module), and launches of kernels another crate defines.
- `DmTensorView::dm_cluster_swap` swaps two clusters and lowers to EDF, with
  VISA/LIR/EDF regression coverage.
- The Transpose Engine accepts doubled input packing for 8-bit and 16-bit elements. A packet can
  now contribute 16 valid elements per cycle, and vISA lowering emits the corresponding
  `fetch_in_width_shift`; 4-bit and 32-bit doubled packing remain unsupported.
- `fetch_chip_lift`, `fetch_cluster_lift`, and `fetch_slice_lift` move axes that a plain `fetch`
  would traverse in `Time` onto a broadcast slot in the `Chip`, `Cluster`, or `Slice` placement.
  Each destination reads a different offset from its local copy, with no data transfer and no change
  to `OutPacket`. The source region must already be replicated.
- A lift works in either execution context and follows the source view selected at runtime. Slice
  bases combine constant offsets with the source address in that context. Lifted reads support
  custom-broadcast switches and DM, VRF, or TRF outputs.

### Removed

- `DmTensor::to_dm_pcopy` and `DmTensorView::to_dm_view_pcopy` are gone; neither ever compiled past
  MIR. Write the same DM to DM copy as a fetch/commit round trip through the Tensor Unit
  (`begin` -> `fetch` -> `collect` -> `commit_trim` -> `commit` / `commit_view`), or as
  `DmTensor::to_dm` / `DmTensorView::to_dm_view` for a DMA copy.

## [v0.6.0]

### Changed

- The `Emulation` backend type is renamed to `Cpu`: it is not a mode you select but what
  running on the CPU means, and a plain `cargo build`/`cargo test` is that build.

### Removed

- The `typecheck` backend is gone: the `Typecheck` backend type, its `PhantomStorage`, the
  `--cfg backend="typecheck"` world, and `cargo furiosa-opt --backend typecheck`. Its two jobs
  moved to their proper homes: mapping and shape validation of a kernel body is
  `cargo furiosa-opt compile` (the translator verifies every selected kernel statically, and
  catches device constraints phantom execution never could), and value validation is a CPU
  test. The `backend` cfg is now npu-only: a build without the cfg IS the CPU build, so a
  plain `cargo build`/`cargo test` and a proxy-less invocation are the same build, byte for
  byte.
- Every API that took a raw address is gone: `HbmTensor::from_addr`, `TrfTensor::from_addr`,
  `VrfTensor::from_addr`, the `TrfAddress` enum, and the `*_at` variants (`to_dm_at`,
  `to_dm_pcopy_at`, `dma_gather_scaled_at`, `to_trf_at`, `to_vrf_at`, `commit_at`). An address was
  never the caller's to pick: the compiler places a kernel's own tensors, the scheduler assigns the
  register-file regions, and the runtime allocates what the host uploads. Write the address-free
  call instead, and where a device function needs an output, `HbmTensor::new()` (the HBM peer of
  `DmTensor::new()`) takes an allocation from whichever of the two is in charge.
- `f16` is no longer a vISA scalar, and leaves the DMA dtype listing with the rest. RNGD has no
  native arithmetic on it, only a fetch widen and a cast narrow, so a kernel could name the type but
  never compute with it.

### Changed

Breaking changes come first, each with what to write instead.

- `cargo furiosa-opt compile` refuses a filter that names a generic fn, per fn: a generic fn
  has no instantiation to compile on its own. Launch it from a target and build that target
  (`cargo furiosa-opt build`/`test`); `compile` stays the direct artifact tool for concrete fns.

- `furiosa-opt-examples` now holds only kernels that compile end to end, plus a `negative` module of
  fixtures that must be refused. Legal functions the compiler cannot lower to an EDF yet (`matmul`,
  `attention`, `vrf_add`, the value-form `runtime_if`, and the parallel-copy and shuffle transfers)
  moved out of the crate, so an example you find here compiles unless it is under `negative`.

- A VE operand that applies to only some elements is written through `Branched`, which opens at the kind of hardware slot it drives and can drive several: `Branched::imm(guard, 100).imm(other, 200).rf(rest, &vrf)`. `imm` takes the next of a pass's three immediate registers, `rf` its one register-file port, once and last. An unconditional operand is still written bare (`.vector_fxp(op, 16384i32)`), and `VeOperand` is gone as a name a kernel writes.
- `TagFilter` is `TagGuard`, and constrains the execution id bit by bit rather than only by group. Write `TagGuard::all()` for `TagFilter::All` and `TagGuard::group(id)` for `TagFilter::Group { id }`. `TagGuard::matches([BitReq; 4])` and `not_matches` are new, taking one `BitReq` per tag bit: `One`, `Zero` or `Ignore`.
- Guards that admit the same execution ids are now the same value, so comparing guards compares predicates: `matches([Ignore; 4])` is `all()`, and `not_matches` of a pattern constraining bit 3 alone is that `group()`.
- A comparison is `Cmp<D>`, typed by the stream's scalar and carrying its boundary positionally, replacing `InputCmp` and its `InputCmpI32`/`InputCmpF32` halves: `Cmp::Less(0)` in place of `InputCmp::I32(InputCmpI32::Less { boundary: 0 })`. `TagMode` is `TagMode<D>` to match.
- `TagMode::ValidCount` and `TagMode::Vrf` are withheld. Neither ran on the host, and `ValidCount` compiled for the device, so a kernel naming it failed only once it ran.
- A guard that could never fire is refused instead of compiled: `not_matches([Ignore; 4])`, which no execution id satisfies, and any slot after an unconditional one, which already claims every element left over.
- One `Cast` no longer stands for four hardware conversion sets. `Cast` is the host-side functional model, and each stage names its own set: `FetchCast` for the Fetch Adapter, `CastEngineCast` for the Cast Engine, `CommitCast` for the Commit Adapter. `commit_cast` used to take anything with a `Cast` impl while the commit unit performs exactly `f32 -> bf16`.
- `commit_cast` and `commit_cast_relu` are two methods, and the `Activation` argument they shared is gone. The hardware has no mode to select, so the choice is the method name rather than a value the compiler must const-evaluate.
- `ContractionCast` is now the seven operand types the multiplier actually accepts: `i4`, `i5`, `i8`, `i9`, `f8e4m3`, `f8e5m2`, `bf16`. `i16`, `u8`, `i32` and `f32` are gone from it; route those through the Vector Engine's Intra-Slice Reduce. The accumulation width moved to its own `ContractionAccumulator` (`i32`, `f32`). On a stream of an unsupported type the contraction method is not offered at all, rather than compiling and failing later.
- A DM `reshape` must preserve its `Element` partition, and an HBM `reshape` its whole size. `reshape` relabels axes over an unchanged buffer, and a resize used to slip through under the same name. Where you meant to change an extent, write `pad` / `unpad` or `tile`.
- A vector stash is read at the scalar and the way it was written. Reinterpret before the write if you want another scalar, and cross-way stash still needs a buffering split. In pair mode a stash is a compile error rather than a run-time panic under `emulation`.
- `dma_tails` reports the largest burst the DMA can actually issue. A packed sub-byte transfer used to come back with a tail of 8 elements where 128 move in one burst.
- A `SwitchConfig` that broadcasts one part of a slice while walking an axis in another compiles, where it used to be refused as a whole. A configuration that is still wrong now names the axis at fault instead of failing as one opaque mapping.
- An unscaled gather refuses an index tensor larger than the hardware allows, instead of lowering something the engine cannot run. The two ways out are in the examples: switch to the scaled gather, or split the index with a loop into smaller unscaled gathers.
- The book is one path for a kernel author: Quick Start is a thin index over Setup and Tooling, Tensor and Contraction, Kernel Design, and Kernel Validation, and the tutorial material lives beside the reference chapter it belongs to.

### Added

- `pad` and `unpad` views, the extent re-declaration a tile could not express: the same cells, declared at a wider or narrower extent, on the DM and HBM views alike. Padding an axis boundary is a re-declaration and is allowed; padding an inner digit of a split axis is not, and narrowing is `tile`'s direction rather than a silent drop.
- `vector_reinterpret::<D2>()` reads the VE's 32-bit stream as another scalar, bit for bit: `1.0f32` becomes `0x3f80_0000`, not `1`. It costs no hardware, so it goes anywhere in a chain and keeps the stage, `Way`, filter context and stash typestate, on `VectorTensor` and `VectorTensorPair` both. The use is f32 bit manipulation, since shift and the Fxp cluster take only `i32`.
- Mapping cells are public. `Cell` says what a position holds, a live index, a padding kind, or out of bounds, and `CellExt` reads them off a mapping.
- All four baked table-lookup conversions work: `f4e2m1` to `f8e4m3` or `f8e5m2`, and `f8e4m3` to `bf16` or `f8e5m2` to `f32`. Two of the four used to fail to compile.
- `FetchCast` gains `i16 -> i32` and `f32 -> bf16`.
- `Tensor::contraction_prewidened`, for writing a host answer key whose operands are already at the accumulator width. `Tensor::contraction` still takes operand-width input and accumulates at `ContractionCast::Output`.
- `FURIOSA_VISIBLE_CHIPS` (comma-separated chip ids) restricts which chips a process will acquire, so two runs can pin disjoint chips instead of racing for the lowest free one. Unset or empty keeps every exposed chip a candidate.

### Fixed

- A conditional operand's slots fire in the order they are written: an element takes the first slot whose guard it satisfies, so a trailing `TagGuard::all()` is the pass's `else`. Software execution had applied every matching slot in turn, disagreeing with the device.
- Unsupported shapes are refused against the kernel's own span instead of panicking inside the compiler: a `TagMode::AxisToggle` pass, a stash read guarded on the group bit alone, and chaining more ops than a VE cluster holds.
- A tiled view built on the host arrived at the NPU as tile 0. `tile` recorded its offset in the view's mapping only, and the Npu backend reads the address, so every host-side tile named the base tensor. The offset now rides the address, and it may be a run-time value, which is what a per-token embedding lookup needs. `emulation` was never affected.
- An HBM view tile naming an inner digit of an axis windowed the whole axis instead. Tiling an `m![H]` argument with `m![H % 120]` dropped the outer repetition from the view, so the tile read one window spanning all of `H` rather than the digit it named. The window now falls where the tile says, on the `to_hbm_view` write path as well as the read.
- `HbmTensor::reshape` freed the device allocation it was supposed to keep alive, so the handle it returned could name freed HBM.
- A kernel argument typed `f4e2m1` or `i4` panicked with `size_in_bytes of f4e2m1 should not be used` while loading its EDF, before any tensor was uploaded. The width travels in bits, and a 4-bit tensor keeps its packed size on the device.
- An `i4` by `i4` contraction failed to compile to EDF with `unsupported types (input: ..., rf: ...)`. A TRF holds `i4` weights promoted to `i5`, and that promotion was left out of the loads the compiler accepted, so no int4 weight contraction reached the device.
- A kernel that scatters into an HBM tensor and then reads it back observed the pre-scatter contents: the read was scheduled before the scatter.
- A stash write placed after a `vector_reinterpret` silently moved to the branch stage and stashed the pass input, so `|x| + |x|` gave `0.0` on negative input.
- `VectorTensorPair::vector_logic` had no vISA translation and died with `Unknown primitive`.
- An operand write to a cluster with no VRF write port, the concat layer for one, was silently moved to another stage instead of being refused.
- The host model of `bf16` matches the hardware: RNGD has no subnormal in `f32` or `bf16` and canonicalizes NaN, where the host did plain IEEE. An answer key can now carry subnormals and NaN payloads and still match the device bit for bit.

### Known issues

- A stash read guarded on the group bit alone does not lower. A cached read needs two things the compiler keeps apart -- which elements the operand applies to, and which group wrote the slice being read -- and a group-shaped guard cannot be told from a zip's read. Constrain another bit as well, or drop the guard.
- `TagMode::AxisToggle` does not lower. `vector_intra_slice_unzip()` is the supported route to a group split.
- A DMA cannot place a tile whose padded axis is an inner digit of a split. The reduced case is pinned in the examples as `mre/dma_padded_inner_digit`.
- An inner-digit tile is supported on an axis the HBM argument names directly. A tile reaching into an axis nested under another is still refused at compile time.

## [v0.5.1]

### Changed

- `compile` names the filter that excluded the kernels it skipped, and says nothing when none was given.

### Fixed

- `cargo-furiosa-opt --version` reported the compiler crate's own version instead of the release it shipped in.

## [v0.5.0]

### Changed
Breaking changes come first, each with what to write instead.

- Transfers no longer take an address. Call `to_hbm` and `to_dm` without one, and reach for `to_hbm_at` or `to_dm_at` only when the address matters. Build a DM tensor with `DmTensor::new` in place of `DmTensor::from_addr`. The runtime allocator places what `to_hbm` produces.
- Host buffers convert through `from_vec` and `into_vec`, replacing `from_buf` and `into_buf`.
- `Tensor::reshape` consumes the tensor it reshapes, on owned tensors and on views alike, and a size mismatch now fails to compile instead of at run time.
- Reading a vector stash twice no longer compiles. The operand type carries the read-once rule that was previously a convention.
- `furiosa-mapping`: `replace_padding` is gone. Write `padding(_, Top)` for the same result.
- A dense contraction no longer materializes the broadcast of its operands, which dominated wide-output GEMM on the host. Output is byte for byte what it was.
- Host to HBM and HBM to host transfers on the NPU backend go through the DMA heap without a staging copy.
- A binary built for `backend="npu"` carries its compiled kernels, so it runs on a machine that has no compiler installed.
- The vendored device runtime moves to a revision whose `furiosa_kernel_run` and `furiosa_profiled_run` take the runtime by mutable pointer.
- A vISA scalar constant carries its domain, integer or float, instead of one shared representation.
- `tracing-subscriber` is taken with the `fmt`, `std`, `env-filter` and `json` features.
- `x86_64-unknown-linux-gnu` is the only host the published artifacts cover; the README says so and points macOS at a container.

### Added
- A kernel can return several HBM tensors as a tuple, in the order it returns them.
- Profiling a run on the device. Install a `tracing` subscriber on the `span::npu` target and each launch reports its spans with their cycle windows. Recording is off until `TUC_PROFILE_LEVEL` names `info` or higher, so a subscriber alone sees nothing. `furiosa-opt-examples` has a minimal example.
- Gathering and scattering DRAM rows through an index tensor: `dma_gather_scaled`, `dma_gather_unscaled`, and `dma_scatter`, with the `dma_tails` mapping primitive to describe the burst packet's tail axis.
- Filling a DM tensor view with a typed constant: `memset(value)`.
- Baked table lookup as a fetch adapter, `fetch_table_lookup`, covering paired nibbles and a non-paired `f8e4m3` to `bf16` table, plus `fetch_zero_point_sub` for zero-point subtraction on fetch.
- Device scalar types `i5`, `i9`, and `f8e5m2`. A 4-bit tensor packs two codes per byte.
- A tile offset can be an expression, such as `tile(i * 2)`.
- Engine configurations report what is wrong before a run rather than producing a bad result. `furiosa-opt-lower` publishes the verifications, and the switch engine checks a named `SwitchConfig` and its custom-broadcast snoop bitmap through a structured `SwitchError`.
- The `emulation` backend runs clip, float, floating-point divide, and local reduce vector nodes, which it used to reject.
- The book has an appendix on the `cargo furiosa-opt` CLI.
- A runtime `if` over a non-unrolled loop index, branching on any comparison of runtime scalars. The statement form, where each arm writes its own output, compiles through to EDF; a value `if` that merges a tensor lowers through vISA and runs under `emulation`, but the scheduler cannot lower the merge yet, so it stops at the vISA stage. `furiosa-opt-examples` covers both shapes.
- A kernel per supported chip and PE topology, so a device config wired wrong shows up as a layout mismatch rather than a bad result.

### Removed
- The `simulation` backend and its `MathStorage` interpreter. `emulation` replaces it and is what a plain `cargo build` or `cargo test` now uses.

### Fixed
- A `tile` start offset on the fetch path counted bytes instead of elements, so a `bf16` tile that asked for element 16 landed on element 8; it now counts elements, as the commit side and the DMA path always did, and the `commit_view_tile` example guards it.
- A tensor that a nested loop body read or aliased from an enclosing scope produced a wrong result; it now lowers as written.
- A write through an `HbmTensorViewMut` covering a whole parameter never reached the caller's tensor; it now lands in place.
- A valid Time Reducer fold with a `Sequential` accumulator was rejected for exceeding the accumulator buffer; the bound now counts only the axis each mode pads, and the error spells out the product it checked.
- A device call the compiler could not translate, a generic `#[device]` entrypoint, and a scalar cast it could not evaluate each surfaced as an internal compiler error; they now report what is wrong.
- A process could not start while chip 0 was busy, even with every other chip idle; it now takes the first free NPU among the chips the host exposes.
- An indirect DMA rejected a multidimensional `validlen`.
- A DSL tactic context was not materialized on a single-chip device.
- The scheduler allocated provenance tensors statically, spending memory nothing read.
- A scalar DRAM to SRAM bias transfer is staged as an aligned single-cell tail, which the DMA engine accepts.

## [v0.4.0]

### Changed
- HBM, DM, and TU tensors now carry an optional address; explicit-address transfers moved to dedicated `*_at` methods (`to_trf_at`, `to_vrf_at`, `to_dm_at`, `to_hbm_at`, `to_dm_pcopy_at`, `commit_at`).

### Added
- `furiosa-mapping`: escaped constant numbers in `m!` and `i!`.

### Fixed
- Fixed secure boot support.

## [v0.3.0]

### Added
- Added `simulation` and `typecheck` backends.

### Changed
- Revised some APIs of virtual ISA.
- `Tensor::reshape` reworked as a `RawTensor` primitive in `furiosa-visa-std`.

### Fixed
- Fixed the binary path to download in `cargo-furiosa-opt`.

## [v0.2.0] - Initial release
