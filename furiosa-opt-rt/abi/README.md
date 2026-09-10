# Furiosa Optimizer ABI

What the compiler, the host runtime and the device agree on.

## Image

`Image` is one compiled device function.
It holds the task code of every chunk for every cluster slot, the function's weight and stacks, which argument slots a call binds as inputs and which as outputs, and the profile spans of each chunk.
The compiler encodes it; the host parses it to load and call it; the device parses the same bytes to find the chunks of its own cluster.

The encoding is a bincode header with fixed-width little-endian integers, followed by the task code.
Each chunk starts on a 256-byte line, so a DMA engine can carry one chunk without touching its neighbors.
A chunk is at most one MiB, the size of a slot on the device.

A call passes its arguments as offsets into device memory: the weight, one reserved word, one word per stack, then one word per bound slot.
Outputs are buffers the caller provides, so a completion carries only a status word.

## Registers and rings

`reg::Reg` is one memory-mapped register: naming it is the one unsafe step, reading and writing
it is not. `ring::Ring` is the driver's queue shape over three of them, `capacity` word entries
and a producer and a consumer index, one slot always empty. `ring::Producer` and `ring::Consumer`
are its two sides; each owns its index and reads the other side's register only when it must.
The bootloader, the firmware and the host each speak their rings through these, so the shape is
implemented once and a change to the driver's rings is a change in one place.

## Boot

`bootloader` is the contract with the signed bootloader: the request the host writes in the driver's queue format, where the firmware image lands, and the codes the bootloader answers with.
It is frozen; see `../bootloader`.

## Layout contract

A device reaches its peers' memory through a 16-entry remote DRAM aperture, each entry a 4 GiB window.
A device of `n` chips divides the aperture so that member `vid` owns `(16 - n) / n` consecutive windows starting at entry `n + vid * (16 - n) / n`.
For `n = 4`:

```
entry:   0   1   2   3 | 4   5   6 | 7   8   9 | 10  11  12 | 13  14  15
         (unused)      | vid 0     | vid 1     | vid 2      | vid 3
                       ^ 4 GiB per entry; a member's windows are consecutive
```

The windows must be consecutive because the PE DMA path restores only the low 32 bits of a destination address, so each window base must advance by exactly 4 GiB.
Any other layout redirects transfers without an error.

This statement is normative for the code the compiler emits.
The same formula lives in the kernel driver (`__set_remote_entry`), in `libpe` (`get_base_raw_dram`) and in the PE operating system's address map; changing one means changing all.
