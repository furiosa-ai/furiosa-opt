# Kernel Validation

The validation sequence distinguishes structural validity, numerical correctness, hardware behavior, and static schedule data.

1. **Kernel compilation** (`cargo furiosa-opt compile`) translates and verifies every `#[device]` body, catching mapping and shape errors without executing anything.
2. **CPU runs** (plain cargo) compare host-side tensor results with a host oracle.
3. **NPU execution** runs the compiled executable device format (EDF) through the Furiosa SDK and physical hardware.
4. **Schedule data** inspects emitted schedule JSON and compares makespan with release, shapes, types, and mappings fixed.

Kernel compilation is static: it proves that every selected kernel translates and verifies, without providing numerical-correctness proof.
Compare schedule makespan only when the release, shapes, data types, and mapping are fixed.
A schedule is not a throughput measurement by itself.
Use [Scheduling and Tuning](../scheduling/index.md) to compare fixed schedule candidates.
Record the source revision, backend, shapes, mappings, oracle result, and schedule artifact for each comparison.
