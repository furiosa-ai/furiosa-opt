# End-to-End Cases

End-to-End Cases connect the mapping, movement, computation, and scheduling contracts into composed workloads.
Choose the nearest case below, then verify its referenced API before adapting mappings.

## Choose a Starting Point

Start implementation from a [Quick Start](../quick-start.md) pattern whenever it can express the required API behavior.
The remaining pages explain design choices; verify their code against the current API before using it.

| Need | Start with | Technical focus |
|------|------------|----------------------------|
| Qwen3 decoder step | [Case Study: Transformer](./transformer.md) | Model mental model, baseline kernel map, decode-only boundaries, portable oracle, and schedule data from the current transformer example. |
| Runtime loop-body skipping | [Case Study: Skip Loop Bodies at Runtime](./skip-loop-bodies-at-runtime.md) | Load an HBM scalar and guard work inside a statically bounded loop. |
