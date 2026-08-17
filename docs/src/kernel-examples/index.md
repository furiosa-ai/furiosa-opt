# End-to-End Cases

End-to-End Cases connect the mapping, movement, computation, and scheduling contracts into composed workloads.
Choose the nearest case below, then verify its referenced API before adapting mappings.

Legend: **Verified and tested** identifies a reader-ready pattern with current source and tests.
**Design guide** identifies explanatory material whose API details must be verified before implementation.

## Choose a Starting Point

Start implementation from a [Quick Start](../quick-start.md) pattern whenever it can express the required API behavior.
The remaining pages explain design choices; verify their code against the current API before using it.

| Need | Start with | Technical focus and status |
|------|------------|----------------------------|
| Qwen3 decoder step | [Case Study: Transformer](./transformer.md) | Model mental model, baseline kernel map, decode-only boundaries, portable oracle, and schedule data from the current transformer example. |
