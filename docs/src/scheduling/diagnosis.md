# Diagnosis

Diagnosis converts a schedule view into one testable bottleneck hypothesis.
Separate data dependencies from shared-resource waits.
A schedule view does not establish a speedup by inspection alone.

## Inspect a dumped schedule

Generate and open the JSON with the [Schedule Viewer tool reference](../tools/schedule-viewer.md).
The viewer exposes the scheduled timeline, operation context, lifetime, connected tensors, and operator description.
Select a node to trace its inputs and outputs.
Zoom the cycle range to isolate the suspected region.

## Read the timeline

The final scheduled cycle defines the makespan, and the intervals that determine it provide the diagnosis:

1. Identify the interval that reaches the final cycle.
2. Check whether its predecessor is a memory dependency, a shared resource, or an unscheduled gap.
3. Compare the relevant context lanes to see whether another operation could overlap.
4. State one candidate change and the reason it should alter that interval.

The schedule shows static compiler behavior.
It does not replace a value oracle, and it does not prove device throughput.

## Keep the comparison valid

Record the source revision, release profile, device-function filter, shapes, element types, mappings, and schedule file for each candidate.
Change only the selected kernel choice between candidates.
If any of those inputs changes, treat the result as a new baseline rather than a tuning comparison.
