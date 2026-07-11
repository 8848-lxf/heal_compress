# Global one-shot pruning

`pruning.selection.global_ranking.select_global_units` sorts all selectable
atomic units by normalized score and deterministic stable ID, then satisfies a
global channel or parameter budget subject to protection and domain limits.
Its output is a `SamplingPruningRequest`; it does not edit the model.

`build_physical_pruning_plan` expands requests into their complete dependency
closure, merges duplicate module-axis requests, rejects conflicting group maps
and freezes every index against the unmodified model. `legalize_pruning_plan`
applies alignment and grouped legality at scope level. An alignment repair is
propagated to BN, successor inputs and residual members; a non-identity mapping
without a saved index map fails closed.

`materialize_pruning` preflights the entire plan, clones by default, and slices
each affected module once. Every sampling request receives one terminal ledger
status: `applied`, `repaired`, `merged` or `skipped`. There is no per-layer
rescore/reindex loop.

