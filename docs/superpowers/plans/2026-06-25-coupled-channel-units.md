# Coupled Channel Unit Pruning Plan

## Goal

Refactor the generic structured pruner so `PruningGroup` is treated as a dependency scope recipe, while concrete search and pruning operate on `CoupledChannelUnit`, `AtomicPruneUnit`, and `ConcreteCoupledPruningGroup`.

## Implementation Steps

1. Add fast toy tests for:
   - single-root-index `CoupledChannelUnit` expansion;
   - local vs global coupled-channel selection;
   - grouped conv `shared_local_mean`, `independent_group_topk`, and `remove_groups`.
2. Add dataclasses and helpers:
   - `CoupledChannelUnit`;
   - `AtomicPruneUnit`;
   - `ConcreteCoupledPruningGroup`;
   - dependency scope, unit, candidate, and concrete group CSV/JSON rows.
3. Extend importance:
   - scope-level per-root-channel importance aggregated over all `GroupItem`s;
   - candidate-level importance from source coupled units;
   - source metadata and invalid-value protection metadata.
4. Add selection layer:
   - `local_scope`;
   - `global_coupled_channel`;
   - `constrained_global`;
   - grouped conv constrained selection modes.
5. Extend grouped conv physical pruning:
   - keep existing shared-local `keep_groups`;
   - add independent per-group repacking while keeping `groups` unchanged;
   - keep `remove_groups` explicit and gated.
6. Integrate the new pipeline into `tests/test_general_pruner.py`:
   - new CLI flags;
   - stable CSV/JSON outputs;
   - selection summary.
7. Run focused unit tests and existing fast pruning tests.

## Design Notes

- Existing `PruningGroup.local_keep()` transforms remain the authoritative dependency recipe.
- `CoupledChannelUnit` is exactly `DependencyScope + root_idx`.
- Grouped conv selection always consumes scope-level aggregated importance, never grouped-layer-only importance.
- `global_coupled_channel` must not apply naive top-k inside grouped conv scopes.
- Residual groups remain protected by default unless explicitly unprotected by caller.
