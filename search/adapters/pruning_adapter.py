"""Adapter over the formal plan-first structured pruning pipeline."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..hashing import canonical_json_hash

from ..candidate import CandidatePhenotype


class FormalPruningAdapter:
    """Build, legalize, materialize, snapshot, hash, and validate pruning."""

    pipeline_name = "formal_plan_first_pruning"

    def __init__(
        self,
        *,
        build_plan_fn: Callable[..., Any] | None = None,
        legalize_plan_fn: Callable[..., Any] | None = None,
        materialize_fn: Callable[..., Any] | None = None,
        snapshot_fn: Callable[..., Any] | None = None,
        hash_fn: Callable[..., Any] | None = None,
        validate_fn: Callable[..., Any] | None = None,
    ) -> None:
        if any(value is None for value in (build_plan_fn, legalize_plan_fn, materialize_fn, snapshot_fn, hash_fn, validate_fn)):
            try:
                from pruning.api import (
                    build_physical_pruning_plan,
                    build_physical_structure_snapshot,
                    compute_physical_hashes,
                    legalize_pruning_plan,
                    materialize_pruning,
                    validate_physical_model,
                )
            except ImportError:
                from heal_compress.pruning.api import (
                    build_physical_pruning_plan,
                    build_physical_structure_snapshot,
                    compute_physical_hashes,
                    legalize_pruning_plan,
                    materialize_pruning,
                    validate_physical_model,
                )
            build_plan_fn = build_plan_fn or build_physical_pruning_plan
            legalize_plan_fn = legalize_plan_fn or legalize_pruning_plan
            materialize_fn = materialize_fn or materialize_pruning
            snapshot_fn = snapshot_fn or build_physical_structure_snapshot
            hash_fn = hash_fn or compute_physical_hashes
            validate_fn = validate_fn or validate_physical_model
        self.build_plan_fn = build_plan_fn
        self.legalize_plan_fn = legalize_plan_fn
        self.materialize_fn = materialize_fn
        self.snapshot_fn = snapshot_fn
        self.hash_fn = hash_fn
        self.validate_fn = validate_fn

    def request_from_phenotype(
        self,
        phenotype: CandidatePhenotype,
        atomic_units: Sequence[Any],
    ) -> Any:
        """Create a SamplingPruningRequest from pruned stable IDs."""

        try:
            from pruning.types import SamplingPruningEntry, SamplingPruningRequest
        except ImportError:
            from heal_compress.pruning.types import SamplingPruningEntry, SamplingPruningRequest

        selected = set(phenotype.pruned_unit_ids)
        group_keep_by_scope = {
            str(scope): {int(group): [int(value) for value in values] for group, values in dict(mapping).items()}
            for scope, mapping in dict(phenotype.metadata.get("group_keep_map_by_scope") or {}).items()
        }
        group_prune_by_scope = {
            str(scope): {int(group): [int(value) for value in values] for group, values in dict(mapping).items()}
            for scope, mapping in dict(phenotype.metadata.get("group_prune_map_by_scope") or {}).items()
        }
        aggregated: dict[tuple[str, str], dict[str, Any]] = {}
        channel_cost = 0
        parameter_cost = 0
        for unit in atomic_units:
            stable_id = str(getattr(unit, "stable_id"))
            if stable_id not in selected:
                continue
            if bool(getattr(unit, "protected", False)):
                continue
            constraints = dict(getattr(unit, "constraints", {}) or {})
            if constraints.get("grouped_conv") and not constraints.get("depthwise"):
                scope_id = str(getattr(unit, "scope_id", ""))
                group_keep_map = dict(getattr(unit, "group_keep_map", {}) or group_keep_by_scope.get(scope_id, {}))
                if not group_keep_map:
                    raise RuntimeError(
                        "grouped_conv_bundle_required:"
                        f"{stable_id}:{constraints.get('grouped_module_path', getattr(unit, 'root_module_path', ''))}"
                    )
            closure_members = [
                member.to_dict() if hasattr(member, "to_dict") else dict(vars(member))
                for member in getattr(unit, "members", [])
            ]
            root_key = (str(getattr(unit, "root_module_path", "")), str(getattr(unit, "root_axis", "out")))
            merged: dict[tuple[str, str], dict[str, Any]] = {}
            for member in closure_members:
                key = (str(member.get("module_path", "")), str(member.get("axis", "")))
                if not all(key):
                    continue
                row = merged.setdefault(
                    key,
                    {
                        "indices": [],
                        "dependency_types": [],
                        "closure_index_map": {},
                    },
                )
                row["indices"] = sorted(set(row["indices"]) | {int(value) for value in member.get("indices", [])})
                dep = str(member.get("dependency_type", ""))
                if dep:
                    row["dependency_types"] = sorted(set(row["dependency_types"]) | {dep})
                index_map = {
                    int(key_): [int(value) for value in values]
                    for key_, values in dict(member.get("index_map", {}) or {}).items()
                }
                if index_map:
                    row["closure_index_map"].update(index_map)
            if root_key not in merged:
                merged[root_key] = {
                    "indices": list(getattr(unit, "root_indices", [])),
                    "dependency_types": ["root"],
                    "closure_index_map": {},
                }
            grouped_module = str(constraints.get("grouped_module_path") or "")
            for (module_path, axis), row in sorted(merged.items()):
                carries_group_map = bool(grouped_module and module_path == grouped_module and axis in {"in", "out", "channel"})
                aggregate = aggregated.setdefault(
                    (module_path, axis),
                    {
                        "indices": set(),
                        "source_atomic_unit_ids": set(),
                        "source_coupled_unit_ids": set(),
                        "scope_ids": set(),
                        "dependency_types": set(),
                        "closure_index_map": {},
                        "closure_members": [],
                        "group_keep_map": {},
                        "group_prune_map": {},
                        "protection_reasons": set(),
                        "constraints_by_atomic_unit": {},
                    },
                )
                aggregate["indices"].update(int(value) for value in row["indices"])
                aggregate["source_atomic_unit_ids"].add(stable_id)
                aggregate["source_coupled_unit_ids"].update(str(value) for value in getattr(unit, "source_coupled_unit_ids", []) or [])
                aggregate["scope_ids"].add(str(getattr(unit, "scope_id", "")))
                aggregate["dependency_types"].update(str(value) for value in row["dependency_types"])
                aggregate["closure_members"].extend(closure_members)
                aggregate["constraints_by_atomic_unit"][stable_id] = constraints
                reason = str(getattr(unit, "protection_reason", ""))
                if reason:
                    aggregate["protection_reasons"].add(reason)
                for root_index, local_indices in dict(row["closure_index_map"]).items():
                    root = int(root_index)
                    local = sorted({int(value) for value in local_indices})
                    existing = aggregate["closure_index_map"].get(root)
                    if existing is not None and existing != local:
                        raise RuntimeError(
                            "conflicting_search_closure_index_map:"
                            f"{module_path}:{axis}:root={root}:old={existing}:new={local}"
                        )
                    aggregate["closure_index_map"][root] = local
                if carries_group_map:
                    scope_id = str(getattr(unit, "scope_id", ""))
                    keep_map = dict(getattr(unit, "group_keep_map", {}) or group_keep_by_scope.get(scope_id, {}))
                    prune_map = dict(getattr(unit, "group_prune_map", {}) or group_prune_by_scope.get(scope_id, {}))
                    if aggregate["group_keep_map"] and aggregate["group_keep_map"] != keep_map:
                        raise RuntimeError(f"conflicting_group_keep_map:{module_path}:{axis}")
                    if aggregate["group_prune_map"] and aggregate["group_prune_map"] != prune_map:
                        raise RuntimeError(f"conflicting_group_prune_map:{module_path}:{axis}")
                    aggregate["group_keep_map"] = keep_map
                    aggregate["group_prune_map"] = prune_map
            channel_cost += int(getattr(unit, "channel_cost", max(len(getattr(unit, "root_indices", [])), 1)))
            parameter_cost += int(getattr(unit, "parameter_cost", 0))
        entries = []
        for ordering, ((module_path, axis), row) in enumerate(sorted(aggregated.items())):
            scope_ids = sorted(value for value in row["scope_ids"] if value)
            scope_id = scope_ids[0] if len(scope_ids) == 1 else f"scope_bundle_{canonical_json_hash(scope_ids)[:16]}"
            entries.append(
                SamplingPruningEntry(
                    request_id=f"search_bundle::{ordering:04d}",
                    scope_id=scope_id,
                    module_path=module_path,
                    axis=axis,
                    prune_indices=sorted(row["indices"]),
                    source_atomic_unit_ids=sorted(row["source_atomic_unit_ids"]),
                    group_keep_map=dict(row["group_keep_map"]),
                    group_prune_map=dict(row["group_prune_map"]),
                    protection_reason=";".join(sorted(row["protection_reasons"])),
                    metadata={
                        "source_coupled_unit_ids": sorted(row["source_coupled_unit_ids"]),
                        "closure_members": list(row["closure_members"]),
                        "dependency_types": sorted(row["dependency_types"]),
                        "closure_index_map": dict(sorted(row["closure_index_map"].items())),
                        "constraints_by_atomic_unit": dict(row["constraints_by_atomic_unit"]),
                    },
                )
            )
        return SamplingPruningRequest(
            entries=entries,
            selected_atomic_unit_ids=sorted(selected),
            requested_channel_cost=channel_cost,
            requested_parameter_cost=parameter_cost,
            one_shot=True,
            selector="two_stage_joint_search",
        )

    def materialize_from_request(
        self,
        model: Any,
        request: Any,
        *,
        alignment_config: Any | None = None,
        grouped_config: Any | None = None,
        example_inputs: Any | None = None,
        fixed_output_contracts: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        plan = self.build_plan_fn(model, request)
        legal_plan = self.legalize_plan_fn(
            model,
            plan,
            alignment_config=alignment_config,
            grouped_config=grouped_config,
        )
        materialized = self.materialize_fn(model, legal_plan, in_place=False, build_snapshot=True)
        snapshot = materialized.snapshot if getattr(materialized, "snapshot", None) is not None else self.snapshot_fn(materialized.model)
        hashes = self.hash_fn(snapshot)
        validation = self.validate_fn(
            materialized.model,
            expected_snapshot=snapshot,
            grouped_config=grouped_config,
            fixed_output_contracts=fixed_output_contracts,
            example_inputs=example_inputs,
        )
        return {
            "model": materialized.model,
            "plan": legal_plan,
            "ledger": materialized.ledger,
            "snapshot": snapshot,
            "hashes": hashes,
            "validation": validation,
        }
