"""Global joint-Taylor anchor planning and deployment contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from ..hashing import canonical_json_hash
from ..stage1.conditional_repair import (
    conditional_dense_floor_repair,
    conditional_grouped_floor_repair,
)


ANCHOR_PRECISION_VARIANTS = (
    "strict_fp32",
    "strict_fp16",
    "maximal_legal_int8",
)


@dataclass(frozen=True)
class AnchorPruningUnit:
    unit_id: str
    prune_domain_id: str
    importance: float
    root_module: str
    parameter_cost: int = 0
    protected: bool = False
    dependent_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorStructure:
    anchor_id: str
    requested_prune_rate: float
    realized_prune_rate: float
    original_params: int
    candidate_params: int
    group_mask: dict[str, int]
    mask_hash: str
    domain_prune_rates: dict[str, float]
    infeasible_under_domain_cap: bool
    repair_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorSweepPlan:
    structures: tuple[AnchorStructure, ...]
    global_ranking: tuple[dict[str, Any], ...]
    maximum_realized_prune_rate: float
    per_domain_max_prune_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "structures": [row.to_dict() for row in self.structures],
            "global_ranking": [dict(row) for row in self.global_ranking],
            "maximum_realized_prune_rate": self.maximum_realized_prune_rate,
            "per_domain_max_prune_rate": self.per_domain_max_prune_rate,
        }


def plan_legal_width_anchor_structures(
    *,
    inventory: Any,
    decoder: Any,
    ranking_rows: Sequence[Mapping[str, Any]],
    ranking_mode: str,
    requested_prune_rates: Sequence[float],
    original_params: int,
    parameter_count_fn: Callable[[dict[str, int]], int],
) -> AnchorSweepPlan:
    """Build one nested anchor path using adjacent legal width actions."""

    if int(original_params) <= 0:
        raise ValueError("anchor_original_params_must_be_positive")
    score_key = (
        "first_order_score"
        if ranking_mode == "prune_only_first_order"
        else "second_order_score"
    )
    scores = {
        str(row["atomic_unit_id"]): float(row[score_key])
        for row in ranking_rows
    }
    if set(scores) != set(inventory.unit_ids):
        raise RuntimeError("legal_width_anchor_ranking_inventory_mismatch")
    current = {
        domain.domain_id: len(domain.legal_keep_widths) - 1
        for domain in inventory.domains
    }
    path: list[dict[str, Any]] = []

    def record(width_genes: Mapping[str, int]) -> dict[str, Any]:
        decoded = decoder.decode(width_genes)
        candidate_params = int(parameter_count_fn(decoded.group_mask))
        if not 0 <= candidate_params <= int(original_params):
            raise RuntimeError("legal_width_anchor_parameter_count_invalid")
        rates = {}
        for domain in inventory.domains:
            pruned = sum(
                int(decoded.group_mask[unit_id]) == 0 for unit_id in domain.unit_ids
            )
            rates[domain.domain_id] = float(pruned / max(domain.original_width, 1))
        return {
            "width_genes": dict(width_genes),
            "decoded": decoded,
            "candidate_params": candidate_params,
            "realized_prune_rate": 1.0 - candidate_params / float(original_params),
            "domain_prune_rates": rates,
        }

    path.append(record(current))
    while any(index > 0 for index in current.values()):
        before = path[-1]["decoded"]
        moves = []
        before_pruned = set(before.pruned_unit_ids)
        for domain_id, index in sorted(current.items()):
            if index <= 0:
                continue
            candidate_genes = dict(current)
            candidate_genes[domain_id] = index - 1
            candidate = record(candidate_genes)
            newly_pruned = set(candidate["decoded"].pruned_unit_ids) - before_pruned
            if not newly_pruned:
                raise RuntimeError(
                    f"legal_width_anchor_nonprogressing_move:{domain_id}:{index}"
                )
            moves.append(
                (
                    sum(scores[unit_id] for unit_id in newly_pruned),
                    domain_id,
                    candidate,
                )
            )
        if not moves:
            break
        _cost, selected_domain, selected = min(
            moves, key=lambda row: (float(row[0]), str(row[1]))
        )
        current = dict(selected["width_genes"])
        selected["transition_domain"] = selected_domain
        path.append(selected)
    maximum = max(row["realized_prune_rate"] for row in path)
    mode_token = (
        "first" if ranking_mode == "prune_only_first_order" else "second"
    )
    structures = []
    for requested in requested_prune_rates:
        target = float(requested)
        selected = min(
            path,
            key=lambda row: (
                round(abs(float(row["realized_prune_rate"]) - target), 12),
                float(row["realized_prune_rate"]) > target,
                row["decoded"].structure_hash,
            ),
        )
        decoded = selected["decoded"]
        structures.append(
            AnchorStructure(
                anchor_id=f"{mode_token}_prune_{int(round(target * 10000)):04d}",
                requested_prune_rate=target,
                realized_prune_rate=float(selected["realized_prune_rate"]),
                original_params=int(original_params),
                candidate_params=int(selected["candidate_params"]),
                group_mask=dict(decoded.group_mask),
                mask_hash=str(decoded.structure_hash),
                domain_prune_rates=dict(selected["domain_prune_rates"]),
                infeasible_under_domain_cap=target > maximum + 1.0e-12,
                repair_metadata={
                    **decoded.to_dict(),
                    "ranking_mode": ranking_mode,
                    "repair_invoked": False,
                    "conditional_channel_refinement": False,
                    "transition_domain": selected.get("transition_domain", ""),
                },
            )
        )
    global_rows = [
        {
            **dict(row),
            "ranking_mode": ranking_mode,
            "importance": float(row[score_key]),
        }
        for row in ranking_rows
    ]
    global_rows.sort(
        key=lambda row: (float(row["importance"]), str(row["atomic_unit_id"]))
    )
    for rank, row in enumerate(global_rows, start=1):
        row["global_rank"] = rank
    return AnchorSweepPlan(
        structures=tuple(structures),
        global_ranking=tuple(global_rows),
        maximum_realized_prune_rate=float(maximum),
        per_domain_max_prune_rate=max(
            (
                max(row["domain_prune_rates"].values(), default=0.0)
                for row in path
            ),
            default=0.0,
        ),
    )


def global_group_ranking(units: Sequence[AnchorPruningUnit]) -> list[dict[str, Any]]:
    rows = [
        {
            "rank": 0,
            "unit_id": str(unit.unit_id),
            "prune_domain_id": str(unit.prune_domain_id),
            "root_module": str(unit.root_module),
            "dependent_modules": list(unit.dependent_modules),
            "importance": float(unit.importance),
            "total_importance": float(unit.importance),
            "parameter_cost_diagnostic": int(unit.parameter_cost),
            "normalization": "none",
            "protected": bool(unit.protected),
        }
        for unit in units
        if not unit.protected and math.isfinite(float(unit.importance))
    ]
    rows.sort(key=lambda row: (float(row["importance"]), str(row["unit_id"])))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _domain_rates(mask: Mapping[str, int], domains: Mapping[str, Sequence[str]]) -> dict[str, float]:
    return {
        domain_id: (
            sum(1 for unit_id in unit_ids if int(mask.get(str(unit_id), 1)) == 0)
            / max(len(unit_ids), 1)
        )
        for domain_id, unit_ids in sorted(domains.items())
    }


def plan_anchor_structures(
    units: Sequence[AnchorPruningUnit],
    *,
    requested_prune_rates: Sequence[float],
    original_params: int,
    parameter_count_fn: Callable[[dict[str, int]], int],
    dense_alignment_by_domain: Mapping[str, int],
    minimum_width_by_domain: Mapping[str, int],
    per_domain_max_prune_rate: float = 0.8,
    grouped_domain_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> AnchorSweepPlan:
    if int(original_params) <= 0:
        raise ValueError("anchor_original_params_must_be_positive")
    if not 0.0 <= float(per_domain_max_prune_rate) <= 0.8:
        raise ValueError("anchor_domain_prune_cap_must_be_between_zero_and_point_eight")
    by_id = {str(unit.unit_id): unit for unit in units}
    domains: dict[str, list[str]] = {}
    for unit in units:
        if unit.protected:
            continue
        domains.setdefault(str(unit.prune_domain_id), []).append(str(unit.unit_id))
    ranking = global_group_ranking(units)
    raw_mask = {unit_id: 1 for unit_id in sorted(by_id)}
    raw_selected_by_domain = {domain_id: 0 for domain_id in domains}
    conditional_costs = {str(unit.unit_id): float(unit.importance) for unit in units}
    candidates: dict[str, dict[str, Any]] = {}
    grouped_specs = dict(grouped_domain_specs or {})

    def record_candidate() -> None:
        repaired = dict(raw_mask)
        for domain_id, unit_ids in sorted(domains.items()):
            if domain_id in grouped_specs:
                continue
            result = conditional_dense_floor_repair(
                {unit_id: raw_mask[unit_id] for unit_id in unit_ids},
                conditional_costs=conditional_costs,
                alignment=int(dense_alignment_by_domain.get(domain_id, 4)),
                minimum_width=int(minimum_width_by_domain.get(domain_id, 1)),
            )
            if result.status != "ok":
                return
            repaired.update(result.repaired_mask)
        group_keep_maps: dict[str, Any] = {}
        group_prune_maps: dict[str, Any] = {}
        for domain_id, spec in sorted(grouped_specs.items()):
            unit_ids = domains.get(domain_id, [])
            result = conditional_grouped_floor_repair(
                {unit_id: raw_mask[unit_id] for unit_id in unit_ids},
                physical_groups=spec["physical_groups"],
                local_indices=spec["local_indices"],
                conditional_costs=conditional_costs,
                allowed_channels_per_group=spec["allowed_channels_per_group"],
            )
            if result.status != "ok":
                return
            repaired.update(result.repaired_mask)
            group_keep_maps[domain_id] = result.group_keep_map
            group_prune_maps[domain_id] = result.group_prune_map
        rates = _domain_rates(repaired, domains)
        if any(value > float(per_domain_max_prune_rate) + 1.0e-12 for value in rates.values()):
            return
        key = canonical_json_hash(dict(sorted(repaired.items())))
        if key in candidates:
            return
        candidate_params = int(parameter_count_fn(repaired))
        if not 0 <= candidate_params <= int(original_params):
            raise RuntimeError("anchor_parameter_count_callback_invalid")
        realized = 1.0 - float(candidate_params) / float(original_params)
        candidates.setdefault(
            key,
            {
                "mask": repaired,
                "mask_hash": key,
                "candidate_params": candidate_params,
                "realized_prune_rate": realized,
                "domain_prune_rates": rates,
                "repair_metadata": {
                    "repair_mode": "global_anchor_conditional_joint_taylor",
                    "group_keep_map_by_scope": group_keep_maps,
                    "group_prune_map_by_scope": group_prune_maps,
                },
            },
        )

    record_candidate()
    for ranking_row in ranking:
        unit_id = str(ranking_row["unit_id"])
        domain_id = str(ranking_row["prune_domain_id"])
        domain_width = len(domains[domain_id])
        cap_count = int(math.floor(domain_width * float(per_domain_max_prune_rate)))
        if raw_selected_by_domain[domain_id] >= cap_count:
            continue
        raw_mask[unit_id] = 0
        raw_selected_by_domain[domain_id] += 1
        record_candidate()

    # The unit-wise global walk can consume a domain cap before every physical
    # group has crossed the same legal-width boundary. Explicitly enumerate the
    # maximum legal state so the planner's reported reachability is not an
    # artifact of importance-order skew.
    for unit_id in raw_mask:
        raw_mask[unit_id] = 1
    for domain_id, unit_ids in sorted(domains.items()):
        if domain_id in grouped_specs:
            spec = grouped_specs[domain_id]
            physical_groups = {
                int(group): [str(unit_id) for unit_id in rows]
                for group, rows in spec["physical_groups"].items()
            }
            group_widths = {len(rows) for rows in physical_groups.values()}
            if len(group_widths) != 1:
                raise RuntimeError(
                    f"anchor_grouped_physical_width_mismatch:{domain_id}"
                )
            group_width = group_widths.pop()
            legal_keeps = [
                int(width)
                for width in spec["allowed_channels_per_group"]
                if 0 < int(width) <= group_width
                and (group_width - int(width)) / group_width
                <= float(per_domain_max_prune_rate) + 1.0e-12
            ]
            target_keep = min(legal_keeps) if legal_keeps else group_width
            for rows in physical_groups.values():
                ordered = sorted(
                    rows,
                    key=lambda unit_id: (conditional_costs[unit_id], unit_id),
                )
                for unit_id in ordered[: group_width - target_keep]:
                    raw_mask[unit_id] = 0
        else:
            width = len(unit_ids)
            cap_count = int(
                math.floor(width * float(per_domain_max_prune_rate))
            )
            maximum = min(
                cap_count,
                width - int(minimum_width_by_domain.get(domain_id, 1)),
            )
            alignment = max(1, int(dense_alignment_by_domain.get(domain_id, 4)))
            legal_count = alignment * (maximum // alignment)
            ordered = sorted(
                unit_ids,
                key=lambda unit_id: (conditional_costs[unit_id], unit_id),
            )
            for unit_id in ordered[:legal_count]:
                raw_mask[unit_id] = 0
    record_candidate()
    if not candidates:
        raise RuntimeError("anchor_no_legal_structures")
    choices = list(candidates.values())
    maximum = max(float(row["realized_prune_rate"]) for row in choices)
    structures: list[AnchorStructure] = []
    for requested in requested_prune_rates:
        target = float(requested)
        selected = min(
            choices,
            key=lambda row: (
                round(abs(float(row["realized_prune_rate"]) - target), 12),
                float(row["realized_prune_rate"]) > target,
                str(row["mask_hash"]),
            ),
        )
        structures.append(
            AnchorStructure(
                anchor_id=f"prune_{int(round(target * 1000)):04d}",
                requested_prune_rate=target,
                realized_prune_rate=float(selected["realized_prune_rate"]),
                original_params=int(original_params),
                candidate_params=int(selected["candidate_params"]),
                group_mask=dict(selected["mask"]),
                mask_hash=str(selected["mask_hash"]),
                domain_prune_rates=dict(selected["domain_prune_rates"]),
                infeasible_under_domain_cap=bool(target > maximum + 1.0e-12),
                repair_metadata=dict(selected["repair_metadata"]),
            )
        )
    return AnchorSweepPlan(
        structures=tuple(structures),
        global_ranking=tuple(ranking),
        maximum_realized_prune_rate=maximum,
        per_domain_max_prune_rate=float(per_domain_max_prune_rate),
    )


def build_engine_matrix_requests(
    structures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for structure in structures:
        anchor_id = str(structure["anchor_id"])
        physical_hash = str(structure["physical_hash"])
        if not physical_hash:
            raise RuntimeError(f"anchor_physical_hash_missing:{anchor_id}")
        for precision in ANCHOR_PRECISION_VARIANTS:
            requests.append(
                {
                    "anchor_id": anchor_id,
                    "physical_hash": physical_hash,
                    "precision_variant": precision,
                    "matrix_id": f"{anchor_id}::{precision}",
                }
            )
    return requests


def assert_formal_latency_isolation(
    *,
    active_process_commands: Sequence[str],
    selected_gpu_uuid: str,
    gpu_processes: Sequence[Mapping[str, Any]],
) -> None:
    forbidden_tokens = (
        "candidate_worker",
        "stage2_process_pool",
        "evaluation_worker",
        "calibration_worker",
        "trtexec",
        "search.cli",
    )
    offenders = [
        command
        for command in active_process_commands
        if any(token in str(command) for token in forbidden_tokens)
    ]
    if offenders:
        raise RuntimeError(f"formal_latency_parallel_worker_active:{len(offenders)}")
    selected_processes = [
        row for row in gpu_processes if str(row.get("gpu_uuid", "")) == str(selected_gpu_uuid)
    ]
    if selected_processes:
        raise RuntimeError(f"formal_latency_selected_gpu_busy:{len(selected_processes)}")


def propose_boundary_bisections(
    evaluated_rows: Sequence[Mapping[str, Any]],
    *,
    max_absolute_map_drop: float = 0.1,
    max_rounds: int = 3,
) -> tuple[float, ...]:
    if int(max_rounds) <= 0:
        return ()
    valid = [row for row in evaluated_rows if bool(row.get("valid_for_tau", False))]
    safe = [row for row in valid if float(row.get("delta_mAP", math.inf)) <= max_absolute_map_drop]
    unsafe = [row for row in valid if float(row.get("delta_mAP", -math.inf)) > max_absolute_map_drop]
    if not safe or not unsafe:
        return ()
    lower = max(float(row["realized_prune_rate"]) for row in safe)
    upper_candidates = [float(row["realized_prune_rate"]) for row in unsafe if float(row["realized_prune_rate"]) > lower]
    if not upper_candidates:
        return ()
    upper = min(upper_candidates)
    if upper <= lower:
        return ()
    return ((lower + upper) / 2.0,)
