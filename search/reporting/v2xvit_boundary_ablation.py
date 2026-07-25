"""Replay V2X-ViT Greedy traces and define single-domain restore ablations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from search.greedy.conservative_joint import select_budget_winner


PRECISION_ORDER = ("FP32", "FP16", "INT8")


def stable_mapping_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(sorted(value.items())),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_space_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    domains = {
        str(row["domain_id"]): {
            "domain_type": str(row["domain_type"]),
            "original_width": int(row["original_width"]),
            "legal_widths": tuple(int(value) for value in row["legal_widths"]),
        }
        for row in payload["pruning_domains"]
    }
    groups = {
        str(row["group_id"]): {
            "allowed_precisions": tuple(str(value).upper() for value in row["allowed_precisions"]),
            "protected": bool(row["protected"]),
        }
        for row in payload["quantization_groups"]
    }
    variable_groups = {
        key: value
        for key, value in groups.items()
        if not value["protected"] and len(set(value["allowed_precisions"])) > 1
    }
    if not domains or not variable_groups:
        raise RuntimeError("v2xvit_boundary_space_contract_empty")
    return {"domains": domains, "precision_groups": variable_groups}


def initial_genotype(contract: Mapping[str, Any]) -> dict[str, Any]:
    widths = {
        key: int(row["original_width"])
        for key, row in contract["domains"].items()
    }
    precision = {}
    for key, row in contract["precision_groups"].items():
        allowed = tuple(row["allowed_precisions"])
        precision[key] = next(value for value in PRECISION_ORDER if value in allowed)
    return {
        "pruning_genes": {},
        "pruning_width_genes": widths,
        "precision_genes": precision,
        "meta": {"created_by": "v2xvit_boundary_trace_replay"},
    }


def adjacent_successor(
    current: Mapping[str, Any],
    *,
    action_type: str,
    gene_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    widths = dict(current["pruning_width_genes"])
    precision = dict(current["precision_genes"])
    if action_type == "domain_width":
        domain = contract["domains"].get(gene_id)
        if domain is None:
            raise RuntimeError(f"trace_replay_unknown_domain:{gene_id}")
        legal = tuple(domain["legal_widths"])
        value = int(widths[gene_id])
        position = legal.index(value)
        if position <= 0:
            raise RuntimeError(f"trace_replay_domain_at_minimum:{gene_id}:{value}")
        widths[gene_id] = int(legal[position - 1])
    elif action_type == "precision":
        group = contract["precision_groups"].get(gene_id)
        if group is None:
            raise RuntimeError(f"trace_replay_unknown_precision_group:{gene_id}")
        ordered = [value for value in PRECISION_ORDER if value in group["allowed_precisions"]]
        value = str(precision[gene_id]).upper()
        position = ordered.index(value)
        if position + 1 >= len(ordered):
            raise RuntimeError(f"trace_replay_precision_at_minimum:{gene_id}:{value}")
        precision[gene_id] = ordered[position + 1]
    else:
        raise RuntimeError(f"trace_replay_unknown_action:{action_type}")
    return {
        "pruning_genes": {},
        "pruning_width_genes": widths,
        "precision_genes": precision,
        "meta": {"created_by": "v2xvit_boundary_trace_replay"},
    }


def _convert_row(raw: Mapping[str, str]) -> dict[str, Any]:
    floats = {
        "BOPS_after",
        "BOPS_before",
        "current_retention",
        "R_bops_vs_fp32",
        "cumulative_total_taylor",
        "cumulative_structural_taylor",
        "cumulative_weight_quant_taylor",
        "cumulative_activation_quant_taylor",
        "R_parameter_retention",
        "parameter_count",
        "mixed_weight_size_bytes",
        "mixed_weight_retention",
    }
    integers = {"step", "global_rank", "structural_repair_count", "precision_repair_count"}
    row: dict[str, Any] = dict(raw)
    for key in floats:
        row[key] = float(raw[key])
    for key in integers:
        row[key] = int(raw[key])
    row["selected"] = str(raw["selected"]).lower() == "true"
    return row


def replay_trace(
    trace_path: str | Path,
    contract: Mapping[str, Any],
    targets: Iterable[float],
    *,
    tolerance: float = 0.005,
) -> dict[str, Any]:
    normalized_targets = tuple(sorted({float(value) for value in targets}, reverse=True))
    bands: dict[float, list[dict[str, Any]]] = {target: [] for target in normalized_targets}
    current = initial_genotype(contract)
    selected_steps = 0
    rows_checked = 0
    current_step = 0
    selected_in_step = 0
    selected_successor: dict[str, Any] | None = None

    def finish_step() -> None:
        nonlocal current, selected_steps, selected_in_step, selected_successor
        if current_step == 0:
            return
        if selected_in_step != 1 or selected_successor is None:
            raise RuntimeError(
                f"trace_replay_selected_count_invalid:step={current_step}:count={selected_in_step}"
            )
        current = selected_successor
        selected_steps += 1
        selected_in_step = 0
        selected_successor = None

    with Path(trace_path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = _convert_row(raw)
            step = int(row["step"])
            if current_step and step != current_step:
                finish_step()
            current_step = step
            successor = adjacent_successor(
                current,
                action_type=str(row["action_type"]),
                gene_id=str(row["domain_layer"]),
                contract=contract,
            )
            physical_hash = stable_mapping_hash(successor["pruning_width_genes"])
            precision_hash = stable_mapping_hash(successor["precision_genes"])
            if physical_hash != row["physical_hash"]:
                raise RuntimeError(
                    f"trace_replay_physical_hash_mismatch:step={step}:{row['domain_layer']}"
                )
            if precision_hash != row["precision_hash"]:
                raise RuntimeError(
                    f"trace_replay_precision_hash_mismatch:step={step}:{row['domain_layer']}"
                )
            rows_checked += 1
            candidate = {**row, "genotype": successor}
            for target in normalized_targets:
                if abs(float(row["current_retention"]) - target) <= float(tolerance):
                    bands[target].append(candidate)
            if row["selected"]:
                selected_in_step += 1
                selected_successor = successor
    finish_step()

    winners: dict[str, dict[str, Any]] = {}
    for target in normalized_targets:
        rows = bands[target]
        if not rows:
            raise RuntimeError(f"trace_replay_budget_band_empty:{target}")
        winner = select_budget_winner(rows, target=target)
        key = f"{target:.3f}"
        winners[key] = {
            "candidate_hash": winner["candidate_hash"],
            "genotype": winner["genotype"],
            "metrics": {name: value for name, value in winner.items() if name != "genotype"},
            "band_candidate_count": len(rows),
        }
    return {
        "schema_version": "v2xvit-greedy-boundary-trace-replay-v1",
        "trace": str(Path(trace_path).resolve()),
        "targets": list(normalized_targets),
        "absolute_tolerance": float(tolerance),
        "rows_checked": rows_checked,
        "selected_steps_checked": selected_steps,
        "all_mapping_hashes_verified": True,
        "winners": winners,
    }


def make_restore_ablations(
    winner005: Mapping[str, Any],
    winner010: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    base = dict(winner005["genotype"])
    base_widths = dict(base["pruning_width_genes"])
    width010 = dict(winner010["genotype"]["pruning_width_genes"])
    domains = contract["domains"]

    def candidate(name: str, replacements: Mapping[str, int]) -> dict[str, Any]:
        widths = dict(base_widths)
        widths.update({str(key): int(value) for key, value in replacements.items()})
        changed = sorted(key for key in widths if widths[key] != base_widths[key])
        if not changed:
            raise RuntimeError(f"restore_ablation_no_change:{name}")
        return {
            "genotype": {
                "pruning_genes": {},
                "pruning_width_genes": widths,
                "precision_genes": dict(base["precision_genes"]),
                "meta": {
                    "created_by": "v2xvit_boundary_restore_ablation",
                    "ablation": name,
                    "base_candidate_hash": winner005["candidate_hash"],
                },
            },
            "base_candidate_hash": winner005["candidate_hash"],
            "changed_domains": changed,
        }

    ffn = {
        key: int(domains[key]["original_width"])
        for key in base_widths
        if key.startswith("ffn_hidden::")
    }
    shrinker = {
        key: int(domains[key]["original_width"])
        for key in base_widths
        if key.startswith("shrinker_m1.")
    }
    attention = {
        key: int(width010[key])
        for key in base_widths
        if key.startswith("attention_dh::")
    }
    stage2 = {
        key: int(domains[key]["original_width"])
        for key in base_widths
        if key.startswith("backbone_m1.blocks.2.")
    }
    result = {
        "A1_restore_ffn": candidate("A1_restore_ffn", ffn),
        "A2_restore_shrinker": candidate("A2_restore_shrinker", shrinker),
        "A3_restore_attention_to_010": candidate("A3_restore_attention_to_010", attention),
        "A4_restore_stage2_backbone": candidate("A4_restore_stage2_backbone", stage2),
    }
    expected_prefixes = {
        "A1_restore_ffn": ("ffn_hidden::",),
        "A2_restore_shrinker": ("shrinker_m1.",),
        "A3_restore_attention_to_010": ("attention_dh::",),
        "A4_restore_stage2_backbone": ("backbone_m1.blocks.2.",),
    }
    for name, row in result.items():
        prefixes = expected_prefixes[name]
        if any(not key.startswith(prefixes) for key in row["changed_domains"]):
            raise RuntimeError(f"restore_ablation_scope_violation:{name}")
    return result
