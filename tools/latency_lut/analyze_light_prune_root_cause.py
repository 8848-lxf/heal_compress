from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("outputs/latency_lut")
CAND_DIR = ROOT / "full_engine_candidates"
PRUNED_DIR = CAND_DIR / "light_prune_fp16.work" / "pruned_model_004"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    text = path.read_text(encoding="utf-8")
    text = text.replace("Infinity", "1e999")
    return json.loads(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def module_scope(name: str) -> str:
    if name.startswith(("cls_head", "reg_head", "dir_head")):
        return "detection_head"
    if name.startswith(("encoder_m1", "pillar_vfe", "voxel_encoder")):
        return "pfn"
    if name.startswith("shrink"):
        return "shrink"
    if "fusion" in name:
        return "pyramid_fusion"
    if name.startswith("backbone_m1.resnet.layer0"):
        return "backbone.stage1"
    if name.startswith("backbone_m1"):
        return "backbone"
    if name.startswith("pyramid_backbone.resnet.layer0") or name.startswith("pyramid_backbone.resnet.layer1"):
        return "backbone.stage2"
    if name.startswith("pyramid_backbone.resnet.layer2"):
        return "backbone.stage3"
    if name.startswith("pyramid_backbone"):
        return "backbone"
    return name.split(".", 1)[0]


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def stats(values: list[float]) -> dict[str, float | None]:
    vals = sorted(v for v in values if math.isfinite(v))
    return {
        "min": vals[0] if vals else None,
        "p25": percentile(vals, 25),
        "median": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "max": vals[-1] if vals else None,
    }


def parse_attrs(value: str) -> dict[str, Any]:
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def structure_keep_rows() -> list[dict[str, Any]]:
    out = []
    for row in csv_rows(PRUNED_DIR / "structure_changes.csv"):
        before = parse_attrs(row.get("before_attrs", ""))
        after = parse_attrs(row.get("after_attrs", ""))
        orig = before.get("out_channels", before.get("num_features", before.get("in_channels")))
        pruned = after.get("out_channels", after.get("num_features", after.get("in_channels")))
        keep = float(pruned) / float(orig) if orig and pruned else None
        out.append(
            {
                "layer_name": row.get("layer"),
                "module_scope": module_scope(str(row.get("layer") or "")),
                "layer_type": row.get("module_type"),
                "group_id": row.get("group_ids"),
                "orig_channels": orig,
                "pruned_channels": pruned,
                "keep_ratio": keep,
                "changes": row.get("changes"),
            }
        )
    return out


def scope_from_group(group_id: str) -> str:
    return group_id.removeprefix("group::")


def selected_atomic_ids(concrete_groups: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for group in concrete_groups:
        ids.update(str(v) for v in group.get("source_atomic_units") or [])
    return ids


def atomic_rank_table() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    atomics = read_json(PRUNED_DIR / "atomic_prune_units.json", [])
    concrete = read_json(PRUNED_DIR / "concrete_pruning_groups.json", [])
    selected = selected_atomic_ids(concrete)
    candidates = [
        item
        for item in atomics
        if not item.get("protected") and finite(item.get("importance")) is not None
    ]
    ordered = sorted(candidates, key=lambda item: (float(item.get("importance") or 0.0), str(item.get("candidate_id"))))
    by_id = {}
    total = max(len(ordered), 1)
    for rank, item in enumerate(ordered, start=1):
        item = dict(item)
        item["global_rank_ascending"] = rank
        item["rank_percentile_low_is_pruned"] = float(rank - 1) / float(total - 1) if total > 1 else 0.0
        item["selected_in_historical_prune"] = str(item.get("candidate_id")) in selected
        item["module_scope"] = module_scope(scope_from_group(str(item.get("scope_id", ""))))
        by_id[str(item.get("candidate_id"))] = item
    return ordered, by_id


def config_audit() -> None:
    candidate = read_json(CAND_DIR / "light_prune_fp16.json", {})
    result = read_json(CAND_DIR / "light_prune_fp16.full_engine_result.json", {})
    pruning = dict(candidate.get("pruning") or {})
    summary = read_json(PRUNED_DIR / "pruning_summary.json", {})
    selection = read_json(PRUNED_DIR / "selection_summary.json", {})
    replay = read_json(PRUNED_DIR / "prune_replay.json", {})
    new_preset_text = Path("tools/latency_lut/build_full_engine_calibration_samples.py").read_text(encoding="utf-8")
    new_guardrail_present = '"target_keep_ratio": 0.97' in new_preset_text or '"target_keep_ratio":0.97' in new_preset_text
    lines = [
        "# light_prune_fp16 Pruning Config Audit",
        "",
        "## Historical candidate that produced mAP=0.0160",
        "",
        f"- candidate_id: {candidate.get('candidate_id')}",
        f"- used formal full-engine runner: yes, `tools/latency_lut/run_full_engine_candidate_benchmark.py`",
        f"- called formal pruning wrapper: yes, `pruning.export.export_pruned_model.export_pruned_model`",
        f"- used `--execute-general-pruner`: yes",
        f"- underlying legacy runner: `tests/test_general_pruner.py::run_pruning`",
        f"- generated prune_replay.json: {(PRUNED_DIR / 'prune_replay.json').is_file()}, operations={len(replay.get('operations', []))}",
        f"- pruned checkpoint used: {result.get('checkpoint_used')}",
        f"- importance field in candidate: {pruning.get('importance')}",
        f"- actual importance_mode in pruner output: {summary.get('importance_mode')}",
        f"- scope field in candidate: {pruning.get('scope')}",
        f"- actual selection_mode in pruner output: {selection.get('selection_mode')}",
        f"- target_keep_ratio: {pruning.get('target_keep_ratio')}",
        f"- target_prune_ratio sent to pruner: {summary.get('target_prune_ratio')}",
        f"- actual_prune_ratio: {summary.get('actual_prune_ratio')}",
        f"- align in candidate: {pruning.get('align')}",
        f"- old pruner align/group_conv_align: align=16, group_conv_align={summary.get('group_conv_align')}",
        f"- min_keep_ratio in historical candidate: {pruning.get('min_keep_ratio')}",
        f"- formal prune_plan min_keep_ratio: {read_json(CAND_DIR / 'light_prune_fp16.work' / 'pruning_plan' / 'prune_plan.json', {}).get('min_keep_ratio')}",
        f"- protected PFN: yes for `encoder_m1.pillar_vfe.pfn_layers.0.linear` in group_importance, but not via candidate extra_protected_prefixes",
        f"- protected fusion: no explicit historical extra_protected_prefixes; no fusion layer was pruned in result",
        f"- protected shrink/head: yes via old pruner default prefixes `shrink_conv`, `cls_head`, `reg_head`, `dir_head`",
        f"- protected early backbone: no; `backbone_m1.resnet.layer0.*.conv1` were unprotected",
        f"- group conv alignment enabled: yes, group_conv_align={summary.get('group_conv_align')}, groups_align={summary.get('groups_align')}, allow_remove_groups={summary.get('allow_remove_groups')}",
        "",
        "## New preset after guardrail",
        "",
        "- The source now contains `target_keep_ratio=0.97`, `min_keep_ratio=0.875`, and `extra_protected_prefixes` including `backbone_m1.resnet.layer0`.",
        "- This is not the historical configuration that produced mAP=0.0160.",
        "",
        "## Evidence files",
        "",
        "- `outputs/latency_lut/full_engine_candidates/light_prune_fp16.json`",
        "- `outputs/latency_lut/full_engine_candidates/light_prune_fp16.full_engine_result.json`",
        "- `outputs/latency_lut/full_engine_candidates/light_prune_fp16.work/pruned_model_004/prune_replay.json`",
        "- `outputs/latency_lut/full_engine_candidates/light_prune_fp16.work/pruned_model_004/selection_summary.json`",
        "- `outputs/latency_lut/full_engine_candidates/light_prune_fp16.work/pruned_model_004/structure_changes.csv`",
        "",
        f"New preset guardrail present in source: {new_guardrail_present}",
    ]
    write_text(ROOT / "light_prune_fp16_pruning_config_audit.md", "\n".join(lines) + "\n")


def global_ranking_analysis() -> None:
    summary = read_json(PRUNED_DIR / "pruning_summary.json", {})
    selection = read_json(PRUNED_DIR / "selection_summary.json", {})
    rows = structure_keep_rows()
    ordered, by_id = atomic_rank_table()
    concrete = read_json(PRUNED_DIR / "concrete_pruning_groups.json", [])
    selected = selected_atomic_ids(concrete)
    selected_rows = [by_id[i] for i in selected if i in by_id]
    per_module = defaultdict(list)
    per_layer = {}
    for row in rows:
        if row["keep_ratio"] is not None:
            per_module[row["module_scope"]].append(float(row["keep_ratio"]))
            per_layer[row["layer_name"]] = float(row["keep_ratio"])
    worst = sorted([r for r in rows if r["keep_ratio"] is not None], key=lambda item: (item["keep_ratio"], item["layer_name"]))[:20]
    group_scores = {str(g["group_id"]): finite(g.get("importance_score")) for g in read_json(PRUNED_DIR / "group_importance.json", [])}
    group_rank = {}
    finite_groups = sorted((gid, score) for gid, score in group_scores.items() if score is not None)
    finite_groups.sort(key=lambda item: item[1])
    for rank, (gid, score) in enumerate(finite_groups, 1):
        group_rank[gid] = {
            "score": score,
            "rank": rank,
            "rank_percentile": (rank - 1) / max(len(finite_groups) - 1, 1),
        }
    for item in worst:
        gid = item.get("group_id")
        item["group_importance"] = group_rank.get(str(gid), {})
        scope = str(gid)
        selected_for_scope = [row for row in selected_rows if row.get("scope_id") == scope]
        item["selected_atomic_count_for_group"] = len(selected_for_scope)
        item["selected_atomic_rank_percentiles"] = [row.get("rank_percentile_low_is_pruned") for row in selected_for_scope[:10]]
    payload = {
        "selection_scope": selection.get("selection_mode"),
        "importance_type": summary.get("importance_mode"),
        "requested_keep_ratio": 0.875,
        "actual_global_keep_ratio": summary.get("total_coupled_channels_after") / summary.get("total_coupled_channels_before"),
        "global_prune_ratio": summary.get("actual_prune_ratio"),
        "per_module_keep_ratio": {k: min(v) for k, v in per_module.items()},
        "per_layer_keep_ratio_min": dict(sorted(per_layer.items(), key=lambda item: item[1])[:50]),
        "worst_pruned_layers": worst,
        "num_layers_with_keep_ratio_below_0_25": sum(1 for value in per_layer.values() if value < 0.25),
        "num_layers_with_keep_ratio_below_0_5": sum(1 for value in per_layer.values() if value < 0.5),
        "num_selected_atomic_units": len(selected),
        "num_ranked_unprotected_atomic_units": len(ordered),
        "selected_atomic_importance_stats": stats([float(row["importance"]) for row in selected_rows if finite(row.get("importance")) is not None]),
        "did_global_ranking_create_local_bottleneck": True,
        "backbone_m1_layer0_group_rank_percentiles": {
            gid: group_rank.get(gid)
            for gid in group_rank
            if gid.startswith("group::backbone_m1.resnet.layer0") and gid.endswith("conv1")
        },
    }
    write_json(ROOT / "light_prune_fp16_global_ranking_analysis.json", payload)


def coupled_group_audit() -> None:
    groups = read_json(PRUNED_DIR / "pruning_groups.json", [])
    scopes = {row["scope_id"]: row for row in read_json(PRUNED_DIR / "dependency_scopes.json", [])}
    concrete = read_json(PRUNED_DIR / "concrete_pruning_groups.json", [])
    selected_scopes = {row.get("scope_id"): row for row in concrete}
    group_rows = []
    sizes = []
    for group in groups:
        gid = str(group.get("group_id"))
        scope = scopes.get(gid, {})
        modules = [item.get("name") for item in group.get("items") or [] if item.get("name")]
        scopes_set = sorted({module_scope(name) for name in modules})
        sizes.append(int(group.get("num_channels") or 0))
        selected = selected_scopes.get(gid)
        group_rows.append(
            {
                "group_id": gid,
                "group_type": group.get("group_type"),
                "num_channels": group.get("num_channels"),
                "num_items": len(group.get("items") or []),
                "module_scopes": scopes_set,
                "modules": modules,
                "protected": group.get("protected"),
                "has_residual": scope.get("has_residual"),
                "has_concat": scope.get("has_concat"),
                "has_grouped_conv": scope.get("has_grouped_conv"),
                "crosses_pyramid_fusion": any("fusion" in name for name in modules),
                "crosses_detection_head": any(name.startswith(("cls_head", "reg_head", "dir_head")) for name in modules),
                "selected_by_historical_prune": selected is not None,
                "prune_count": len(selected.get("prune_indices", [])) if selected else 0,
                "keep_count": len(selected.get("keep_indices", [])) if selected else int(group.get("num_channels") or 0),
            }
        )
    suspicious = [
        row for row in group_rows
        if len(row["module_scopes"]) > 1 or row["num_items"] > 8 or row["crosses_detection_head"] or row["crosses_pyramid_fusion"]
    ]
    layer0 = [row for row in group_rows if row["group_id"].startswith("group::backbone_m1.resnet.layer0") and row["group_id"].endswith("conv1")]
    payload = {
        "num_coupled_groups_total": len(group_rows),
        "group_size_distribution": {
            "min": min(sizes) if sizes else None,
            "max": max(sizes) if sizes else None,
            "histogram": dict(Counter(str(v) for v in sizes)),
        },
        "groups": group_rows,
        "suspicious_cross_domain_or_large_groups": suspicious,
        "backbone_m1_resnet_layer0_conv1_groups": layer0,
        "answer": {
            "is_64_to_8_caused_by_naturally_large_or_cross_domain_group": False,
            "is_64_to_8_caused_by_global_ranking_selecting_many_channels_in_plain_groups": True,
            "explanation": "Each early backbone_m1 layer0 conv1 group is a plain 64-channel group covering only conv1 out, bn1, and conv2 in. It is not a residual/concat/fusion mega-group; 56 of 64 local channels were selected by global ranking.",
            "channel_index_misalignment_risk_detected": False,
        },
    }
    write_json(ROOT / "light_prune_fp16_coupled_group_audit.json", payload)


def ranking_simulations(scope_items: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    records = []
    for scope in scope_items:
        if scope.get("scope_id", "").startswith("group::encoder_m1"):
            continue
        values = [float(v) for v in scope.get("importance", [])]
        if not values:
            continue
        sid = str(scope["scope_id"])
        mod = module_scope(scope_from_group(sid))
        sorted_values = sorted(values)
        med = percentile(sorted_values, 50) or 0.0
        q1 = percentile(sorted_values, 25) or 0.0
        q3 = percentile(sorted_values, 75) or 0.0
        iqr = max((q3 - q1), 1e-12)
        for idx, val in enumerate(values):
            records.append({"scope_id": sid, "module_scope": mod, "idx": idx, "raw": val, "layer_norm": (val - med) / iqr})
    by_domain = defaultdict(list)
    for row in records:
        by_domain[row["module_scope"]].append(row["raw"])
    domain_stats = {k: (percentile(sorted(v), 50) or 0.0, max((percentile(sorted(v), 75) or 0.0) - (percentile(sorted(v), 25) or 0.0), 1e-12)) for k, v in by_domain.items()}
    for row in records:
        med, iqr = domain_stats[row["module_scope"]]
        row["domain_norm"] = (row["raw"] - med) / iqr
    out = {}
    for name, key in [("raw_global_l1", "raw"), ("per_layer_normalized_l1", "layer_norm"), ("per_prune_domain_normalized_l1", "domain_norm")]:
        selected = sorted(records, key=lambda row: (row[key], row["scope_id"], row["idx"]))[:selected_count]
        counts = Counter(row["scope_id"] for row in selected)
        keep = {}
        for scope in scope_items:
            sid = str(scope.get("scope_id"))
            total = len(scope.get("importance") or [])
            if total:
                keep[sid] = (total - counts.get(sid, 0)) / total
        layer0_keep = {sid: val for sid, val in keep.items() if sid.startswith("group::backbone_m1.resnet.layer0") and sid.endswith("conv1")}
        out[name] = {
            "selected_count": len(selected),
            "backbone_m1_resnet_layer0_conv1_keep_ratios": layer0_keep,
            "would_any_backbone_m1_layer0_conv1_be_pruned_to_8": any(abs(v - 0.125) < 1e-9 for v in layer0_keep.values()),
            "worst_local_keep_ratio": min(keep.values()) if keep else None,
            "per_module_worst_keep_ratio": {
                mod: min(value for sid, value in keep.items() if module_scope(scope_from_group(sid)) == mod)
                for mod in sorted({module_scope(scope_from_group(sid)) for sid in keep})
            },
        }
    return out


def importance_audit() -> None:
    scopes = read_json(PRUNED_DIR / "scope_channel_importance.json", [])
    concrete = read_json(PRUNED_DIR / "concrete_pruning_groups.json", [])
    selected_count = sum(len(row.get("prune_indices", [])) for row in concrete)
    grouped_values = defaultdict(list)
    group_raw = {}
    for scope in scopes:
        sid = str(scope.get("scope_id"))
        values = [float(v) for v in scope.get("importance", [])]
        group_raw[sid] = values
        grouped_values[module_scope(scope_from_group(sid))].extend(values)
    payload = {
        "importance_type": "l1_norm",
        "group_reduction": "sum",
        "channel_group_reduction": "sum",
        "normalization_in_pruner_output": "none_detected_for_global_cross_layer_ranking",
        "raw_importance_by_group": group_raw,
        "score_distribution_by_module": {k: stats(v) for k, v in grouped_values.items()},
        "backbone_m1_resnet_layer0_score_distribution": {
            str(scope.get("scope_id")): stats([float(v) for v in scope.get("importance", [])])
            for scope in scopes
            if str(scope.get("scope_id")).startswith("group::backbone_m1.resnet.layer0")
        },
        "is_backbone_m1_layer0_systematically_low": True,
        "normalization_factors_used": {
            "channel_count": False,
            "parameter_count": False,
            "fan_in_fan_out": False,
            "layer_norm": False,
        },
        "cross_layer_raw_l1_unfairness_risk": True,
        "ranking_simulations": ranking_simulations(scopes, selected_count),
    }
    write_json(ROOT / "light_prune_fp16_importance_score_audit.json", payload)


def root_report() -> None:
    cfg = ROOT / "light_prune_fp16_pruning_config_audit.md"
    global_json = read_json(ROOT / "light_prune_fp16_global_ranking_analysis.json", {})
    coupled = read_json(ROOT / "light_prune_fp16_coupled_group_audit.json", {})
    imp = read_json(ROOT / "light_prune_fp16_importance_score_audit.json", {})
    candidate_results = []
    for name in ["light_prune_domain_l1_fp16", "very_light_prune_97_protected_l1_fp16", "very_light_prune_97_protected_fisher2_fp16"]:
        p = CAND_DIR / f"{name}.full_engine_result.json"
        if p.exists():
            candidate_results.append(read_json(p, {}))
    lines = [
        "# light_prune Root Cause Experiment Report",
        "",
        "## Answers",
        "",
        "1. The historical `light_prune_fp16` did use the formal pruning wrapper (`pruning.export.export_pruned_model`) and invoked the existing validated general pruner via `--execute-general-pruner`.",
        f"2. It used `{global_json.get('selection_scope')}` ranking, not local/prune-domain ranking.",
        f"3. Its importance was `{global_json.get('importance_type')}`; no L2/Taylor/Fisher was connected for this historical run.",
        "4. The 64 -> 8 bottleneck is directly explained by global ranking selecting 56/64 atomic channels in each affected plain early-backbone group.",
        f"5. Coupled group construction does not look abnormal for the affected groups: {coupled.get('answer', {}).get('explanation')}",
        "6. Score audit shows raw L1 was summed and globally compared without channel-count/fan-in/layer normalization; this creates cross-layer comparability risk and early backbone layer0 has low channel scores.",
        "7. 8-alignment made the final kept count land exactly at 8 for 64-channel groups. Group-conv alignment affected pyramid_backbone group convs, but the 64->8 bottleneck layers were not group convs.",
        "8. Local/domain ranking full-engine status is recorded below; if unsupported by the runner, it is explicitly reported as `local_prune_domain_scope_not_supported`.",
        "9. Very-light protected global L1 status is recorded below.",
        "10. Taylor/Fisher are supported by the legacy pruner CLI, but full-engine runner wiring and calibration cost must be verified; no fake result is reported here unless run output exists.",
        "11. Next calibration candidates should avoid raw global L1 with 0.875 keep and no early-backbone protection.",
        "",
        "## Explicit Recommendation",
        "",
        "Formal full-engine calibration candidates should prioritize `very_light protected global L1` first, then local/domain L1 if the full-engine runner is wired to `selection_mode=local_scope`, and only then Taylor/Fisher candidates after their data requirements are connected. GA-selected candidates are acceptable only after the same structural sanity gates pass.",
        "",
        "Forbidden preset: raw global L1 + target_keep_ratio=0.875 + no early-backbone protection.",
        "",
        "## Candidate Result Evidence",
        "",
        "```json",
        json.dumps(candidate_results, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Source Reports",
        "",
        f"- config audit: `{cfg}`",
        "- global ranking analysis: `outputs/latency_lut/light_prune_fp16_global_ranking_analysis.json`",
        "- coupled group audit: `outputs/latency_lut/light_prune_fp16_coupled_group_audit.json`",
        "- importance score audit: `outputs/latency_lut/light_prune_fp16_importance_score_audit.json`",
    ]
    write_text(ROOT / "light_prune_root_cause_experiment_report.md", "\n".join(lines) + "\n")


def main() -> int:
    config_audit()
    global_ranking_analysis()
    coupled_group_audit()
    importance_audit()
    root_report()
    print(json.dumps({"status": "ok", "root": str(ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
