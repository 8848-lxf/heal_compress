"""Evidence-only matrices and deployment contract for Transformer d_h sweeps."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from search.model_families.transformer.dh_candidate_grid import dense_head_dimension_grid
from search.model_families.transformer.dh_alignment_audit import parse_engine_memory_audit


PROFILES = ("P32", "P16", "P8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def _precision_bops(structure_dir: Path, profile: str) -> dict[str, float]:
    costs = _read(structure_dir / "runtime_costs.json")
    inventory = _read(structure_dir / "inventory.json")
    roles = {
        str(row.get("module_path", "")): str(row.get("canonical_role", ""))
        for row in inventory.get("rows", ()) if str(row.get("module_path", ""))
    }
    transformer = {
        "q_projection", "k_projection", "v_projection", "output_projection",
        "fused_qkv_projection", "ffn1", "ffn2",
    }
    weighted = 0.0
    for row in costs.get("layers", ()):
        path = str(row["module_path"])
        role = roles.get(path, "")
        if profile == "P32" and role in transformer:
            bits = 32
        elif profile == "P8" and role in {"q_projection", "k_projection"}:
            bits = 8
        else:
            bits = 16
        weighted += float(row["macs"]) * bits * bits
    qk = float(costs.get("qk_macs", 0.0)) * 32 * 32
    av_bits = 32 if profile == "P32" else 16
    av = float(costs.get("av_macs", 0.0)) * av_bits * av_bits
    return {"weighted_bops": weighted, "qk_bops": qk, "av_bops": av, "total_bops": weighted + qk + av}


def _write_inventory_contracts(output_root: Path) -> None:
    """Consolidate model-specific inventories without cross-model defaults."""

    groups: dict[str, Any] = {}
    dependencies: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        short = model.removeprefix("lidar_")
        families = json.loads(
            (output_root / "inventory" / f"{short}_attention_families.json").read_text(encoding="utf-8")
        )
        groups[model] = families
        for family in families:
            dependencies.append(
                {
                    "model": model,
                    "attention_family": family["family_id"],
                    "module_paths": family["module_paths"],
                    "heads": family["heads"],
                    "original_d_h": family["original_d_h"],
                    "qkv_dependency": "Q/K/V share identical head-local positions",
                    "output_dependency": "W_O input columns follow V/head-local positions",
                    "scale_dependency": "scale=1/sqrt(target_d_h)",
                    "model_specific_dependency": family.get("shared_dependency", ""),
                }
            )
        source = output_root / "inventory" / f"{short}_unsupported_attention_family.csv"
        if source.is_file():
            with source.open(newline="", encoding="utf-8") as handle:
                unsupported.extend(dict(row) for row in csv.DictReader(handle))
        # Compare every physical projection named by the module inventory with
        # the accepted D0 ONNX origin map.  Conditional HGT type branches may
        # legitimately be absent from this fixed deployment graph, but that
        # state must be explicit rather than silently assigned a precision.
        inventory_path = output_root / "inventory" / f"{short}_attention_dh_inventory.csv"
        first_family = families[0]
        origin_path = (
            output_root / "structures" / model / str(first_family["family_id"])
            / f"dh_{int(first_family['original_d_h']):03d}" / "canonical_origin_map.json"
        )
        if inventory_path.is_file() and origin_path.is_file():
            origin = json.loads(origin_path.read_text(encoding="utf-8"))
            mapped = {str(row.get("module_path", "")) for row in origin.get("entries", ())}
            with inventory_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for role, column in (
                        ("q_projection", "q_projection"),
                        ("k_projection", "k_projection"),
                        ("v_projection", "v_projection"),
                        ("output_projection", "out_projection"),
                    ):
                        for physical_path in json.loads(row[column]):
                            canonical_path = str(physical_path)
                            canonical_path = canonical_path.replace(".to_qkv[Q]", ".q_proj")
                            canonical_path = canonical_path.replace(".to_qkv[K]", ".k_proj")
                            canonical_path = canonical_path.replace(".to_qkv[V]", ".v_proj")
                            canonical_path = canonical_path.replace(".to_out.0", ".out_proj")
                            if canonical_path not in mapped:
                                unsupported.append(
                                    {
                                        "model": model,
                                        "attention_family": row["attention_family"],
                                        "module_path": row["module_path"],
                                        "role": role,
                                        "physical_projection": physical_path,
                                        "canonical_projection": canonical_path,
                                        "reason": (
                                            "conditional_parameterized_branch_absent_from_accepted_onnx"
                                            if "_linears.1" in canonical_path
                                            else "missing_role_mapping"
                                        ),
                                    }
                                )
    _write_json(output_root / "inventory" / "attention_family_groups.json", groups)
    _write_json(output_root / "inventory" / "structural_dependency_map.json", dependencies)
    unsupported_path = output_root / "inventory" / "unsupported_attention_family.csv"
    unsupported_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in unsupported for key in row}) or ["model", "module_path", "reason"]
    with unsupported_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unsupported)


def _write_precision_evidence(output_root: Path) -> dict[str, int]:
    realized_rows: list[dict[str, Any]] = []
    qdq_rows: list[dict[str, Any]] = []
    for engine_dir in sorted(output_root.glob("engines/lidar_*/*/dh_*/*")):
        if not engine_dir.is_dir():
            continue
        try:
            model = engine_dir.parents[2].name
            family = engine_dir.parents[1].name
            d_h = int(engine_dir.parent.name.removeprefix("dh_"))
            profile = engine_dir.name
        except (IndexError, ValueError):
            continue
        requested = engine_dir / "requested_realized.csv"
        if requested.is_file():
            with requested.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    realized_rows.append({"model": model, "attention_family": family, "d_h": d_h, "profile": profile, **row})
        inventory = engine_dir / "qdq_inventory.csv"
        if inventory.is_file():
            with inventory.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    qdq_rows.append({"model": model, "attention_family": family, "d_h": d_h, "profile": profile, **row})
    _write_csv(output_root / "reports" / "requested_realized_precision.csv", realized_rows)
    conflicts = [row for row in realized_rows if str(row.get("conflict", "")).lower() in {"1", "true", "yes"}]
    _write_csv(output_root / "reports" / "precision_conflicts.csv", conflicts)
    _write_csv(output_root / "reports" / "qdq_inventory.csv", qdq_rows)
    return {"requested_realized_rows": len(realized_rows), "precision_conflicts": len(conflicts), "qdq_rows": len(qdq_rows)}


def collect(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        families_by_model[model] = json.loads(
            (output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json").read_text(encoding="utf-8")
        )
        for family in families_by_model[model]:
            family_id = str(family["family_id"])
            original = int(family["original_d_h"])
            for candidate in dense_head_dimension_grid(
                original,
                heads=int(family["heads"]),
                low_width_extension=original <= 16,
            ):
                structure_dir = output_root / "structures" / model / family_id / f"dh_{candidate.d_h:03d}"
                structure = _read(structure_dir / "structure_result.json")
                for profile in PROFILES:
                    engine_dir = output_root / "engines" / model / family_id / f"dh_{candidate.d_h:03d}" / profile
                    build = _read(engine_dir / "baseline_result.json")
                    memory = build.get("engine_memory", {}) if build else {}
                    build_log = engine_dir / "engine_build" / "engine_build.log"
                    if build and not memory and build_log.is_file():
                        memory = parse_engine_memory_audit(build_log)
                    alignment = _read(engine_dir / "engine_alignment_audit.json")
                    fixed = {name: _read(engine_dir / "evaluation" / name / "evaluation_acceptance.json") for name in ("smoke10", "fixed50", "fixed500")}
                    latency_rows = _read(output_root / "latency" / model / family_id / profile / "formal_latency.json")
                    latency = next((value for value in latency_rows if int(value.get("d_h", -1)) == candidate.d_h and not (value.get("baseline_replay") and int(value.get("replay_index", 0)) > 0)), {}) if isinstance(latency_rows, list) else {}
                    bops = _precision_bops(structure_dir, profile) if structure else {}
                    row = {
                        "model": model,
                        "attention_family": family_id,
                        "profile": profile,
                        **candidate.to_dict(),
                        "params": structure.get("physical_parameter_count"),
                        "parameter_reduction": structure.get("parameter_reduction"),
                        "weighted_macs": structure.get("weighted_macs"),
                        "qk_macs": structure.get("qk_macs"),
                        "av_macs": structure.get("av_macs"),
                        "BOPS": bops.get("total_bops"),
                        "QKV_BOPS": bops.get("weighted_bops"),
                        "QK_BOPS": bops.get("qk_bops"),
                        "AV_BOPS": bops.get("av_bops"),
                        "pytorch_legal": structure.get("physical_forward_finite", False),
                        "onnx_legal": structure.get("onnx_checker_passed", False),
                        "engine_build": build.get("status") == "ok",
                        "requested_realized": build.get("requested_realized_conflict_count") == 0 if build else False,
                        "QK_precision": "FP32",
                        "QK_accumulator": "FP32",
                        "qdq_count": build.get("qdq_count", 0),
                        "cast_count": build.get("cast_count"),
                        "reformat_count": build.get("reformat_count"),
                        "max_scratch_bytes": memory.get("max_scratch_bytes"),
                        "activation_memory_bytes": memory.get("activation_bytes"),
                        "weights_memory_bytes": memory.get("weights_bytes"),
                        "padding_status": alignment.get("padding_status", "not_yet_verified"),
                        "tensor_core_hint": alignment.get("tensor_core_hint"),
                        "fallback": alignment.get("fallback_hint"),
                        "tactic": ";".join(sorted({str(value.get("tactic", "")) for value in alignment.get("evidence", ()) if value.get("tactic")})),
                        "fusion": ";".join(sorted({str(value.get("layer_type", "")) for value in alignment.get("evidence", ()) if "fused" in str(value.get("name", "")).lower()})),
                        "smoke10": fixed["smoke10"].get("status"),
                        "fixed50": fixed["fixed50"].get("status"),
                        "fixed500": fixed["fixed500"].get("status"),
                        "fixed500_AP30": fixed["fixed500"].get("AP@0.3"),
                        "fixed500_AP50": fixed["fixed500"].get("AP@0.5"),
                        "fixed500_AP70": fixed["fixed500"].get("AP@0.7"),
                        "fixed500_mAP": fixed["fixed500"].get("mAP"),
                        "p50_ms": latency.get("p50_ms"),
                        "p90_ms": latency.get("p90_ms"),
                        "p95_ms": latency.get("p95_ms"),
                        "p99_ms": latency.get("p99_ms"),
                        "mean_ms": latency.get("mean_ms"),
                        "std_ms": latency.get("std_ms"),
                        "speedup": latency.get("speedup_profile"),
                        "speedup_neighbor": latency.get("speedup_neighbor"),
                        "latency_beneficial": latency.get("latency_beneficial"),
                        "status": "not_yet_verified",
                    }
                    rows.append(row)
    index = {(row["model"], row["attention_family"], row["profile"], row["d_h"]): row for row in rows}
    for row in rows:
        family = next(value for value in families_by_model[row["model"]] if value["family_id"] == row["attention_family"])
        d0 = int(family["original_d_h"])
        same_profile = index[(row["model"], row["attention_family"], row["profile"], d0)]
        p32_same_width = index[(row["model"], row["attention_family"], "P32", row["d_h"])]
        p32_d0 = index[(row["model"], row["attention_family"], "P32", d0)]
        values = (row.get("fixed500_mAP"), same_profile.get("fixed500_mAP"), p32_same_width.get("fixed500_mAP"), p32_d0.get("fixed500_mAP"))
        if all(value is not None for value in values):
            m, baseline, p32, p32_base = map(float, values)
            row["delta_mAP_structure"] = m - baseline
            row["delta_mAP_precision"] = m - p32
            row["structure_precision_interaction"] = m - p32 - baseline + p32_base
            if m - baseline < -0.010:
                row["accuracy_status"] = "UNSAFE"
            elif m - baseline < -0.003:
                row["accuracy_status"] = "BORDERLINE"
            elif abs(m - baseline) <= 0.003:
                row["accuracy_status"] = "SAFE"
            else:
                row["accuracy_status"] = "SAFE_IMPROVEMENT_WITHIN_FIXED500_NOISE_INTERPRETATION"
        else:
            row["delta_mAP_structure"] = None
            row["delta_mAP_precision"] = None
            row["structure_precision_interaction"] = None
            row["accuracy_status"] = "not_yet_verified"
        if row["engine_build"] and row["requested_realized"] and row["fixed500"] == "ok":
            row["status"] = row["accuracy_status"]
    return rows


def _entries(rows: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    return [
        {"family": row["attention_family"], "profile": row["profile"], "d_h": row["d_h"]}
        for row in rows
        if predicate(row)
    ]


def _alignment_evidence(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    build_exact = [
        row for row in selected
        if row["engine_build"] and row["requested_realized"]
        and row["padding_status"] in {"EXACT_NONALIGNED", "EXACT_ALIGNED", "INTERNAL_PADDED"}
        and not row["fallback"]
    ]
    safe = [row for row in build_exact if str(row["accuracy_status"]).startswith("SAFE")]
    beneficial = [row for row in safe if row["latency_beneficial"] is True]
    categories = {
        "odd": lambda value: value % 2 == 1,
        "multiple_of_2_only": lambda value: value % 2 == 0 and value % 4 != 0,
        "multiple_of_4_only": lambda value: value % 4 == 0 and value % 8 != 0,
    }
    robust_by_profile = {
        profile: {
            name: any(row["profile"] == profile and predicate(int(row["d_h"])) for row in beneficial)
            for name, predicate in categories.items()
        }
        for profile in PROFILES
    }
    robust_profiles = [profile for profile, evidence in robust_by_profile.items() if all(evidence.values())]
    return {
        "build_supported": len(build_exact) > 0,
        "build_supported_nonaligned": any(int(row["d_h"]) % 4 for row in build_exact),
        "accuracy_supported_nonaligned": any(int(row["d_h"]) % 4 for row in safe),
        "latency_supported_nonaligned": any(int(row["d_h"]) % 4 for row in beneficial),
        "robust_by_profile": robust_by_profile,
        "robustly_supported": len(robust_profiles) >= 2,
        "robust_profiles": robust_profiles,
        "multiple_of_8_required": False if any(int(row["d_h"]) % 8 and row in beneficial for row in safe) else None,
        "multiple_of_4_required": False if any(int(row["d_h"]) % 4 and row in beneficial for row in safe) else None,
        "arbitrary_integer_supported": (
            True
            if selected and all(
                row["engine_build"] and row["requested_realized"] and str(row["accuracy_status"]).startswith("SAFE")
                for row in selected
            )
            else None
        ),
    }


def _write_root_conclusion(output_root: Path, rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> None:
    families: dict[str, list[dict[str, Any]]] = {}
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        families[model] = json.loads((output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json").read_text(encoding="utf-8"))
    lines = [
        "# H800 Transformer Q/K/V unified d_h alignment conclusion",
        "",
        f"Generated: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}",
        "",
        "This document is evidence-gated. `not_yet_verified` means the corresponding real artifact or run is still absent.",
        "",
    ]
    answers: list[tuple[str, str]] = []
    original = "; ".join(
        f"{model}: " + ", ".join(f"{row['family_id']} H={row['heads']} d_h={row['original_d_h']}" for row in values)
        for model, values in families.items()
    )
    answers.append(("1. Original H and d_h", original))
    tested = _entries(rows, lambda row: row["pytorch_legal"])
    answers.append(("2. Physically tested d_h", json.dumps(tested, sort_keys=True) if tested else "not_yet_verified"))
    odd = _entries(rows, lambda row: row["pytorch_legal"] and int(row["d_h"]) % 2 == 1)
    answers.append(("3. Physically legal odd d_h", json.dumps(odd, sort_keys=True) if odd else "not_yet_verified"))
    for number, profile, label in ((4, "P32", "FP32"), (5, "P16", "F3-FP16"), (6, "P8", "SQ1-INT8")):
        built = _entries(rows, lambda row, p=profile: row["profile"] == p and row["engine_build"] and row["requested_realized"] and int(row["d_h"]) % 4 != 0)
        answers.append((f"{number}. Non-4-aligned {label} engine support", json.dumps(built, sort_keys=True) if built else "not_yet_verified"))
    padding = sorted({str(row["padding_status"]) for row in rows if row["engine_build"]})
    answers.append(("7. Padding", ", ".join(padding) if padding else "not_yet_verified"))
    tc = _entries(rows, lambda row: row["tensor_core_hint"] is True)
    answers.append(("8. Tensor Core evidence", f"{len(tc)} candidate rows have tactic-name Tensor Core evidence" if tc else "not_yet_verified"))
    realized_path = output_root / "reports" / "requested_realized_precision.csv"
    complete_mha = []
    if realized_path.is_file():
        with realized_path.open(newline="", encoding="utf-8") as handle:
            complete_mha = [row for row in csv.DictReader(handle) if any(token in (str(row.get("tactic", "")) + str(row.get("tensorrt_layer_name", ""))).lower() for token in ("fmha", "fused_mha", "multiheadattention"))]
    answers.append(("9. Complete fused MHA", f"observed rows={len(complete_mha)}" if complete_mha else "not observed in current exact EngineInspector evidence"))
    tactic_widths = len({(row["model"], row["attention_family"], row["profile"], row["tactic"]) for row in rows if row["tactic"]})
    answers.append(("10. Tactic transitions", f"{tactic_widths} distinct family/profile tactic signatures; see reports/dh_alignment_tactic_transitions.csv" if tactic_widths else "not_yet_verified"))
    fixed_count = sum(row["fixed500"] == "ok" for row in rows)
    answers.append(("11. fixed500 mAP", f"{fixed_count}/{len(rows)} candidate rows complete; see reports/dh_alignment_full_matrix.csv"))
    safe = _entries(rows, lambda row: str(row["accuracy_status"]).startswith("SAFE"))
    answers.append(("12. Accuracy versus width", f"safe rows={len(safe)}; complete matrix preserves raw AP and same-profile deltas" if safe else "not_yet_verified"))
    interactions = [float(row["structure_precision_interaction"]) for row in rows if row["profile"] == "P8" and row["structure_precision_interaction"] is not None]
    answers.append(("13. SQ1-structure interaction", f"range=[{min(interactions):.9f}, {max(interactions):.9f}]" if interactions else "not_yet_verified"))
    latency = [row for row in rows if row["p50_ms"] is not None]
    answers.append(("14. Formal p50", f"formal rows={len(latency)}; see reports/dh_alignment_latency_boundary.csv" if latency else "not_yet_verified (five-minute isolation gate not yet satisfied)"))
    neighbor = _entries(rows, lambda row: row["speedup_neighbor"] is not None and float(row["speedup_neighbor"]) > 1.0 and int(row["d_h"]) % 4 != 0)
    answers.append(("15. Nonaligned faster than aligned/neighbor", json.dumps(neighbor, sort_keys=True) if neighbor else "not_yet_verified"))
    per_model = {model: _alignment_evidence(rows, model) for model in families}
    answers.append(("16. Is 8-alignment unnecessary?", json.dumps({model: None if value["multiple_of_8_required"] is None else not value["multiple_of_8_required"] for model, value in per_model.items()}, sort_keys=True)))
    answers.append(("17. Is 4-alignment unnecessary?", json.dumps({model: None if value["multiple_of_4_required"] is None else not value["multiple_of_4_required"] for model, value in per_model.items()}, sort_keys=True)))
    answers.append(("18. Arbitrary integer deployable?", json.dumps({model: value["arbitrary_integer_supported"] for model, value in per_model.items()}, sort_keys=True)))
    answers.append(("19. Same conclusion across models?", "not_yet_verified" if not all(value["robustly_supported"] for value in per_model.values()) else "yes within H800/TRT10.9 evidence scope"))
    allowed = _entries(rows, lambda row: row["latency_beneficial"] is True and str(row["accuracy_status"]).startswith("SAFE"))
    answers.append(("20. Widths allowed for future search", json.dumps(allowed, sort_keys=True) if allowed else "not_yet_verified"))
    experimental = _entries(rows, lambda row: row["engine_build"] and row["requested_realized"] and str(row["accuracy_status"]).startswith("SAFE") and row["latency_beneficial"] is not True)
    answers.append(("21. Experimental widths", json.dumps(experimental, sort_keys=True) if experimental else "not_yet_verified"))
    rejected = _entries(rows, lambda row: row["status"] == "UNSAFE" or (row["engine_build"] and not row["requested_realized"]))
    answers.append(("22. Forbidden widths", json.dumps(rejected, sort_keys=True) if rejected else "none confirmed yet"))
    answers.append(("23. Portability", "All conclusions are restricted to H800 SM90 + TensorRT 10.9.0.34; portable=false."))
    answers.append(("24. Evidence for future CNN+Transformer joint rules", "yes" if all(value["robustly_supported"] for value in per_model.values()) else "not_yet_verified"))
    for heading, value in answers:
        lines.extend((f"## {heading}", "", value, ""))
    (output_root / "root_conclusion.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_reports(output_root: Path) -> dict[str, Any]:
    _write_inventory_contracts(output_root)
    precision_evidence = _write_precision_evidence(output_root)
    rows = collect(output_root)
    reports = output_root / "reports"
    _write_csv(reports / "dh_alignment_full_matrix.csv", rows)
    _write_csv(reports / "dh_alignment_build_support.csv", [{key: row.get(key) for key in ("model", "attention_family", "profile", "d_h", "engine_build", "requested_realized", "padding_status", "fallback", "tensor_core_hint", "tactic")} for row in rows])
    _write_csv(reports / "dh_alignment_accuracy_boundary.csv", [{key: row.get(key) for key in ("model", "attention_family", "profile", "d_h", "fixed500_AP30", "fixed500_AP50", "fixed500_AP70", "fixed500_mAP", "delta_mAP_structure", "accuracy_status")} for row in rows])
    _write_csv(reports / "dh_alignment_latency_boundary.csv", [{key: row.get(key) for key in ("model", "attention_family", "profile", "d_h", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "speedup", "speedup_neighbor", "latency_beneficial")} for row in rows])
    _write_csv(reports / "dh_alignment_tactic_transitions.csv", [{key: row.get(key) for key in ("model", "attention_family", "profile", "d_h", "alignment_class", "padding_status", "tactic", "fusion", "cast_count", "reformat_count")} for row in rows])
    _write_csv(reports / "dh_alignment_precision_interaction.csv", [{key: row.get(key) for key in ("model", "attention_family", "profile", "d_h", "delta_mAP_structure", "delta_mAP_precision", "structure_precision_interaction")} for row in rows])
    contract: dict[str, Any] = {
        "cobevt": {}, "v2xvit": {},
        "alignment_conclusion": {
            "multiple_of_16_required": None,
            "multiple_of_8_required": None,
            "multiple_of_4_required": None,
            "arbitrary_integer_supported": None,
        },
        "evidence_scope": {"gpu": "H800_SM90", "tensorrt": "10.9.0.34", "portable": False},
    }
    for model, key in (("lidar_cobevt", "cobevt"), ("lidar_v2xvit", "v2xvit")):
        for profile, profile_key in (("P32", "fp32"), ("P16", "f3_fp16"), ("P8", "sq1_int8")):
            selected = [row for row in rows if row["model"] == model and row["profile"] == profile]
            contract[key][profile_key] = {
                "build_supported": [{"family": row["attention_family"], "d_h": row["d_h"]} for row in selected if row["engine_build"] and row["requested_realized"]],
                "accuracy_safe": [{"family": row["attention_family"], "d_h": row["d_h"]} for row in selected if row["accuracy_status"].startswith("SAFE")],
                "latency_beneficial": [{"family": row["attention_family"], "d_h": row["d_h"]} for row in selected if row["latency_beneficial"] is True],
            }
    contract["alignment_conclusion"]["by_model"] = {
        model: _alignment_evidence(rows, model)
        for model in ("lidar_cobevt", "lidar_v2xvit")
    }
    _write_json(reports / "transformer_dh_alignment_contract.json", contract)
    _write_json(output_root / "transformer_dh_alignment_contract.json", contract)
    for name in (
        "dh_alignment_full_matrix.csv", "dh_alignment_build_support.csv",
        "dh_alignment_accuracy_boundary.csv", "dh_alignment_latency_boundary.csv",
        "dh_alignment_tactic_transitions.csv", "dh_alignment_precision_interaction.csv",
    ):
        (output_root / name).write_bytes((reports / name).read_bytes())
    for model, short in (("lidar_cobevt", "cobevt"), ("lidar_v2xvit", "v2xvit")):
        _write_csv(output_root / f"formal_latency_{short}.csv", [row for row in rows if row["model"] == model and row["p50_ms"] is not None])
    _write_root_conclusion(output_root, rows, contract)
    completed = sum(row["status"] != "not_yet_verified" for row in rows)
    summary = {"matrix_rows": len(rows), "completed_rows": completed, **precision_evidence, "contract": contract}
    _write_json(reports / "report_summary.json", summary)
    return summary


__all__ = ["collect", "write_reports"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = write_reports(Path(args.output_root).resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
