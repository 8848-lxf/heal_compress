from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def audit_param_accounting_record(
    *,
    candidate_id: str,
    target_keep_ratio: float,
    param_keep_ratio: float,
    num_pruned_units: int,
    changed_layers: list[dict[str, Any]],
    pre_prune_ops: list[dict[str, Any]],
) -> dict[str, Any]:
    reductions = expansions = nochange = 0
    expanded_layers = []
    for layer in changed_layers:
        b = layer.get("baseline") or {}
        c = layer.get("candidate") or {}
        bsum = int(b.get("C_in") or 0) + int(b.get("C_out") or 0)
        csum = int(c.get("C_in") or 0) + int(c.get("C_out") or 0)
        if csum > bsum:
            expansions += 1
            expanded_layers.append(
                {
                    "layer_name": layer.get("layer_name"),
                    "baseline_C_in": b.get("C_in"),
                    "baseline_C_out": b.get("C_out"),
                    "candidate_C_in": c.get("C_in"),
                    "candidate_C_out": c.get("C_out"),
                    "reason": "pre_prune_group_alignment_normalization" if pre_prune_ops else "unknown",
                }
            )
        elif csum < bsum:
            reductions += 1
        else:
            nochange += 1
    pre_norm = bool(pre_prune_ops)
    invalid = []
    if float(param_keep_ratio) > 1.0:
        invalid.append("param_keep_ratio_gt_1")
    if int(num_pruned_units) == 0:
        invalid.append("num_pruned_units_zero")
    if expansions:
        invalid.append("channel_expansion_detected")
    if pre_norm and expansions:
        invalid.append("pre_prune_normalization_expansion")
    return {
        "candidate_id": candidate_id,
        "target_keep_ratio": float(target_keep_ratio),
        "param_keep_ratio": float(param_keep_ratio),
        "param_keep_ratio_gt_1": float(param_keep_ratio) > 1.0,
        "num_reduction_layers": reductions,
        "num_expansion_layers": expansions,
        "num_nochange_layers": nochange,
        "expanded_layers": expanded_layers,
        "pre_prune_normalization_detected": pre_norm,
        "normalization_increases_params": pre_norm and bool(expansions),
        "valid_pruned_candidate": not invalid,
        "invalid_reasons": invalid,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for cdir in sorted(Path(args.export_dir).iterdir()):
        if not cdir.is_dir():
            continue
        cand = _load_json(cdir / "candidate.json", {})
        structure = _load_json(cdir / "structure_audit.json", {})
        export = _load_json(cdir / "export_pruned_model_report.json", {})
        summary = export.get("existing_general_pruner_summary") or {}
        rows.append(
            audit_param_accounting_record(
                candidate_id=cdir.name,
                target_keep_ratio=float((cand.get("pruning") or {}).get("target_keep_ratio") or 1.0),
                param_keep_ratio=float(structure.get("param_keep_ratio") or 0.0),
                num_pruned_units=int(summary.get("num_pruned_groups") or 0),
                changed_layers=structure.get("changed_conv_layers") or [],
                pre_prune_ops=summary.get("pre_prune_group_alignment_ops") or [],
            )
        )
    summary = {
        "num_candidates": len(rows),
        "num_param_keep_ratio_gt_1": sum(1 for r in rows if r["param_keep_ratio_gt_1"]),
        "num_with_channel_expansion": sum(1 for r in rows if r["num_expansion_layers"] > 0),
        "num_with_pre_prune_normalization": sum(1 for r in rows if r["pre_prune_normalization_detected"]),
        "num_valid_pruned_candidates": sum(1 for r in rows if r["valid_pruned_candidate"]),
        "normalization_audit_pass": all(r["valid_pruned_candidate"] for r in rows),
    }
    out = {"candidates": rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Prune Normalization Param Accounting v8.3\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v81")
    p.add_argument("--output-json", default="outputs/latency_lut/prune_normalization_param_accounting_v83.json")
    p.add_argument("--output-md", default="outputs/latency_lut/prune_normalization_param_accounting_v83.md")
    args = p.parse_args(argv)
    print(json.dumps(audit(args)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
