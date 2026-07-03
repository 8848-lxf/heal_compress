from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def audit_ratio_record(
    *,
    candidate_id: str,
    target_keep_ratio: float,
    export_cli_prune_ratio: float | None,
    general_pruner_args_prune_ratio: float | None,
    domain_requested_keep_ratio: float | None,
    domain_requested_prune_ratio: float | None,
    actual_param_keep_ratio: float | None,
) -> dict[str, Any]:
    expected_prune = round(1.0 - float(target_keep_ratio), 12)
    inversion = _close(export_cli_prune_ratio, target_keep_ratio) or _close(general_pruner_args_prune_ratio, target_keep_ratio)
    propagated = (
        _close(export_cli_prune_ratio, expected_prune)
        and _close(general_pruner_args_prune_ratio, expected_prune)
        and _close(domain_requested_keep_ratio, target_keep_ratio)
        and _close(domain_requested_prune_ratio, expected_prune)
    )
    actual_keep = float(actual_param_keep_ratio or 0.0)
    return {
        "candidate_id": candidate_id,
        "candidate_target_keep_ratio": float(target_keep_ratio),
        "candidate_target_prune_ratio_expected": expected_prune,
        "export_cli_prune_ratio": None if export_cli_prune_ratio is None else float(export_cli_prune_ratio),
        "general_pruner_args_prune_ratio": None if general_pruner_args_prune_ratio is None else float(general_pruner_args_prune_ratio),
        "domain_selection_requested_keep_ratio": None if domain_requested_keep_ratio is None else float(domain_requested_keep_ratio),
        "domain_selection_requested_prune_ratio": None if domain_requested_prune_ratio is None else float(domain_requested_prune_ratio),
        "actual_global_param_keep_ratio": actual_keep,
        "actual_global_param_prune_ratio": 1.0 - actual_keep if actual_keep else None,
        "keep_ratio_correctly_propagated": propagated,
        "prune_ratio_inversion_detected": inversion,
        "evidence": f"target_keep={target_keep_ratio}, expected_prune={expected_prune}, cli_prune={export_cli_prune_ratio}, general_pruner_prune={general_pruner_args_prune_ratio}",
    }


def monotonic_keep_ratio_trend(actual_by_target: dict[float, list[float]]) -> bool:
    ordered = sorted((float(k), sum(v) / len(v)) for k, v in actual_by_target.items() if v)
    # As target_keep increases, actual_keep must not decrease.
    for (_, lower_actual), (_, higher_actual) in zip(ordered, ordered[1:]):
        if higher_actual + 1e-9 < lower_actual:
            return False
    return True


def audit(args: argparse.Namespace) -> dict[str, Any]:
    candidate_payload = _load_json(args.candidates, {"candidates": []})
    candidate_by_id = {c["candidate_id"]: c for c in candidate_payload.get("candidates", [])}
    rows: list[dict[str, Any]] = []
    actual_by_target: dict[float, list[float]] = {}
    for cdir in sorted(Path(args.export_dir).iterdir()):
        if not cdir.is_dir():
            continue
        cid = cdir.name
        candidate = _load_json(cdir / "candidate.json", candidate_by_id.get(cid, {}))
        pruning = candidate.get("pruning") or {}
        target_keep = float(pruning.get("target_keep_ratio", 1.0 - float(pruning.get("target_prune_ratio", 0.0))))
        export_report = _load_json(cdir / "export_pruned_model_report.json", {})
        summary = export_report.get("existing_general_pruner_summary") or {}
        selection = _load_json(cdir / "domain_selection_summary.json", {})
        first_domain = (selection.get("domains") or [{}])[0]
        structure = _load_json(cdir / "pruned_model_export_report.json", {})
        actual_keep = structure.get("param_keep_ratio")
        row = audit_ratio_record(
            candidate_id=cid,
            target_keep_ratio=target_keep,
            export_cli_prune_ratio=summary.get("target_prune_ratio"),
            general_pruner_args_prune_ratio=summary.get("selection_summary", {}).get("target_prune_ratio", summary.get("target_prune_ratio")),
            domain_requested_keep_ratio=first_domain.get("requested_keep_ratio"),
            domain_requested_prune_ratio=first_domain.get("requested_prune_ratio"),
            actual_param_keep_ratio=actual_keep,
        )
        rows.append(row)
        if actual_keep:
            actual_by_target.setdefault(target_keep, []).append(float(actual_keep))
    summary = {
        "num_candidates": len(rows),
        "target_keep_ratio_correctly_propagated": bool(rows) and all(r["keep_ratio_correctly_propagated"] for r in rows),
        "prune_ratio_inversion_detected": any(r["prune_ratio_inversion_detected"] for r in rows),
        "actual_param_keep_by_target_keep": {str(k): v for k, v in sorted(actual_by_target.items())},
        "num_unique_actual_param_keep_ratio": len({round(v, 6) for vals in actual_by_target.values() for v in vals}),
        "monotonic_keep_ratio_trend": monotonic_keep_ratio_trend(actual_by_target),
    }
    out = {"candidates": rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Pruning Ratio Semantics v8.1\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v81.json")
    parser.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v81")
    parser.add_argument("--output-json", default="outputs/latency_lut/pruning_ratio_semantics_v81.json")
    parser.add_argument("--output-md", default="outputs/latency_lut/pruning_ratio_semantics_v81.md")
    args = parser.parse_args(argv)
    out = audit(args)
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
