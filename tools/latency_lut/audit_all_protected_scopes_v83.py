from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return [] if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _match_prefix(module: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        if module == prefix or module.startswith(prefix + ".") or prefix in module:
            return prefix
    return ""


def audit_scope_rows(
    *,
    candidate_id: str,
    scopes: list[dict[str, Any]],
    selection_domains: list[dict[str, Any]],
    protected_prefixes: list[str],
) -> list[dict[str, Any]]:
    by_domain = {str(d.get("root_node") or d.get("domain_id", "")): d for d in selection_domains}
    by_domain.update({str(d.get("domain_id", "")): d for d in selection_domains})
    rows = []
    for scope in scopes:
        root = str(scope.get("root_module") or scope.get("scope_id") or "")
        scope_id = str(scope.get("scope_id") or "")
        dom = by_domain.get(scope_id) or by_domain.get(scope_id.replace("group::", "root_node::group::")) or {}
        rows.append(
            {
                "candidate_id": candidate_id,
                "scope_id": scope_id,
                "root_module": root,
                "is_prunable": not bool(scope.get("protected")),
                "protected": bool(scope.get("protected")),
                "protected_reason": str(scope.get("protected_reason") or ""),
                "matched_protected_prefix": _match_prefix(root or scope_id, protected_prefixes),
                "num_units": int(scope.get("num_channels") or dom.get("num_units") or 0),
                "num_pruned_units": int(dom.get("actual_num_prune") or 0),
                "actual_keep_ratio": float(dom.get("actual_keep_ratio") if dom.get("actual_keep_ratio") is not None else 1.0),
            }
        )
    return rows


def _is_head(row: dict[str, Any]) -> bool:
    text = f"{row.get('scope_id','')} {row.get('root_module','')} {row.get('protected_reason','')}".lower()
    return any(k in text for k in ["cls_head", "reg_head", "dir_head", "det_head", "head"])


def summarize_protected_scopes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    protected = [r for r in rows if r["protected"]]
    reasons: dict[str, int] = {}
    prefixes = sorted({r["matched_protected_prefix"] for r in protected if r.get("matched_protected_prefix")})
    non_head = [r for r in protected if not _is_head(r)]
    head = [r for r in protected if _is_head(r)]
    for row in protected:
        reasons[row.get("protected_reason") or "unknown"] = reasons.get(row.get("protected_reason") or "unknown", 0) + 1
    dominant = max(reasons, key=reasons.get) if reasons else ""
    return {
        "num_scopes": len(rows),
        "num_protected_scopes": len(protected),
        "protected_reasons": reasons,
        "protected_prefixes": prefixes,
        "protected_non_head_scopes": non_head,
        "num_non_head_protected_scopes": len(non_head),
        "num_head_protected_scopes": len(head),
        "protection_explains_ratio_gap": bool(non_head),
        "dominant_protection_reason": dominant,
        "recommendation": "remove non-head default protected prefixes from pruning smoke unless structurally required; keep detection head output protection and structural residual coupling only.",
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for cdir in sorted(Path(args.export_dir).iterdir()):
        if not cdir.is_dir():
            continue
        cand = _load_json(cdir / "candidate.json", {})
        prefixes = list((cand.get("pruning") or {}).get("extra_protected_prefixes") or [])
        scopes = _load_json(cdir / "dependency_scopes.json", [])
        selection = _load_json(cdir / "domain_selection_summary.json", {})
        all_rows.extend(audit_scope_rows(candidate_id=cdir.name, scopes=scopes, selection_domains=selection.get("domains", []), protected_prefixes=prefixes))
    summary = summarize_protected_scopes(all_rows)
    out = {"scopes": all_rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Protected Scopes v8.3\n\n```json\n" + json.dumps({k: v for k, v in summary.items() if k != "protected_non_head_scopes"}, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v81")
    p.add_argument("--output-json", default="outputs/latency_lut/all_protected_scopes_v83.json")
    p.add_argument("--output-md", default="outputs/latency_lut/all_protected_scopes_v83.md")
    args = p.parse_args(argv)
    print(json.dumps(audit(args)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
