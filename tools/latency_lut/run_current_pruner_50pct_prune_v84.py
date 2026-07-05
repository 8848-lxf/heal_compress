from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.audit_grouped_conv_keep_distribution_v84 import audit_records, summarize
from tools.latency_lut.fixed_width_boundary_registry_v84 import protected_prefixes_for_fixed_width_boundaries




def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tests/test_general_pruner.py",
        "--checkpoint",
        args.checkpoint,
        "--model-config",
        args.model_config,
        "--heal-root",
        args.heal_root,
        "--prune-ratio",
        "0.5",
        "--importance-mode",
        "l2_norm",
        "--selection-mode",
        "root_node_local_unit_ratio",
        "--group-conv-selection-mode",
        "independent_group_topk",
        "--group-conv-align",
        "8",
        "--align",
        "8",
        "--protect-residual-add",
        "false",
        "--protect-neck-and-heads",
        "false",
        "--device",
        args.device,
        "--skip-forward-check",
        "--allow-save-on-forward-fail",
        "--disable-pre-prune-group-normalization",
        "--output-dir",
        str(out / "work"),
    ]
    for prefix in protected_prefixes_for_fixed_width_boundaries():
        cmd.extend(["--extra-protected-prefix", prefix])
    result: dict[str, Any] = {
        "command": cmd,
        "physical_prune_success": False,
        "forward_or_export_success": False,
        "failure_reasons": [],
    }
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
        result["returncode"] = proc.returncode
        result["log_tail"] = proc.stdout[-12000:]
        if proc.returncode != 0:
            result["failure_reasons"].append("current_pruner_subprocess_failed")
    except Exception as exc:
        result["returncode"] = -1
        result["log_tail"] = str(exc)
        result["failure_reasons"].append(type(exc).__name__)
    work_dirs = sorted(out.glob("work*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    work = work_dirs[-1] if work_dirs else out / "work"
    for name in [
        "prune_replay.json",
        "coupled_channel_units.json",
        "root_node_local_domains.json",
        "domain_selection_summary.json",
        "grouped_conv_selection_report.json",
        "pruning_summary.json",
        "structure_changes.json",
    ]:
        src = work / name
        if src.is_file():
            shutil.copy2(src, out / name)
    summary = _load_json(out / "pruning_summary.json", {})
    result["physical_prune_success"] = bool(summary.get("num_pruned_groups", 0) > 0 and not summary.get("pre_prune_group_alignment_ops"))
    if summary.get("pre_prune_group_alignment_ops"):
        result["failure_reasons"].append("pre_prune_normalization_detected")
    reports = _load_json(out / "grouped_conv_selection_report.json", [])
    rows = audit_records("current_pruner", reports if isinstance(reports, list) else [], require_group_keep_map=True)
    _write_json(out / "current_grouped_conv_audit.json", {"records": rows, "summary": summarize(rows)})
    _write_json(out / "error_report.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", default="outputs/latency_lut/tp_vs_current_pruner_50pct_v84/current_pruner")
    args = parser.parse_args(argv)
    print(json.dumps(_run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
