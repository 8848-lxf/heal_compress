from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.compare_tp_native_vs_current_pruner_v84 import summarize_comparison


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint": args.checkpoint,
        "model": "lidar_pyramid",
        "dummy_input_source": "HEALLiDARAdapter.build_synthetic_batch",
        "prune_ratio": 0.5,
        "keep_ratio": 0.5,
        "importance": "l2_norm",
        "ranking_scope": "local",
        "protect_policy": {
            "protect_head": False,
            "protect_neck": False,
            "protect_encoder": False,
            "protect_pfn_to_scatter_output_boundary": True,
            "protect_pointpillarscatter_fixed_channel_boundary": True,
            "protect_extra_prefixes": [],
        },
        "grouped_conv_policy_current_pruner": {
            "mode": "independent_group_topk",
            "groups_preserved": True,
            "same_keep_count_per_original_group": True,
            "group_keep_map_required": True,
            "keep_count_per_group_align": 8,
            "no_channel_expansion": True,
        },
        "tp_native_policy": {
            "use_raw_torch_pruning_depgraph": True,
            "importance": "l2_norm",
            "local_pruning": True,
            "custom_group_balance_constraint": False,
        },
    }


def _run(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"cmd": cmd, "returncode": proc.returncode, "log_tail": proc.stdout[-8000:]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/latency_lut/tp_vs_current_pruner_50pct_v84")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(argv)
    root = ROOT
    out = Path(args.output_dir)
    tp_dir = out / "tp_native"
    cur_dir = out / "current_pruner"
    rep_dir = out / "reports"
    for d in (tp_dir, cur_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    _write(out / "experiment_config.json", experiment_config(args))
    run_logs = []
    if not args.skip_run:
        run_logs.append(_run([sys.executable, "tools/latency_lut/run_tp_native_50pct_prune_v84.py", "--checkpoint", args.checkpoint, "--model-config", args.model_config, "--heal-root", args.heal_root, "--device", args.device, "--output-dir", str(tp_dir)], root))
        run_logs.append(_run([sys.executable, "tools/latency_lut/run_current_pruner_50pct_prune_v84.py", "--checkpoint", args.checkpoint, "--model-config", args.model_config, "--heal-root", args.heal_root, "--device", args.device, "--output-dir", str(cur_dir)], root))
    _write(rep_dir / "run_logs.json", run_logs)
    tp_audit = _load(tp_dir / "tp_grouped_conv_audit.json", {})
    cur_audit = _load(cur_dir / "current_grouped_conv_audit.json", {})
    report = summarize_comparison(
        list(tp_audit.get("records", [])),
        list(cur_audit.get("records", [])),
        _load(tp_dir / "error_report.json", {}),
        _load(cur_dir / "error_report.json", {}),
    )
    _write(rep_dir / "tp_vs_current_pruner_groupedconv_compare_v84.json", report)
    (rep_dir / "tp_vs_current_pruner_groupedconv_compare_v84.md").write_text(
        "# TP Native vs Current Pruner Grouped Conv v8.4\n\n```json\n"
        + json.dumps(report, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report["direct_comparison"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
