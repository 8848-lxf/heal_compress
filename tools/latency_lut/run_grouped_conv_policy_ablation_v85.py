from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.compare_grouped_conv_policy_ablation_v85 import build_policy_ablation_summary


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/latency_lut/grouped_conv_policy_ablation_v85")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    reports = out / "reports"
    for sub in ["baseline", "tp_native_50_l2", "current_strict_50_l2", "current_relaxed_total_align8_50_l2", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    config = {
        "prune_ratio": 0.5,
        "keep_ratio": 0.5,
        "importance": "l2_norm",
        "ranking_scope": "local",
        "protect_policy": {
            "protect_pfn_encoder_to_pointpillarscatter_boundary": True,
            "protect_head": False,
            "protect_neck": False,
            "protect_extra_prefixes": [],
        },
        "variants": ["baseline", "tp_native_50_l2", "current_strict_50_l2", "current_relaxed_total_align8_50_l2"],
    }
    latency = {
        "baseline": {"engine_build_success": False, "latency_p50_ms": None, "latency_mean_ms": None, "failure_reason": "not_run_in_policy_audit"},
        "tp_native_50_l2": {"engine_build_success": False, "latency_p50_ms": None, "speedup_vs_baseline": None, "failure_reason": "not_run_in_policy_audit"},
        "current_strict_50_l2": {"engine_build_success": False, "latency_p50_ms": None, "speedup_vs_baseline": None, "failure_reason": "not_run_in_policy_audit"},
        "current_relaxed_total_align8_50_l2": {"engine_build_success": False, "latency_p50_ms": None, "speedup_vs_baseline": None, "failure_reason": "not_run_in_policy_audit"},
    }
    audits: dict[str, list[dict[str, Any]]] = {}
    summary = build_policy_ablation_summary(audits=audits, latency=latency)
    _write(out / "experiment_config.json", config)
    _write(reports / "latency_smoke_v85.json", latency)
    (reports / "latency_smoke_v85.md").write_text("# Latency Smoke v8.5\n\nNot run in pruning-policy audit; no fake latency recorded.\n", encoding="utf-8")
    _write(reports / "grouped_conv_policy_ablation_v85.json", summary)
    (reports / "grouped_conv_policy_ablation_v85.md").write_text("# Grouped Conv Policy Ablation v8.5\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
