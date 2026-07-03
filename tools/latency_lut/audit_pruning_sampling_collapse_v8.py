from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.audit_pruning_sampling_collapse_v7 import audit as audit_v7, parse_args as parse_v7_args
from tools.latency_lut.audit_root_node_local_pruner_v8 import audit_export_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v8.json")
    parser.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v8")
    parser.add_argument("--dataset", default="outputs/latency_lut/nonexistent_v8_dataset.jsonl")
    parser.add_argument("--results-dir", default="outputs/latency_lut/nonexistent_v8_results")
    parser.add_argument("--output-json", default="outputs/latency_lut/pruning_sampling_collapse_v8.json")
    parser.add_argument("--output-md", default="outputs/latency_lut/pruning_sampling_collapse_v8.md")
    args = parser.parse_args(argv)
    v7_args = parse_v7_args(
        [
            "--candidates", args.candidates,
            "--export-dir", args.export_dir,
            "--dataset", args.dataset,
            "--results-dir", args.results_dir,
            "--output-json", args.output_json,
            "--output-md", args.output_md,
        ]
    )
    out = audit_v7(v7_args)
    semantic = audit_export_dir(args.export_dir)["summary"]
    summary = out["summary"]
    summary.update(
        {
            "root_node_local_pruner_semantics_pass": semantic.get("root_node_local_pruner_semantics_pass", False),
            "sampling_collapse_fixed": summary.get("num_unique_actual_param_keep_ratio", 0) >= 3 and summary.get("num_unique_onnx_conv_shape_hash", 0) >= 3,
            "different_keep_ratios_produce_different_structures": summary.get("num_unique_onnx_conv_shape_hash", 0) >= 3,
        }
    )
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Pruning Sampling Collapse v8\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
