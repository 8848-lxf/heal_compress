from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
UNIAD = ROOT.parent
for p in (UNIAD, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.test_general_pruner import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_HEAL_ROOT, load_heal_model, setup_logger
from tools.latency_lut.audit_grouped_conv_keep_distribution_v84 import audit_records, summarize
from tools.latency_lut.fixed_width_boundary_registry_v84 import protected_prefixes_for_fixed_width_boundaries


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _l2_prune_indices(module: nn.Conv2d, prune_ratio: float) -> tuple[list[int], list[int]]:
    weight = module.weight.detach().float()
    scores = weight.pow(2).view(weight.shape[0], -1).sum(dim=1)
    prune_count = int(round(module.out_channels * prune_ratio))
    prune_count = max(0, min(module.out_channels - 1, prune_count))
    if prune_count <= 0:
        return [], list(range(module.out_channels))
    _, prune_idx = torch.topk(scores, prune_count, largest=False, sorted=False)
    pruned = sorted(int(i) for i in prune_idx.tolist())
    kept = [i for i in range(module.out_channels) if i not in set(pruned)]
    return pruned, kept


def _protected(name: str, prefixes: list[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    error: dict[str, Any] = {"physical_prune_success": False, "forward_or_export_success": False, "failure_reasons": []}
    pruning_groups: list[dict[str, Any]] = []
    grouped_records: list[dict[str, Any]] = []
    try:
        logger = setup_logger(out)
        device = torch.device(args.device)
        model_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            model_config=args.model_config,
            heal_root=args.heal_root,
        )
        model, _adapter = load_heal_model(model_args, device, logger)
        protected_prefixes = protected_prefixes_for_fixed_width_boundaries()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d) or _protected(name, protected_prefixes):
                continue
            pruned, kept = _l2_prune_indices(module, 0.5)
            group = {
                "root_module": name,
                "root_type": "Conv2d",
                "pruning_fn": "tp.prune_conv_out_channels",
                "requested_prune_ratio": 0.5,
                "num_channels_before": int(module.out_channels),
                "num_pruned_idxs": len(pruned),
                "num_kept_idxs": len(kept),
                "pruned_idxs": pruned,
                "kept_idxs": kept,
                "tp_check_pruning_group_pass": False,
                "tp_group_ops": [
                    {"module": name, "module_type": "Conv2d", "pruning_fn": "prune_conv_out_channels", "axis": "out", "idxs": pruned}
                ],
            }
            pruning_groups.append(group)
            if module.groups > 1:
                grouped_records.append(
                    {
                        "module": name,
                        "groups_before": int(module.groups),
                        "groups_after": int(module.groups),
                        "per_group_before": int(module.out_channels // module.groups),
                        "C_in_before": int(module.in_channels),
                        "C_out_before": int(module.out_channels),
                        "C_in_after": int(module.in_channels),
                        "C_out_after": len(kept),
                        "kept_out_indices": kept,
                        "kept_in_indices": [i for i in range(module.in_channels) if i < len(kept)],
                        "group_keep_map": {},
                    }
                )
        # Actual TP physical surgery is intentionally best-effort. The selected
        # groups above are saved before any mutation so failures remain auditable.
        try:
            import torch_pruning as tp  # type: ignore

            error["torch_pruning_version"] = getattr(tp, "__version__", "unknown")
            error["failure_reasons"].append("tp_native_physical_prune_not_executed_in_diagnostic_path")
        except Exception as exc:
            error["failure_reasons"].append(f"torch_pruning_import_failed:{type(exc).__name__}")
        rows = audit_records("tp_native", grouped_records, require_group_keep_map=False)
        _write_json(out / "tp_pruning_groups.json", pruning_groups)
        _write_json(out / "prune_replay.json", {"operations": [op for g in pruning_groups for op in g["tp_group_ops"]]})
        _write_json(out / "tp_grouped_conv_audit.json", {"records": rows, "summary": summarize(rows)})
        _write_json(out / "structure_audit.json", {"physical_prune_success": False, "reason": "diagnostic_pre_prune_group_audit_only"})
    except Exception as exc:
        error["failure_reasons"].append(type(exc).__name__)
        error["traceback"] = traceback.format_exc()
    _write_json(out / "error_report.json", error)
    return error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/latency_lut/tp_vs_current_pruner_50pct_v84/tp_native")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
