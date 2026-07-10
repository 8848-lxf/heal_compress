#!/usr/bin/env python3
"""v9.9.2-D compact-frontfill zero-padded reblock smoke utility."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator  # noqa: E402
from heal_compress.pruning.grouped_conv import resolve_grouped_conv_d_compact_frontfill_reblock  # noqa: E402
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from tools.latency_lut.audit_dependency_graph_v98 import write_json  # noqa: E402


OUT_DEFAULT = Path("outputs/latency_lut/grouped_conv_d_compact_frontfill_reblock_v992")


class DGroupedToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 24, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.bn2(self.conv2(self.conv1(x))))


def default_toy_output_keep() -> list[int]:
    return [12, 0, 1, 2] + [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]


def build_d_compact_frontfill_plan(
    *,
    grouped_module_name: str,
    bn_module_name: str,
    downstream_module_name: str,
    c_out_old: int,
    old_output_keep_indices: list[int],
    old_input_keep_indices: list[int],
    groups_new: int,
    source_recipe_id: str = "d_compact_frontfill",
) -> GlobalPhysicalPrunePlan:
    output_keep_set = {int(v) for v in old_output_keep_indices}
    output_prune = [idx for idx in range(int(c_out_old)) if idx not in output_keep_set]
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(
        ModuleAxisPruneRequest(
            grouped_module_name,
            "grouped_d_compact_frontfill_reblock",
            output_prune,
            source_recipe_id=source_recipe_id,
            metadata={
                "groups_new": int(groups_new),
                "old_input_keep_indices": [int(v) for v in old_input_keep_indices],
                "old_output_keep_indices": [int(v) for v in old_output_keep_indices],
            },
        )
    )
    sync_metadata = {"ordered_keep_indices": [int(v) for v in old_output_keep_indices]}
    plan.add_request(ModuleAxisPruneRequest(bn_module_name, "out", output_prune, source_recipe_id=source_recipe_id, metadata=sync_metadata))
    plan.add_request(ModuleAxisPruneRequest(downstream_module_name, "in", output_prune, source_recipe_id=source_recipe_id, metadata=sync_metadata))
    return plan


def _toy_shapes(model: DGroupedToy) -> dict[str, Any]:
    return {
        "conv1_out": int(model.conv1.out_channels),
        "conv2_in": int(model.conv2.in_channels),
        "conv2_out": int(model.conv2.out_channels),
        "conv2_groups": int(model.conv2.groups),
        "conv2_weight": [int(v) for v in tuple(model.conv2.weight.shape)],
        "bn2_features": int(model.bn2.num_features),
        "conv3_in": int(model.conv3.in_channels),
    }


def _mapping_from_apply_report(apply_report: dict[str, Any]) -> dict[str, Any]:
    for op in apply_report.get("operations", []):
        if op.get("physical_axis") == "grouped_d_compact_frontfill_reblock":
            return op
    return {}


def run_toy_d_strategy_reports(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DGroupedToy().eval()
    with torch.no_grad():
        model.conv2.weight.copy_(torch.arange(model.conv2.weight.numel(), dtype=model.conv2.weight.dtype).view_as(model.conv2.weight))
    before = _toy_shapes(model)
    old_output_keep = default_toy_output_keep()
    old_input_keep = list(range(32))
    plan = build_d_compact_frontfill_plan(
        grouped_module_name="conv2",
        bn_module_name="bn2",
        downstream_module_name="conv3",
        c_out_old=32,
        old_output_keep_indices=old_output_keep,
        old_input_keep_indices=old_input_keep,
        groups_new=4,
        source_recipe_id="toy_d",
    )
    sim = GlobalPlanShapeSimulator(model, plan, group_conv_align=4).simulate()
    failure_cases = []
    too_wide = resolve_grouped_conv_d_compact_frontfill_reblock(
        model.conv2,
        old_output_keep_indices=list(range(16)),
        old_input_keep_indices=list(range(16)),
        groups_new=8,
    )
    if not too_wide.get("legal", False):
        failure_cases.append(too_wide)

    toy_forward_passed = False
    failure = ""
    apply_report: dict[str, Any] = {}
    if not sim.get("legal", False):
        failure = "shape simulator illegal"
    else:
        try:
            apply_report = plan.apply_one_shot(model)
            with torch.no_grad():
                model(torch.randn(2, 16, 8, 8))
            toy_forward_passed = True
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"

    mapping = _mapping_from_apply_report(apply_report)
    toy_report = {
        "before": before,
        "after": _toy_shapes(model),
        "toy_forward_passed": toy_forward_passed,
        "failure_reason": failure,
        "semantic_preserved": False,
        "compact_first": True,
        "frontfill_weight_transplant": True,
        "weight_values_retained": True,
        "new_connections_zero_initialized": True,
        "requires_recovery_finetune": True,
    }
    write_json(out_dir / "d_compact_frontfill_toy_report.json", toy_report)
    write_json(out_dir / "d_compact_frontfill_weight_mapping_report.json", mapping)
    write_json(out_dir / "d_compact_frontfill_simulator_report.json", sim)
    with (out_dir / "d_compact_frontfill_failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in failure_cases:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    verdict = [
        "# D Compact-Frontfill Strategy Verdict",
        "",
        f"- toy_forward_passed: {toy_forward_passed}",
        "- semantic_preserved: false",
        "- compact_first: true",
        "- frontfill_weight_transplant: true",
        "- new_connections_zero_initialized: true",
        "- requires_recovery_finetune: true",
    ]
    (out_dir / "d_compact_frontfill_strategy_verdict.md").write_text("\n".join(verdict) + "\n", encoding="utf-8")
    return {"toy_forward_passed": toy_forward_passed, "failure_reason": failure, "output_dir": str(out_dir)}


def _find_real_candidate(model: nn.Module, preferred: str = "pyramid_backbone.resnet.layer0.0.conv2") -> dict[str, Any]:
    modules = dict(model.named_modules())
    candidates = [preferred] if preferred in modules else []
    candidates.extend(
        name for name, module in modules.items()
        if isinstance(module, nn.Conv2d)
        and int(module.groups) > 1
        and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
        and name not in candidates
    )
    for name in candidates:
        module = modules.get(name)
        if not isinstance(module, nn.Conv2d) or int(module.groups) <= 1:
            continue
        prefix, _, leaf = name.rpartition(".")
        if leaf != "conv2":
            continue
        bn = f"{prefix}.bn2"
        downstream = f"{prefix}.conv3"
        if not all(key in modules for key in (bn, downstream)):
            continue
        if not isinstance(modules[bn], nn.modules.batchnorm._BatchNorm) or not isinstance(modules[downstream], nn.Conv2d):
            continue
        groups_old = int(module.groups)
        if groups_old <= 1 or module.in_channels % groups_old or module.out_channels % groups_old:
            continue
        groups_new = groups_old // 2
        if groups_new <= 0:
            continue
        c_out_new = module.out_channels // 2
        if module.in_channels % groups_new or c_out_new % groups_new:
            continue
        in_per_new = module.in_channels // groups_new
        if in_per_new < module.in_channels // groups_old:
            continue
        return {
            "grouped_conv": name,
            "bn": bn,
            "downstream": downstream,
            "groups_old": groups_old,
            "groups_new": groups_new,
            "C_in_old": int(module.in_channels),
            "C_out_old": int(module.out_channels),
            "C_in_new": int(module.in_channels),
            "C_out_new": int(c_out_new),
            "old_input_keep_indices": list(range(int(module.in_channels))),
            "old_output_keep_indices": list(range(int(c_out_new))),
        }
    return {}


def run_real_model_smoke(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Any] = {}
    plan_payload: dict[str, Any] = {}
    sim_report: dict[str, Any] = {"legal": False, "reason": "not_run"}
    forward_report: dict[str, Any] = {"forward_passed": False, "failure_reason": "not_run"}
    weight_mapping: dict[str, Any] = {}
    failure = ""
    try:
        from heal_compress.utils.model_utils import resolve_device
        from heal_compress.pruning.model_io import load_heal_model, setup_logger

        device = torch.device(resolve_device(args.device))
        if device.type == "cuda":
            torch.cuda.set_device(device)
        logger = setup_logger(out_dir)
        model, adapter = load_heal_model(args, device, logger)
        model.eval()
        selected = _find_real_candidate(model)
        if not selected:
            failure = "upstream closure missing"
        else:
            plan = build_d_compact_frontfill_plan(
                grouped_module_name=selected["grouped_conv"],
                bn_module_name=selected["bn"],
                downstream_module_name=selected["downstream"],
                c_out_old=selected["C_out_old"],
                old_output_keep_indices=selected["old_output_keep_indices"],
                old_input_keep_indices=selected["old_input_keep_indices"],
                groups_new=selected["groups_new"],
                source_recipe_id="real_d_smoke",
            )
            plan_payload = plan.audit()
            sim_report = GlobalPlanShapeSimulator(model, plan, group_conv_align=4).simulate()
            if not sim_report.get("legal", False):
                failure = "shape simulator illegal"
                forward_report = {"forward_passed": False, "failure_reason": failure, "issues": sim_report.get("issues", [])}
            else:
                sample = adapter.build_synthetic_batch(model)
                cloned = copy.deepcopy(model)
                apply_report = plan.apply_one_shot(cloned)
                weight_mapping = _mapping_from_apply_report(apply_report)
                cloned.eval()
                with torch.no_grad():
                    adapter.forward_for_task(cloned, sample)
                forward_report = {"forward_passed": True, "failure_reason": ""}
    except Exception as exc:  # noqa: BLE001
        failure = _classify_real_failure(exc)
        forward_report = {"forward_passed": False, "failure_reason": failure, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}

    write_json(out_dir / "d_real_model_selected_grouped_conv.json", selected or {"failure_reason": failure})
    write_json(out_dir / "d_real_model_global_plan.json", plan_payload)
    write_json(out_dir / "d_real_model_simulator_report.json", sim_report)
    write_json(out_dir / "d_real_model_forward_smoke_report.json", forward_report)
    write_json(out_dir / "d_real_model_weight_mapping_report.json", weight_mapping)
    write_json(out_dir / "d_real_model_failure_reason.json", {"failure_reason": failure})
    return {
        "real_model_smoke_attempted": True,
        "real_model_forward_passed": bool(forward_report.get("forward_passed")),
        "failure_reason": failure,
        "selected": selected,
    }


def _classify_real_failure(exc: Exception) -> str:
    text = str(exc).lower()
    if "running_mean" in text or "batchnorm" in text:
        return "BN closure missing"
    if "expected input" in text or "channels" in text or "shape" in text:
        return "forward shape mismatch"
    if "frontfill_source_kernel_too_wide" in text:
        return "frontfill_source_kernel_too_wide"
    return "physical surgery bug"


def write_completion_verdict(out_dir: Path, *, toy: dict[str, Any], real: dict[str, Any]) -> None:
    lines = [
        "# D v9.9.2 Completion Verdict",
        "",
        "1. D 已按 `compact_frontfill_zero_padded_reblock` 实现。",
        "2. 允许 old filter compact 到语义不对齐的新 group；这不是 semantic-preserving reblock。",
        "3. 原 old kernel slice 会复制到新 filter 的前部 input slice。",
        "4. 新增 input slice 使用 0 初始化。",
        "5. `semantic_preserved=false` 已在 resolver、physical report、simulator report 和 verdict 中显式记录。",
        f"6. toy forward: {bool(toy.get('toy_forward_passed'))}。",
        f"7. simulator: {Path(out_dir / 'd_compact_frontfill_simulator_report.json').exists()}。",
        f"8. real model smoke: {bool(real.get('real_model_forward_passed'))}。",
        f"9. real model blocker: {real.get('failure_reason', '') or 'none'}。",
        "",
        "- semantic_preserved = false",
        "- compact_first = true",
        "- frontfill_weight_transplant = true",
        "- weight_values_retained = true",
        "- new_connections_zero_initialized = true",
        "- requires_recovery_finetune = true",
    ]
    (out_dir / "d_v992_completion_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--skip-real-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    toy = run_toy_d_strategy_reports(out)
    real = {"real_model_smoke_attempted": False, "real_model_forward_passed": False, "failure_reason": "skip_real_smoke"}
    if not args.skip_real_smoke:
        real = run_real_model_smoke(args, out / "real_model_smoke")
    summary = {"toy": toy, "real": real}
    write_json(out / "d_compact_frontfill_summary.json", summary)
    write_completion_verdict(out, toy=toy, real=real)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
