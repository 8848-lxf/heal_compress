#!/usr/bin/env python3
"""v9.9.1 optional C strategy smoke for ordinary grouped Conv2d."""

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
from heal_compress.pruning.grouped_conv import resolve_grouped_conv_true_group_block_keep  # noqa: E402
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from tools.latency_lut.audit_dependency_graph_v98 import write_json  # noqa: E402


OUT_DEFAULT = Path("outputs/latency_lut/grouped_conv_c_true_group_block_v991")


class GroupedBlockToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(16, 128, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.bn2(self.conv2(self.conv1(x))))


def _block_indices(groups: list[int], per_group: int) -> list[int]:
    out: list[int] = []
    for group_id in groups:
        out.extend(range(group_id * per_group, (group_id + 1) * per_group))
    return out


def build_true_group_block_plan(
    *,
    grouped_module_name: str,
    upstream_module_name: str,
    upstream_bn_module_name: str = "",
    bn_module_name: str,
    downstream_module_name: str,
    old_groups: int,
    in_per_group: int,
    out_per_group: int,
    pruned_old_groups: list[int],
    source_recipe_id: str = "c_true_group_block",
) -> GlobalPhysicalPrunePlan:
    input_prune = _block_indices(pruned_old_groups, in_per_group)
    output_prune = _block_indices(pruned_old_groups, out_per_group)
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest(upstream_module_name, "out", input_prune, source_recipe_id=source_recipe_id))
    if upstream_bn_module_name:
        plan.add_request(ModuleAxisPruneRequest(upstream_bn_module_name, "out", input_prune, source_recipe_id=source_recipe_id))
    plan.add_request(
        ModuleAxisPruneRequest(
            grouped_module_name,
            "grouped_true_group_block",
            pruned_old_groups,
            source_recipe_id=source_recipe_id,
            metadata={
                "old_groups": old_groups,
                "pruned_old_groups": pruned_old_groups,
                "kept_old_groups": [group_id for group_id in range(old_groups) if group_id not in set(pruned_old_groups)],
                "in_per_group_before": in_per_group,
                "out_per_group_before": out_per_group,
                "groups_after": old_groups - len(pruned_old_groups),
            },
        )
    )
    plan.add_request(ModuleAxisPruneRequest(bn_module_name, "out", output_prune, source_recipe_id=source_recipe_id))
    plan.add_request(ModuleAxisPruneRequest(downstream_module_name, "in", output_prune, source_recipe_id=source_recipe_id))
    return plan


def _toy_shapes(model: GroupedBlockToy) -> dict[str, Any]:
    return {
        "conv1_out": int(model.conv1.out_channels),
        "conv2_in": int(model.conv2.in_channels),
        "conv2_out": int(model.conv2.out_channels),
        "conv2_groups": int(model.conv2.groups),
        "conv2_weight": [int(v) for v in tuple(model.conv2.weight.shape)],
        "bn2_features": int(model.bn2.num_features),
        "conv3_in": int(model.conv3.in_channels),
    }


def run_toy_c_strategy_reports(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = GroupedBlockToy().eval()
    before = _toy_shapes(model)
    pruned_groups = [0, 3, 7, 31]
    plan = build_true_group_block_plan(
        grouped_module_name="conv2",
        upstream_module_name="conv1",
        bn_module_name="bn2",
        downstream_module_name="conv3",
        old_groups=32,
        in_per_group=4,
        out_per_group=4,
        pruned_old_groups=pruned_groups,
        source_recipe_id="toy_c",
    )
    sim = GlobalPlanShapeSimulator(model, plan, group_conv_align=4).simulate()
    failure_cases = []
    partial = resolve_grouped_conv_true_group_block_keep(model.conv2, prune_indices=[1, 2])
    if not partial.get("legal", False):
        failure_cases.append(partial)
    apply_report: dict[str, Any] = {}
    forward_passed = False
    error = ""
    if sim.get("legal", False):
        try:
            apply_report = plan.apply_one_shot(model)
            with torch.no_grad():
                model(torch.randn(2, 16, 8, 8))
            forward_passed = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = "simulator_illegal"

    toy_report = {
        "before": before,
        "after": _toy_shapes(model),
        "pruned_old_groups": pruned_groups,
        "toy_forward_passed": forward_passed,
        "failure_reason": error,
    }
    write_json(out_dir / "c_true_group_block_toy_report.json", toy_report)
    write_json(out_dir / "c_true_group_block_global_plan_report.json", {**plan.audit(), "apply_report": apply_report})
    write_json(out_dir / "c_true_group_block_simulator_report.json", sim)
    with (out_dir / "c_true_group_block_failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in failure_cases:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    verdict = [
        "# C True Group-Block Strategy Verdict",
        "",
        f"- toy_forward_passed: {forward_passed}",
        "- default grouped Conv2d strategy remains A (`flat_output_groups_fixed`); C is optional.",
        f"- partial_group_reject_reason: {partial.get('reason', '')}",
    ]
    (out_dir / "c_true_group_block_strategy_verdict.md").write_text("\n".join(verdict) + "\n", encoding="utf-8")
    return {"toy_forward_passed": forward_passed, "output_dir": str(out_dir), "failure_reason": error}


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
        if not isinstance(module, nn.Conv2d) or module.groups <= 1:
            continue
        prefix, _, leaf = name.rpartition(".")
        if leaf != "conv2":
            continue
        upstream = f"{prefix}.conv1"
        bn = f"{prefix}.bn2"
        downstream = f"{prefix}.conv3"
        if not all(key in modules for key in (upstream, bn, downstream)):
            continue
        if not isinstance(modules[upstream], nn.Conv2d) or not isinstance(modules[bn], nn.modules.batchnorm._BatchNorm) or not isinstance(modules[downstream], nn.Conv2d):
            continue
        if module.in_channels % module.groups or module.out_channels % module.groups:
            continue
        return {
            "grouped_conv": name,
            "upstream": upstream,
            "upstream_bn": f"{prefix}.bn1" if f"{prefix}.bn1" in modules else "",
            "bn": bn,
            "downstream": downstream,
            "groups": int(module.groups),
            "in_per_group": int(module.in_channels // module.groups),
            "out_per_group": int(module.out_channels // module.groups),
        }
    return {}


def run_real_model_smoke(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Any] = {}
    plan_payload: dict[str, Any] = {}
    sim_report: dict[str, Any] = {"legal": False, "reason": "not_run"}
    forward_report: dict[str, Any] = {"forward_passed": False, "failure_reason": "not_run"}
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
            failure = "missing_simple_bottleneck_grouped_conv_candidate"
        else:
            plan = build_true_group_block_plan(
                grouped_module_name=selected["grouped_conv"],
                upstream_module_name=selected["upstream"],
                upstream_bn_module_name=selected.get("upstream_bn", ""),
                bn_module_name=selected["bn"],
                downstream_module_name=selected["downstream"],
                old_groups=selected["groups"],
                in_per_group=selected["in_per_group"],
                out_per_group=selected["out_per_group"],
                pruned_old_groups=[0],
                source_recipe_id="real_c_smoke",
            )
            plan_payload = plan.audit()
            sim_report = GlobalPlanShapeSimulator(model, plan, group_conv_align=1).simulate()
            if sim_report.get("legal", False):
                sample = adapter.build_synthetic_batch(model)
                cloned = copy.deepcopy(model)
                plan.apply_one_shot(cloned)
                cloned.eval()
                with torch.no_grad():
                    adapter.forward_for_task(cloned, sample)
                forward_report = {"forward_passed": True, "failure_reason": ""}
            else:
                failure = "simulator_illegal"
                forward_report = {"forward_passed": False, "failure_reason": failure, "issues": sim_report.get("issues", [])}
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        forward_report = {"forward_passed": False, "failure_reason": failure, "traceback": traceback.format_exc()}

    write_json(out_dir / "c_real_model_selected_grouped_conv.json", selected or {"failure_reason": failure})
    write_json(out_dir / "c_real_model_global_plan.json", plan_payload)
    write_json(out_dir / "c_real_model_simulator_report.json", sim_report)
    write_json(out_dir / "c_real_model_forward_smoke_report.json", forward_report)
    write_json(out_dir / "c_real_model_failure_reason.json", {"failure_reason": failure})
    return {
        "real_model_smoke_attempted": True,
        "real_model_forward_passed": bool(forward_report.get("forward_passed")),
        "failure_reason": failure,
        "selected": selected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--model-config", default="${MODEL_ROOT}/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-root", default="../../HEAL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--skip-real-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    toy = run_toy_c_strategy_reports(out)
    real = {"real_model_smoke_attempted": False, "real_model_forward_passed": False, "failure_reason": "skip_real_smoke"}
    if not args.skip_real_smoke:
        real = run_real_model_smoke(args, out / "real_model_smoke")
    summary = {"toy": toy, "real": real}
    write_json(out / "c_true_group_block_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
