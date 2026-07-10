#!/usr/bin/env python3
"""Torch-Pruning oracle for ordinary grouped ConvTranspose2d behavior."""

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

from tools.latency_lut.audit_dependency_graph_v98 import write_json  # noqa: E402


OUT_DEFAULT = Path("outputs/latency_lut/dependency_graph_audit_v991/tp_grouped_convtranspose_oracle")


class GroupedConvTransposeToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Conv2d(16, 64, 1)
        self.deconv = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=128,
            kernel_size=2,
            stride=2,
            groups=4,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(128)
        self.down = nn.Conv2d(128, 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.down(x)
        return x


def _shape(module: nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attr in ("in_channels", "out_channels", "groups", "num_features"):
        if hasattr(module, attr):
            payload[attr] = int(getattr(module, attr))
    if hasattr(module, "weight") and getattr(module, "weight") is not None:
        payload["weight_shape"] = [int(v) for v in tuple(module.weight.shape)]  # type: ignore[union-attr]
    if hasattr(module, "bias") and getattr(module, "bias") is not None:
        payload["bias_shape"] = [int(v) for v in tuple(module.bias.shape)]  # type: ignore[union-attr]
    return payload


def _model_shapes(model: GroupedConvTransposeToy) -> dict[str, Any]:
    return {
        "up": _shape(model.up),
        "deconv": _shape(model.deconv),
        "bn": _shape(model.bn),
        "down": _shape(model.down),
    }


def _case(case_name: str, *, root_kind: str, idxs: list[int]) -> dict[str, Any]:
    import torch_pruning as tp

    model = GroupedConvTransposeToy().eval()
    example = torch.randn(1, 16, 8, 8)
    before = _model_shapes(model)
    row: dict[str, Any] = {
        "case_name": case_name,
        "root_kind": root_kind,
        "idxs": idxs,
        "before": before,
        "build_group_ok": False,
        "check_pruning_group": False,
        "prune_ok": False,
        "forward_passed": False,
        "exception_type": "",
        "exception_message": "",
        "traceback": "",
        "group_details": "",
    }
    try:
        dg = tp.DependencyGraph().build_dependency(model, example_inputs=example)
        if root_kind == "deconv.out":
            group = dg.get_pruning_group(model.deconv, tp.prune_conv_out_channels, idxs)
        elif root_kind == "up.out":
            group = dg.get_pruning_group(model.up, tp.prune_conv_out_channels, idxs)
        else:
            raise ValueError(f"unknown root_kind={root_kind}")
        row["build_group_ok"] = True
        row["check_pruning_group"] = bool(dg.check_pruning_group(group))
        try:
            row["group_details"] = str(group.details())
        except Exception as exc:  # noqa: BLE001
            row["group_details"] = f"details_failed:{type(exc).__name__}: {exc}"
        if row["check_pruning_group"]:
            group.prune()
            row["prune_ok"] = True
        with torch.no_grad():
            out = model(example)
        row["forward_passed"] = True
        row["forward_output_shape"] = [int(v) for v in tuple(out.shape)]
    except Exception as exc:  # noqa: BLE001
        row["exception_type"] = type(exc).__name__
        row["exception_message"] = str(exc)
        row["traceback"] = traceback.format_exc()
    finally:
        row["after"] = _model_shapes(model)
        row["tp_deleted_deconv_out_channels"] = int(before["deconv"]["out_channels"] - row["after"]["deconv"].get("out_channels", 0))
        row["tp_deleted_deconv_in_channels"] = int(before["deconv"]["in_channels"] - row["after"]["deconv"].get("in_channels", 0))
    return row


def _analyze(out_cases: list[dict[str, Any]], in_cases: list[dict[str, Any]]) -> tuple[str, str]:
    out_pass = {case["case_name"]: bool(case["forward_passed"]) for case in out_cases}
    in_pass = {case["case_name"]: bool(case["forward_passed"]) for case in in_cases}
    all_failed = not any(out_pass.values()) and not any(in_pass.values())
    balanced_ok = out_pass.get("out_local_balanced_expanded", False) and in_pass.get("in_group_balanced", False)
    unbalanced_ok = out_pass.get("out_global_unbalanced", False) or in_pass.get("in_global_unbalanced", False)
    if all_failed:
        recommendation = "do_not_migrate_tp_unsupported"
    elif balanced_ok and not unbalanced_ok:
        recommendation = "migrate_group_balanced_input_output_resolver"
    else:
        recommendation = "investigate_further_due_to_ambiguous_tp_behavior"

    lines = [
        "# TP Grouped ConvTranspose2d Strategy Analysis",
        "",
        "1. TP 是否支持 ordinary grouped ConvTranspose2d？",
        f"   - build/check/prune/forward 结果见 JSON；任一 forward_passed={any(out_pass.values()) or any(in_pass.values())}。",
        "2. TP 对 grouped ConvTranspose2d out pruning 使用的策略：",
        f"   - global arbitrary output pruning forward_passed={out_pass.get('out_global_unbalanced', False)}。",
        f"   - group-balanced local output pruning forward_passed={out_pass.get('out_local_balanced_expanded', False)}。",
        "3. TP 对 grouped ConvTranspose2d in pruning 使用的策略：",
        f"   - global arbitrary input pruning via up.out forward_passed={in_pass.get('in_global_unbalanced', False)}。",
        f"   - group-balanced input pruning via up.out forward_passed={in_pass.get('in_group_balanced', False)}。",
        "4. TP 的行为是否与普通 grouped Conv2d 相同？",
        "   - 需要以报告中的 shape 和 forward 结果为准；ConvTranspose2d weight shape 为 [C_in, C_out/groups, kH, kW]。",
        "5. 是否建议迁移到当前剪枝器？",
        f"   - recommendation={recommendation}。",
        "6. 如果迁移，建议 resolver 名称和约束：",
        "   - grouped_convtranspose_group_balanced_input_output_resolver，除非报告显示 TP 完全不支持或行为含糊。",
    ]
    return recommendation, "\n".join(lines) + "\n"


def run_oracle(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_out = [5, 7, 9]
    out_balanced = [group * 32 + local for group in range(4) for local in local_out]
    local_in = [0, 3]
    in_balanced = [group * 16 + local for group in range(4) for local in local_in]

    out_cases = [
        _case("out_global_unbalanced", root_kind="deconv.out", idxs=[5, 7, 9]),
        _case("out_local_balanced_expanded", root_kind="deconv.out", idxs=out_balanced),
    ]
    in_cases = [
        _case("in_global_unbalanced", root_kind="up.out", idxs=[0, 3, 17, 21]),
        _case("in_group_balanced", root_kind="up.out", idxs=in_balanced),
    ]
    recommendation, analysis = _analyze(out_cases, in_cases)
    write_json(out_dir / "tp_grouped_convtranspose_out_pruning_report.json", {"cases": out_cases})
    write_json(out_dir / "tp_grouped_convtranspose_in_pruning_report.json", {"cases": in_cases})
    (out_dir / "tp_grouped_convtranspose_strategy_analysis.md").write_text(analysis, encoding="utf-8")
    write_json(
        out_dir / "tp_grouped_convtranspose_migration_recommendation.json",
        {
            "recommendation": recommendation,
            "allowed_values": [
                "do_not_migrate_tp_unsupported",
                "migrate_group_balanced_input_output_resolver",
                "migrate_true_group_block_only",
                "investigate_further_due_to_ambiguous_tp_behavior",
            ],
        },
    )
    return {
        "output_dir": str(out_dir),
        "num_out_cases": len(out_cases),
        "num_in_cases": len(in_cases),
        "recommendation": recommendation,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_oracle(Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
