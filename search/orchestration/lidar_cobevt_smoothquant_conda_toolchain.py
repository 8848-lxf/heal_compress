"""Fresh-process staged ONNX exports for the CoBEVT SmoothQuant audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
    selective_smoothquant_config,
)
from search.orchestration.lidar_cobevt_smoothquant_projection import (
    capture_projection_inputs,
)


@dataclass(frozen=True)
class ExportStageSpec:
    stage: str
    scope: str
    profile_id: str
    requires_stage: str | None = None


def export_stage_specs() -> tuple[ExportStageSpec, ...]:
    return (
        ExportStageSpec("E0", "toy_linear", "SQ1"),
        ExportStageSpec("E1", "real_q_projection", "SQ1"),
        ExportStageSpec("E2", "single_attention_qk", "SQ1"),
        ExportStageSpec("E3", "full_model", "SQ1"),
        ExportStageSpec("E4", "full_model", "SQ2"),
        ExportStageSpec("E5", "full_model", "SQ3", requires_stage="E4"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


class _Projection(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        self.q_proj = copy.deepcopy(source)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.q_proj(value)


class _QKBlock(nn.Module):
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear) -> None:
        super().__init__()
        self.q_proj = copy.deepcopy(q_proj)
        self.k_proj = copy.deepcopy(k_proj)
        self.heads = 8
        self.d_qk = 32

    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q_input).reshape(1, -1, self.heads, self.d_qk)
        k = self.k_proj(k_input).reshape(1, -1, self.heads, self.d_qk)
        q = q.permute(0, 2, 1, 3) * (self.d_qk**-0.5)
        k = k.permute(0, 2, 1, 3)
        return torch.matmul(q, k.transpose(-1, -2))


def _quantize(
    model: nn.Module,
    examples: Sequence[tuple[torch.Tensor, ...]],
    *,
    alpha: float,
) -> nn.Module:
    import modelopt.torch.quantization as mtq  # type: ignore

    config = selective_smoothquant_config(
        ("q_projection", "k_projection"), alpha=float(alpha)
    )

    def forward_loop(target: nn.Module) -> None:
        target.eval()
        with torch.no_grad():
            for example in examples:
                target(*example)

    quantized = mtq.quantize(model, config, forward_loop=forward_loop)
    return quantized if isinstance(quantized, nn.Module) else model


def _export(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    destination: Path,
    *,
    input_names: Sequence[str],
) -> dict[str, Any]:
    import onnx

    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)
    if not torch.is_tensor(outputs) or not bool(torch.isfinite(outputs).all()):
        raise RuntimeError("staged_export_nonfinite_output")
    torch.onnx.export(
        model,
        inputs,
        str(destination),
        input_names=list(input_names),
        output_names=["output"],
        opset_version=17,
        do_constant_folding=False,
    )
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph)
    onnx.save(inferred, str(destination))
    return {
        "onnx_path": str(destination),
        "onnx_sha256": _sha256(destination),
        "onnx_checker": "passed",
        "shape_type_inference": "passed",
        "output_shape": list(outputs.shape),
        "output_finite": True,
    }


def run_stage(
    *,
    stage: str,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    manifest: Path,
    device: torch.device,
    alpha: float,
) -> dict[str, Any]:
    specs = {row.stage: row for row in export_stage_specs()}
    if stage not in {"E0", "E1", "E2"}:
        raise ValueError(f"local_staged_export_unsupported:{stage}")
    spec = specs[stage]
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "E0":
        torch.manual_seed(20260720)
        source = nn.Linear(32, 16).to(device).eval()
        inputs = (torch.linspace(-2.0, 2.0, 128, device=device).reshape(4, 32),)
        model = _quantize(_Projection(source).to(device), [inputs], alpha=alpha)
        result = _export(
            model,
            inputs,
            output_dir / "model.onnx",
            input_names=("input",),
        )
    else:
        modules, captures, capture = capture_projection_inputs(
            checkpoint=checkpoint,
            config=config,
            heal_root=heal_root,
            manifest=manifest,
            device=device,
            max_rows_per_frame=32,
        )
        block = sorted(modules)[0]
        if stage == "E1":
            inputs = (captures[(block, "q_projection")][0].to(device),)
            model = _quantize(
                _Projection(modules[block]["q_projection"]).to(device),
                [inputs],
                alpha=alpha,
            )
            result = _export(
                model,
                inputs,
                output_dir / "model.onnx",
                input_names=("q_projection_input",),
            )
        else:
            inputs = (
                captures[(block, "q_projection")][0].to(device),
                captures[(block, "k_projection")][0].to(device),
            )
            model = _quantize(
                _QKBlock(
                    modules[block]["q_projection"], modules[block]["k_projection"]
                ).to(device),
                [inputs],
                alpha=alpha,
            )
            result = _export(
                model,
                inputs,
                output_dir / "model.onnx",
                input_names=("q_projection_input", "k_projection_input"),
            )
        result["capture_manifest"] = capture
        result["attention_block"] = block
    payload = {
        **asdict(spec),
        **result,
        "alpha": float(alpha),
        "process_id": os.getpid(),
        "python": os.path.realpath(__import__("sys").executable),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "ok",
    }
    _write_json(output_dir / "stage_result.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("E0", "E1", "E2"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_stage(
        stage=args.stage,
        output_dir=Path(args.output_dir),
        checkpoint=Path(args.checkpoint),
        config=Path(args.config),
        heal_root=Path(args.heal_root),
        manifest=Path(args.manifest),
        device=torch.device(args.device),
        alpha=args.alpha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
