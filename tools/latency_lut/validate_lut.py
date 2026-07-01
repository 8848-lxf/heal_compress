from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.calibration import (
    load_calibration_model,
    load_calibration_samples,
    regression_metrics,
)
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate latency LUT/calibration against full-engine samples.")
    parser.add_argument("--samples", default="outputs/latency_lut/full_engine_samples.jsonl")
    parser.add_argument("--lut", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--output", default="outputs/latency_lut/validation_report.json")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser.parse_args(argv)


def _topk_overlap(pred: list[float], real: list[float], k: int) -> float | None:
    if not pred or not real:
        return None
    k = min(k, len(pred), len(real))
    pred_top = {idx for idx, _ in sorted(enumerate(pred), key=lambda item: item[1])[:k]}
    real_top = {idx for idx, _ in sorted(enumerate(real), key=lambda item: item[1])[:k]}
    return len(pred_top & real_top) / float(k)


def _kendall(pred: list[float], real: list[float]) -> float | None:
    n = len(pred)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = pred[i] - pred[j]
            b = real[i] - real[j]
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    denom = concordant + discordant
    return None if denom == 0 else (concordant - discordant) / float(denom)


def run(args: argparse.Namespace) -> dict:
    samples = load_calibration_samples(args.samples)
    _db = LatencyLUTDatabase.from_jsonl(args.lut) if Path(args.lut).is_file() else LatencyLUTDatabase()
    model = load_calibration_model(args.calibration)
    preds = [model.predict(sample.predicted_lut_ms, sample.features) for sample in samples]
    real = [sample.real_engine_p50_ms for sample in samples]
    metrics = regression_metrics(samples, preds)
    metrics["Kendall"] = _kendall(preds, real)
    metrics["Top-5 ranking overlap"] = _topk_overlap(preds, real, 5)
    metrics["Top-10 ranking overlap"] = _topk_overlap(preds, real, 10)
    metrics["constraint violation recall"] = None
    report = {
        "num_samples": len(samples),
        "lut_path": str(args.lut),
        "calibration_path": str(args.calibration),
        "metrics": metrics,
        "dry_run": bool(args.dry_run),
        "note": "no samples available" if not samples else None,
        "calibration_sufficient": len(samples) >= 10,
        "warning": "validation/calibration is insufficient because num_samples < 10" if len(samples) < 10 else None,
    }
    if not args.dry_run:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
