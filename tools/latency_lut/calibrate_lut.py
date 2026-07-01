from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.calibration import (
    IdentityCalibrationModel,
    LinearCalibrationModel,
    load_calibration_samples,
    regression_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit latency LUT full-engine calibration model.")
    parser.add_argument("--calibration-samples", "--calibration_samples", dest="calibration_samples", default="outputs/latency_lut/full_engine_samples.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/calibration_model.json")
    parser.add_argument("--model", choices=["identity", "linear"], default="linear")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    samples = load_calibration_samples(args.calibration_samples)
    if args.model == "identity" or not samples:
        model = IdentityCalibrationModel()
    else:
        model = LinearCalibrationModel.fit(samples)
    preds = [model.predict(sample.predicted_lut_ms, sample.features) for sample in samples]
    report = {
        **model.to_dict(),
        "num_samples": len(samples),
        "metrics": regression_metrics(samples, preds),
        "dry_run": bool(args.dry_run),
        "note": "identity model emitted because no samples were available" if not samples else None,
        "calibration_sufficient": len(samples) >= 10,
        "warning": "calibration is insufficient because num_samples < 10" if len(samples) < 10 else None,
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
