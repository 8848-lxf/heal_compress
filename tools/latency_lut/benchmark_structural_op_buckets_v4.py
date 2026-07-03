from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


WIDTHS = [8, 16, 32, 64, 128]


def _load_buckets(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("buckets") or data)


def _bucket_from_existing(src: dict[str, Any], width: int | None = None) -> dict[str, Any]:
    lat = float(src.get("latency_p50_ms") or src.get("p50_ms") or src.get("latency_ms") or 0.0)
    p90 = float(src.get("latency_p90_ms") or src.get("p90_ms") or lat)
    return {
        "bucket_id": str(src.get("bucket_id")),
        "component_type": str(src.get("component_type")),
        "width": width or (src.get("shape_signature") or {}).get("C") or 0,
        "H": (src.get("shape_signature") or {}).get("H") or 0,
        "W": (src.get("shape_signature") or {}).get("W") or 0,
        "precision": str(src.get("precision") or ""),
        "precision_transition": str(src.get("precision_transition") or ""),
        "latency_p50_ms": lat,
        "latency_p90_ms": p90,
        "latency_mean_ms": float(src.get("latency_ms") or lat),
        "latency_std_ms": float(src.get("uncertainty_ms") or 0.0),
        "source": str(src.get("source") or "measured"),
        "valid": str(src.get("source") or "") == "measured",
        "invalid_reason": "" if str(src.get("source") or "") == "measured" else f"{src.get('source') or 'unmeasured'}_not_exact",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    existing = _load_buckets(args.source_buckets)
    out: list[dict[str, Any]] = []
    for src in existing:
        if src.get("source") == "measured":
            out.append(_bucket_from_existing(src))
    measured_types = {row["component_type"] for row in out if row["valid"]}
    required = []
    for width in WIDTHS:
        for precision in ("FP32", "FP16"):
            required.append(("residual_add", width, precision, ""))
        for branches in (2, 3):
            for precision in ("FP32", "FP16"):
                required.append((f"concat_{branches}branch", width, precision, ""))
        for transition in ("FP16->FP32", "FP32->FP16", "FP16->INT8_QDQ", "INT8_QDQ->FP16"):
            required.append(("precision_boundary", width, "", transition))
    for component_type, width, precision, transition in required:
        if component_type in measured_types or (component_type == "precision_boundary" and any("boundary" in row["component_type"] for row in out)):
            continue
        out.append(
            {
                "bucket_id": f"{component_type}_w{width}_{precision or transition}",
                "component_type": component_type,
                "width": width,
                "H": 16,
                "W": 16,
                "precision": precision,
                "precision_transition": transition,
                "latency_p50_ms": 0.0,
                "latency_p90_ms": 0.0,
                "latency_mean_ms": 0.0,
                "latency_std_ms": 0.0,
                "source": "unmeasured",
                "valid": False,
                "invalid_reason": "structural_microbenchmark_not_run",
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"buckets": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "total_buckets": len(out),
        "valid_measured_buckets": sum(1 for row in out if row.get("valid")),
        "invalid_buckets": sum(1 for row in out if not row.get("valid")),
        "by_component_type": dict(Counter(row["component_type"] for row in out)),
        "invalid_by_reason": dict(Counter(row.get("invalid_reason") for row in out if not row.get("valid"))),
    }
    Path(args.report).write_text("# Structural Op Latency Buckets v4 Report\n\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-buckets", "--source_buckets", dest="source_buckets", default="outputs/latency_lut/non_compute_latency_buckets_v3.json")
    parser.add_argument("--output", default="outputs/latency_lut/structural_op_latency_buckets_v4.json")
    parser.add_argument("--report", default="outputs/latency_lut/structural_op_latency_buckets_v4_report.md")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
