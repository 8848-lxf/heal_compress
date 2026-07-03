from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


FEATURE_NAMES = [
    "T_compute_covered",
    "T_boundary_cast",
    "T_boundary_qdq",
    "T_plugin_or_scatter",
    "T_grid_sample_or_geometry",
    "T_elementwise_merge",
    "T_memory_reformat",
    "T_fixed_overhead",
    "num_precision_switches",
    "observed_fp32_layers",
    "observed_fp16_layers",
    "observed_int8_layers",
    "num_cast_inserted",
    "num_qdq_nodes_inserted",
    "pruning_keep_ratio",
    "param_keep_ratio",
    "channel_keep_ratio",
]


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    if not Path(path).is_file():
        return rows
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not row.get("valid_for_calibration", True):
            continue
        if row.get("T_real_p50") is None or row.get("T_lut_raw") is None:
            continue
        if _float(row.get("T_lut_raw")) <= 0.0:
            continue
        if int(row.get("missing_key_count") or 0) > 0 or int(row.get("unavailable_key_count") or 0) > 0:
            continue
        out.append(row)
    return out


def _matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[Any, Any]:
    import numpy as np

    x = []
    y = []
    for row in rows:
        x.append([_float(row.get(name)) for name in features] + [1.0])
        y.append(_float(row.get("T_real_p50")))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _fit_linear(rows: list[dict[str, Any]], *, ridge_alpha: float = 1.0) -> dict[str, Any]:
    import numpy as np

    x, y = _matrix(rows, FEATURE_NAMES)
    n, p = x.shape
    use_ridge = n < p or ridge_alpha > 0.0
    if use_ridge:
        reg = np.eye(p, dtype=float) * float(ridge_alpha)
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
        regularization = "ridge"
    else:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        regularization = "ordinary_least_squares"
    pred = x @ coef
    residuals = [
        {
            "candidate_id": row.get("candidate_id"),
            "T_real_p50": float(target),
            "T_pred": float(value),
            "residual": float(value - target),
        }
        for row, target, value in zip(rows, y, pred)
    ]
    return {
        "model_type": "least_squares_linear",
        "regularization": regularization,
        "ridge_alpha": float(ridge_alpha) if regularization == "ridge" else 0.0,
        "label": "T_real_p50",
        "feature_names": FEATURE_NAMES,
        "coefficients": [float(v) for v in coef[:-1]],
        "intercept": float(coef[-1]),
        "residuals": residuals,
        "train_metrics": _metrics([float(v) for v in y], [float(v) for v in pred]),
    }


def _predict_linear(model: dict[str, Any], row: dict[str, Any]) -> float:
    values = [_float(row.get(name)) for name in model["feature_names"]]
    return float(sum(c * v for c, v in zip(model["coefficients"], values)) + float(model["intercept"]))


def _leave_one_out(rows: list[dict[str, Any]], *, ridge_alpha: float) -> dict[str, Any]:
    if len(rows) < 3:
        return {"status": "skipped", "reason": "need_at_least_3_samples"}
    preds = []
    labels = []
    per_candidate = []
    for idx, row in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != idx]
        model = _fit_linear(train, ridge_alpha=ridge_alpha)
        pred = _predict_linear(model, row)
        label = _float(row.get("T_real_p50"))
        preds.append(pred)
        labels.append(label)
        per_candidate.append(
            {
                "candidate_id": row.get("candidate_id"),
                "T_real_p50": label,
                "T_pred": pred,
                "residual": pred - label,
            }
        )
    return {"status": "success", "metrics": _metrics(labels, preds), "predictions": per_candidate}


def _metrics(y: list[float], pred: list[float]) -> dict[str, float | None]:
    if not y:
        return {"MAE": None, "MAPE": None, "RMSE": None, "Pearson": None, "Spearman": None, "Top5_overlap": None, "Top10_overlap": None}
    err = [p - t for p, t in zip(pred, y)]
    mae = sum(abs(v) for v in err) / len(err)
    mape = sum(abs(p - t) / max(abs(t), 1e-9) for p, t in zip(pred, y)) / len(y) * 100.0
    rmse = math.sqrt(sum(v * v for v in err) / len(err))
    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "Pearson": _pearson(pred, y),
        "Spearman": _spearman(pred, y),
        "Top5_overlap": _topk_overlap(y, pred, 5),
        "Top10_overlap": _topk_overlap(y, pred, 10),
    }


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (da * db)


def _rank(values: list[float]) -> list[float]:
    out = [0.0] * len(values)
    for rank, (_value, idx) in enumerate(sorted((v, i) for i, v in enumerate(values)), start=1):
        out[idx] = float(rank)
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    return _pearson(_rank(a), _rank(b))


def _topk_overlap(y: list[float], pred: list[float], k: int) -> float | None:
    if not y:
        return None
    kk = min(int(k), len(y))
    truth = {idx for _value, idx in sorted((v, i) for i, v in enumerate(y))[:kk]}
    guessed = {idx for _value, idx in sorted((v, i) for i, v in enumerate(pred))[:kk]}
    return len(truth & guessed) / float(kk)


def train_calibrator(
    dataset: str | Path,
    output: str | Path,
    report_path: str | Path,
    *,
    model: str = "auto",
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    rows = _load_rows(dataset)
    valid = _valid_rows(rows)
    linear = _fit_linear(valid, ridge_alpha=ridge_alpha) if valid else {
        "model_type": "least_squares_linear",
        "regularization": "ridge",
        "ridge_alpha": float(ridge_alpha),
        "label": "T_real_p50",
        "feature_names": FEATURE_NAMES,
        "coefficients": [0.0 for _ in FEATURE_NAMES],
        "intercept": 0.0,
        "residuals": [],
        "train_metrics": _metrics([], []),
    }
    validation = _leave_one_out(valid, ridge_alpha=ridge_alpha)
    tiny_mlp = {"status": "skipped", "skipped_reason": "insufficient_samples", "min_samples": 50}
    selected = "least_squares_linear"
    payload = {
        **linear,
        "selected_model": selected,
        "num_samples_total": len(rows),
        "num_samples_valid": len(valid),
        "tiny_mlp": tiny_mlp,
        "validation_method": "leave_one_out" if validation.get("status") == "success" else "skipped",
        "validation": validation,
        "current_calibrator_is_preliminary": len(valid) < 50,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "success" if valid else "failed_no_valid_samples",
        "valid_samples": len(valid),
        "total_samples": len(rows),
        "selected_model": selected,
        "metrics": payload["train_metrics"],
        "validation": validation,
        "tiny_mlp": tiny_mlp,
        "current_calibrator_is_preliminary": payload["current_calibrator_is_preliminary"],
    }
    lines = [
        "# Full Engine Latency Calibration V2 Report",
        "",
        f"- total_samples: {len(rows)}",
        f"- valid_samples: {len(valid)}",
        f"- selected_model: {selected}",
        f"- regularization: {payload['regularization']}",
        f"- current_calibrator_is_preliminary: {payload['current_calibrator_is_preliminary']}",
        f"- MAE: {payload['train_metrics'].get('MAE')}",
        f"- MAPE: {payload['train_metrics'].get('MAPE')}",
        f"- RMSE: {payload['train_metrics'].get('RMSE')}",
        f"- Pearson: {payload['train_metrics'].get('Pearson')}",
        f"- Spearman: {payload['train_metrics'].get('Spearman')}",
        f"- Top5_overlap: {payload['train_metrics'].get('Top5_overlap')}",
        f"- Top10_overlap: {payload['train_metrics'].get('Top10_overlap')}",
        f"- validation_method: {payload['validation_method']}",
        f"- val_MAPE: {(validation.get('metrics') or {}).get('MAPE')}",
        f"- val_Spearman: {(validation.get('metrics') or {}).get('Spearman')}",
        "",
        "Tiny MLP is skipped unless valid_samples >= 50.",
    ]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/latency_lut/full_engine_calibration_dataset_v2.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_latency_calibrator_v2.json")
    parser.add_argument("--report", default="outputs/latency_lut/full_engine_latency_calibration_v2_report.md")
    parser.add_argument("--model", choices=["auto", "least_squares_linear", "tiny_mlp"], default="auto")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_calibrator(
        args.dataset,
        args.output,
        args.report,
        model=args.model,
        ridge_alpha=args.ridge_alpha,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
