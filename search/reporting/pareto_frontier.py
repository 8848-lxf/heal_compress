"""Official full-validation Pareto fronts and plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _official_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("evaluation_protocol", "")) == "full_validation"
        and bool(row.get("full_validation_success", False))
    ]


def official_pareto_front(
    rows: Sequence[Mapping[str, Any]],
    *,
    resource_key: str,
    map_key: str = "mAP",
) -> list[dict[str, Any]]:
    """Maximize mAP and minimize one measured resource."""

    if "latency" in resource_key.lower() and resource_key != "formal_latency_p50_ms":
        raise ValueError("official_latency_requires_formal_real_latency")
    valid = _official_rows(rows)
    for row in valid:
        if not math.isfinite(float(row.get(map_key, float("nan")))):
            raise ValueError(f"official_pareto_map_nonfinite:{row.get('candidate_id', '')}")
        if not math.isfinite(float(row.get(resource_key, float("nan")))):
            raise ValueError(
                f"official_pareto_resource_nonfinite:{resource_key}:{row.get('candidate_id', '')}"
            )
    front = []
    for candidate in valid:
        resource = float(candidate[resource_key])
        map_value = float(candidate[map_key])
        dominated = any(
            float(other[resource_key]) <= resource
            and float(other[map_key]) >= map_value
            and (
                float(other[resource_key]) < resource
                or float(other[map_key]) > map_value
            )
            for other in valid
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda row: (
            float(row[resource_key]),
            -float(row[map_key]),
            str(row.get("candidate_id", "")),
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_front(
    all_rows: Sequence[Mapping[str, Any]],
    front: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    x_label: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    official = _official_rows(all_rows)
    front_ids = {str(row.get("candidate_id", "")) for row in front}
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    scatter = axis.scatter(
        [float(row[x_key]) for row in official],
        [float(row["mAP"]) for row in official],
        c=[float(row["formal_latency_p50_ms"]) for row in official],
        cmap="viridis",
        alpha=0.72,
        edgecolors=[
            "black" if str(row.get("candidate_id", "")) in front_ids else "none"
            for row in official
        ],
        linewidths=1.0,
    )
    if front:
        axis.plot(
            [float(row[x_key]) for row in front],
            [float(row["mAP"]) for row in front],
            color="black",
            linewidth=1.0,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("mAP (full validation)")
    axis.grid(True, alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="formal p50 latency (ms)")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)


def _plot_combined(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    x_label: str,
    front: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    official = _official_rows(rows)
    front_ids = {str(row.get("candidate_id", "")) for row in front}
    marker_by_source = {
        "anchor": "s",
        "baseline": "^",
        "ga": "o",
        "greedy": "D",
    }
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    scatter = None
    sources = sorted(
        {str(row.get("candidate_source", "ga")).lower() for row in official}
    )
    for source in sources:
        subset = [
            row
            for row in official
            if str(row.get("candidate_source", "ga")).lower() == source
        ]
        scatter = axis.scatter(
            [float(row[x_key]) for row in subset],
            [float(row["mAP"]) for row in subset],
            c=[float(row["formal_latency_p50_ms"]) for row in subset],
            cmap="viridis",
            vmin=min(float(row["formal_latency_p50_ms"]) for row in official),
            vmax=max(float(row["formal_latency_p50_ms"]) for row in official),
            marker=marker_by_source.get(source, "o"),
            label=source,
            alpha=0.78,
            edgecolors=[
                "black"
                if str(row.get("candidate_id", "")) in front_ids
                else "none"
                for row in subset
            ],
            linewidths=1.0,
        )
    for row in front:
        axis.annotate(
            str(row.get("candidate_id", "")),
            (float(row[x_key]), float(row["mAP"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("mAP (full validation)")
    axis.grid(True, alpha=0.25)
    if sources:
        axis.legend(title="candidate source")
    if scatter is not None:
        figure.colorbar(scatter, ax=axis, label="formal p50 latency (ms)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_official_pareto_artifacts(
    rows: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    destination = Path(output_dir)
    specs = {
        "bops": ("R_BOPS", "BOPS retention", "pareto_map_vs_bops_retention"),
        "param": ("R_param", "Physical parameter retention", "pareto_map_vs_param_retention"),
        "latency": ("formal_latency_p50_ms", "Formal p50 latency (ms)", "pareto_map_vs_latency_p50"),
    }
    result: dict[str, Any] = {}
    for name, (resource_key, label, stem) in specs.items():
        front = official_pareto_front(rows, resource_key=resource_key)
        csv_path = destination / f"{stem}.csv"
        png_path = destination / f"{stem}.png"
        pdf_path = destination / f"{stem}.pdf"
        _write_csv(csv_path, front)
        _plot_front(
            rows,
            front,
            x_key=resource_key,
            x_label=label,
            png_path=png_path,
            pdf_path=pdf_path,
        )
        result[name] = {
            "point_count": len(front),
            "csv": str(csv_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
        }
    combined_bops = destination / "pareto_combined_bops_map_latency.png"
    combined_param = destination / "pareto_combined_param_map_latency.png"
    _plot_combined(
        rows,
        x_key="R_BOPS",
        x_label="BOPS retention",
        front=official_pareto_front(rows, resource_key="R_BOPS"),
        output_path=combined_bops,
    )
    _plot_combined(
        rows,
        x_key="R_param",
        x_label="Physical parameter retention",
        front=official_pareto_front(rows, resource_key="R_param"),
        output_path=combined_param,
    )
    result["combined"] = {
        "bops_map_latency_png": str(combined_bops),
        "param_map_latency_png": str(combined_param),
    }
    return result
