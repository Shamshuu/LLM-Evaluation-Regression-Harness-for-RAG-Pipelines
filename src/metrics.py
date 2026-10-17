"""Metric aggregation and regression report generation."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


QUALITY_METRICS = ("recall", "correctness", "groundedness")
LATENCY_REGRESSION_FACTOR = 1.10


def percentile(values: Iterable[float], percentile_rank: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty collection")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_rank / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "recall": fmean(float(row["metrics"]["recall"]) for row in results),
        "correctness": fmean(float(row["metrics"]["correctness"]) for row in results),
        "groundedness": fmean(float(row["metrics"]["groundedness"]) for row in results),
        "latency_ms": percentile((float(row["metrics"]["latency_ms"]) for row in results), 95),
    }


def generate_regression_report(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    output_path: str | Path = "results/regression_report.csv",
) -> list[dict[str, Any]]:
    """Write the four-row before/after/delta report and return its rows."""

    if len(results_a) != len(results_b) or not results_a:
        raise ValueError("baseline and candidate results must be equally sized and non-empty")
    baseline = aggregate_results(results_a)
    candidate = aggregate_results(results_b)
    metric_names = ("Recall", "Correctness", "Groundedness", "P95_Latency")
    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        key = "latency_ms" if metric == "P95_Latency" else metric.lower()
        before = baseline[key]
        after = candidate[key]
        delta = after - before
        regression = after > before * LATENCY_REGRESSION_FACTOR if metric == "P95_Latency" else after < before
        flag = "REGRESS" if regression else "IMPROVE" if delta > 0 else "NEUTRAL"
        rows.append({"metric": metric, "baseline": before, "candidate": after, "delta": delta, "flag": flag})

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "baseline", "candidate", "delta", "flag"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "baseline": f"{row['baseline']:.6f}", "candidate": f"{row['candidate']:.6f}", "delta": f"{row['delta']:.6f}"})
    return rows
