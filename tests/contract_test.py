"""Contract checks for the required evaluation artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/eval_set.json"
PINS = ROOT / "config/eval_pins.json"
PROMPTS = ROOT / "prompts/judge_prompts.json"
REPORT = ROOT / "results/regression_report.csv"
METRICS = ("recall", "correctness", "groundedness", "latency_ms")
REPORT_METRICS = ("Recall", "Correctness", "Groundedness", "P95_Latency")


def load_json(path: Path):
    assert path.is_file(), f"missing {path.relative_to(ROOT)}"
    return json.loads(path.read_text(encoding="utf-8"))


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def details(name: str) -> list[dict]:
    records = load_json(ROOT / f"results/eval_{name}_details.json")
    assert isinstance(records, list)
    dataset_ids = {row["id"] for row in load_json(DATASET)}
    assert {row["id"] for row in records} == dataset_ids
    assert len(records) == len(dataset_ids)
    for row in records:
        assert isinstance(row["generated_answer"], str)
        assert isinstance(row["retrieved_context_titles"], list)
        assert set(row["metrics"]) == set(METRICS)
        assert 0 <= row["metrics"]["recall"] <= 1
        assert row["metrics"]["correctness"] in (0, 1)
        assert row["metrics"]["groundedness"] in (0, 1)
        assert isinstance(row["metrics"]["latency_ms"], (int, float))
        assert math.isfinite(row["metrics"]["latency_ms"])
    return records


def test_dataset_schema_and_size():
    records = load_json(DATASET)
    assert isinstance(records, list) and 30 <= len(records) <= 50
    assert len({row["id"] for row in records}) == len(records)
    for row in records:
        assert set(("id", "question", "ground_truth_answer", "ground_truth_context_titles")) <= set(row)
        assert isinstance(row["ground_truth_context_titles"], list) and row["ground_truth_context_titles"]


def test_pins_have_required_typed_values():
    pins = load_json(PINS)
    assert isinstance(pins["dataset_size"], int)
    for key in ("llm_judge_model", "pipeline_a_embedder", "pipeline_b_embedder"):
        assert isinstance(pins[key], str) and pins[key]


def test_prompts_are_separate_and_strict():
    prompts = load_json(PROMPTS)
    assert all(isinstance(prompts[key], str) for key in ("correctness_prompt", "groundedness_prompt"))
    text = prompts["groundedness_prompt"].lower()
    assert "context" in text and ("only" in text or "strictly" in text)
    assert re.search(r"score\s+0", text)
    assert any(term in text for term in ("outside", "external", "real-world", "prior"))


def test_detail_files_have_exact_nested_metric_schema():
    details("A")
    details("B")


def test_report_schema_and_recomputed_aggregates():
    rows_a, rows_b = details("A"), details("B")
    with REPORT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["metric", "baseline", "candidate", "delta", "flag"]
        rows = list(reader)
    assert len(rows) == 4
    assert {row["metric"] for row in rows} == set(REPORT_METRICS)
    aggregates = {}
    for metric in METRICS[:-1]:
        aggregates[metric] = (
            fmean(row["metrics"][metric] for row in rows_a),
            fmean(row["metrics"][metric] for row in rows_b),
        )
    aggregates["latency_ms"] = (
        p95([row["metrics"]["latency_ms"] for row in rows_a]),
        p95([row["metrics"]["latency_ms"] for row in rows_b]),
    )
    names = dict(zip(REPORT_METRICS, ("recall", "correctness", "groundedness", "latency_ms")))
    for row in rows:
        baseline = float(row["baseline"])
        candidate = float(row["candidate"])
        assert baseline == pytest.approx(aggregates[names[row["metric"]]][0], abs=5e-3)
        assert candidate == pytest.approx(aggregates[names[row["metric"]]][1], abs=5e-3)
        assert float(row["delta"]) == pytest.approx(candidate - baseline, abs=5e-3)
        expected = candidate > baseline * 1.10 if row["metric"] == "P95_Latency" else candidate < baseline
        assert row["flag"] == ("REGRESS" if expected else "IMPROVE" if candidate > baseline else "NEUTRAL")


def test_candidate_demonstrates_intentional_regression():
    with REPORT.open(newline="", encoding="utf-8") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert float(rows["Correctness"]["delta"]) >= 0
    assert float(rows["Groundedness"]["delta"]) < 0 or float(rows["P95_Latency"]["delta"]) > 0
    assert any(row["flag"] == "REGRESS" for row in rows.values())


def test_compose_has_vector_service_and_healthcheck():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert re.search(r"qdrant|vector|weaviate|chroma|pgvector", text, re.I)
    assert re.search(r"healthcheck:\s*\n", text)


def test_env_example_documents_safe_api_and_vector_settings():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\w*API_KEY\w*=", text)
    assert re.search(r"(?m)^\w*(VECTOR|QDRANT)\w*(HOST|PORT|URL)\w*=", text)
    assert not re.search(r"sk-[A-Za-z0-9_-]{20,}", text)
