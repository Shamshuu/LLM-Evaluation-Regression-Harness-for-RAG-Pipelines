"""CLI entry point for the reproducible two-pipeline evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .judge import LLMJudge
from .metrics import generate_regression_report
from .pipelines import PipelineA, PipelineB


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "eval_set.json"


def dataset_sha256() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def _recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    return len(set(retrieved_titles).intersection(gold_titles)) / len(gold_titles)


def evaluate_pipeline(pipeline: Any, dataset: list[dict[str, Any]], judge: LLMJudge) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for record in dataset:
        output = pipeline.retrieve_and_generate(record["question"])
        scores = judge.score(
            generated_answer=output["answer"],
            reference_answer=record["ground_truth_answer"],
            contexts=output["retrieved_contexts"],
        )
        details.append(
            {
                "id": record["id"],
                "generated_answer": output["answer"],
                "retrieved_context_titles": output["context_titles"],
                "retrieved_contexts": output["retrieved_contexts"],
                "metrics": {
                    "recall": _recall(output["context_titles"], record["ground_truth_context_titles"]),
                    "correctness": scores["correctness"],
                    "groundedness": scores["groundedness"],
                    "latency_ms": output["latency_ms"],
                },
            }
        )
    return details


def run_evaluation() -> list[dict[str, Any]]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    expected_size = json.loads((ROOT / "config" / "eval_pins.json").read_text(encoding="utf-8"))["dataset_size"]
    if len(dataset) != expected_size:
        raise RuntimeError(f"dataset has {len(dataset)} rows but config pins {expected_size}")

    judge = LLMJudge()
    results_a = evaluate_pipeline(PipelineA(), dataset, judge)
    results_b = evaluate_pipeline(PipelineB(), dataset, judge)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "eval_A_details.json").write_text(json.dumps(results_a, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results" / "eval_B_details.json").write_text(json.dumps(results_b, indent=2) + "\n", encoding="utf-8")
    rows = generate_regression_report(results_a, results_b, ROOT / "results" / "regression_report.csv")

    pins_path = ROOT / "config" / "eval_pins.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    pins["dataset_sha256"] = dataset_sha256()
    pins_path.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    rows = run_evaluation()
    print("metric,baseline,candidate,delta,flag")
    for row in rows:
        print(
            f"{row['metric']},{row['baseline']:.3f},{row['candidate']:.3f},"
            f"{row['delta']:.3f},{row['flag']}"
        )


if __name__ == "__main__":
    main()
