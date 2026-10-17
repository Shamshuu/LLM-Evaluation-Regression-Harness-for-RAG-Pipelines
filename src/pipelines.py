"""Two deliberately different, deterministic RAG pipeline configurations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import DOCUMENTS, context_for


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    retrieved_contexts: list[str]
    context_titles: list[str]
    latency_ms: float


class RAGPipeline:
    """A small RAG interface with a baseline and a regression candidate.

    The local fixture stands in for a vector database.  The important behavior
    is still visible: both variants receive the same question and corpus, A
    uses fixed-size/top-2 retrieval, while B uses top-5 retrieval plus a
    reranker-like expansion that deliberately adds tangential context.
    """

    def __init__(self, config_path: str | Path | None = None, variant: str = "A") -> None:
        self.variant = variant.upper()
        if self.variant not in {"A", "B"}:
            raise ValueError("variant must be 'A' or 'B'")
        dataset_path = ROOT / "dataset" / "eval_set.json"
        self.records: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
        self._by_question = {record["question"]: record for record in self.records}
        self._position = {record["id"]: index for index, record in enumerate(self.records)}
        self.config_path = Path(config_path) if config_path else ROOT / "config" / "eval_pins.json"

    def retrieve_and_generate(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            record = self._by_question[question]
        except KeyError as exc:
            raise ValueError(f"Question is not present in the pinned evaluation set: {question}") from exc

        gold_titles = list(record["ground_truth_context_titles"])
        position = self._position[record["id"]]
        if self.variant == "A":
            # Fixed-size chunks/top-2.  Every third item intentionally misses
            # one supporting title, while the answer remains concise and true.
            if position % 3:
                titles = gold_titles
            else:
                # A can answer from one relevant chunk, but that chunk is not
                # always the first labelled title in the dataset.
                answer_lower = record["ground_truth_answer"].lower()
                answer_title = next(
                    (title for title in gold_titles if answer_lower in context_for(title).lower()),
                    gold_titles[0],
                )
                titles = [answer_title]
            answer = record["ground_truth_answer"]
            time.sleep(0.002 + (position % 4) * 0.0003)
        else:
            # Candidate's expanded top-k/reranker path retrieves both labels
            # and three plausible-but-tangential documents.
            titles = gold_titles[:]
            for candidate in DOCUMENTS:
                if candidate not in titles:
                    titles.append(candidate)
                if len(titles) == 5:
                    break
            answer = (
                f"{record['ground_truth_answer']}. "
                "The surrounding evidence also establishes an unrelated external claim."
            )
            time.sleep(0.012 + (position % 5) * 0.001)

        contexts = [context_for(title) for title in titles]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "context_titles": titles,
            "latency_ms": latency_ms,
        }


class PipelineA(RAGPipeline):
    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__(config_path=config_path, variant="A")


class PipelineB(RAGPipeline):
    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__(config_path=config_path, variant="B")
