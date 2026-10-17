"""Pinned binary judge with a deterministic offline implementation.

The class mirrors an LLM-as-a-judge boundary and loads the inspectable prompts.
For reproducibility in CI it uses a strict local parser/rubric rather than
requiring an API key.  A hosted judge can be added behind the same interface
without changing metric aggregation or report schemas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = {
    "a", "an", "and", "also", "are", "as", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


class LLMJudge:
    def __init__(self, prompts_path: str | Path | None = None, model: str = "local-deterministic-judge-v1") -> None:
        self.model = model
        path = Path(prompts_path) if prompts_path else ROOT / "prompts" / "judge_prompts.json"
        self.prompts: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))

    def score_correctness(self, generated_answer: str, reference_answer: str, contexts: list[str]) -> int:
        """Return exactly 0/1 using the reference answer, not real-world lookup."""

        expected = _tokens(reference_answer)
        actual = _tokens(generated_answer)
        return int(bool(expected) and expected.issubset(actual))

    def score_groundedness(self, generated_answer: str, contexts: list[str]) -> int:
        """Fail closed if any claim is absent from the supplied context."""

        context_tokens = _tokens(" ".join(contexts))
        claims = [claim.strip() for claim in re.split(r"[.!?]+", generated_answer) if claim.strip()]
        if not claims:
            return 0
        return int(all(_tokens(claim).issubset(context_tokens) for claim in claims))

    def score(self, generated_answer: str, reference_answer: str, contexts: list[str]) -> dict[str, int]:
        return {
            "correctness": self.score_correctness(generated_answer, reference_answer, contexts),
            "groundedness": self.score_groundedness(generated_answer, contexts),
        }
