# RAG Evaluation Regression Harness

This repository is a small, reproducible evaluation harness for comparing two
retrieval-augmented generation configurations on the same pinned evaluation
set. It reports retrieval recall, answer correctness, groundedness, and
end-to-end p95 latency independently so an accuracy improvement cannot hide a
groundedness or latency regression.

## What is included

- `dataset/eval_set.json` — 36 fixed HotpotQA-style multi-hop records with two
  labelled supporting document titles each.
- `src/pipelines.py` — Pipeline A uses fixed-size/top-2 retrieval. Pipeline B
  simulates a reranker and top-5 expansion, adding tangential context and an
  unsupported extra claim by design.
- `src/judge.py` — a pinned, deterministic local judge with separate binary
  correctness and groundedness decisions. It uses the inspectable prompts in
  `prompts/judge_prompts.json` and needs no API key.
- `src/metrics.py` — independent averages for quality metrics and interpolated
  p95 for latency, with direction-aware regression flags.
- `results/eval_A_details.json` and `results/eval_B_details.json` — auditable
  question-level outputs, retrieved titles/context, and raw metric values.
- `results/regression_report.csv` — the before/after/delta table.

The candidate is intentionally engineered to demonstrate the failure mode:
correctness remains at least as good as baseline, while groundedness falls
because its answer contains an unsupported external claim. Its expanded
retrieval also increases latency.

## Local run

The only runtime code dependency is Python 3.11+ standard library. To run the
contract tests, install the pinned development dependency first:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_eval.py
pytest -q tests/contract_test.py
```

The command rewrites these reproducible artifacts:

```text
results/eval_A_details.json
results/eval_B_details.json
results/regression_report.csv
```

It also records the SHA-256 of the exact dataset bytes in
`config/eval_pins.json`. The judge model, embedder IDs, chunker version,
top-k values, and temperature are all pinned there.

## Compose run

Compose provisions a Qdrant vector database with a healthcheck before running
the evaluator image:

```bash
cp .env.example .env       # optional; values are placeholders
docker compose up --build
```

The fixture implementation is intentionally offline and does not require a
live Qdrant or hosted LLM to make the report reproducible. Qdrant is included
as the documented vector-database integration boundary required for a real
deployment; replacing the fixture retrieval adapter with a Qdrant client does
not change the evaluator or report contract.

## Report semantics

`results/regression_report.csv` has exactly these columns:

| metric | baseline | candidate | delta | flag |
| --- | ---: | ---: | ---: | --- |
| `Recall` | mean A | mean B | B - A | quality regression if B < A |
| `Correctness` | mean A | mean B | B - A | quality regression if B < A |
| `Groundedness` | mean A | mean B | B - A | quality regression if B < A |
| `P95_Latency` | p95 A | p95 B | B - A | regression if B > A × 1.10 |

`REGRESS`, `IMPROVE`, and `NEUTRAL` are computed from the values in the CSV,
not written as a prose conclusion. A correct answer with an unsupported extra
sentence is therefore correctness `1` and groundedness `0`.
