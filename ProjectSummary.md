# LLM Evaluation Regression Harness for RAG Pipelines: Project Summary

---

## 1. What This Project Does

This project is an **automated, reproducible evaluation and regression harness** for Retrieval-Augmented Generation (RAG) pipelines.

It compares two versions of a RAG pipeline against an identical benchmark of 36 multi-hop questions:
- **Baseline ([`PipelineA`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/pipelines.py#L94-L97))**: Represents the stable production system (fixed-size top-2 retrieval, concise factual answers).
- **Candidate ([`PipelineB`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/pipelines.py#L99-L102))**: Represents an experimental modification (expanded top-5 retrieval with tangential context).

The harness evaluates both pipelines across four independent dimensions (Recall, Correctness, Groundedness, and P95 Latency) and generates a structured, auditable regression report:

```text
metric,baseline,candidate,delta,flag
Recall,0.833,1.000,0.167,IMPROVE
Correctness,1.000,1.000,0.000,NEUTRAL
Groundedness,1.000,0.000,-1.000,REGRESS
P95_Latency,3.166,16.171,13.005,REGRESS
```

---

## 2. The Purpose: Why This Harness Exists

In typical RAG development, teams often measure only **Answer Accuracy / Correctness**. However, this creates a dangerous blind spot:
- A developer might increase retrieval `top_k` from 2 to 5 to boost recall.
- The model still outputs the correct answer, so **Correctness appears unchanged or improved**.
- **The Hidden Failure**: Because extra tangential documents were fed into the prompt, the model starts hallucinating or adding unsupported claims, and inference latency jumps dramatically.

**This harness solves that problem by strictly decoupling Groundedness from Correctness and Latency:**
1. A correct answer that contains unsupported claims scores **`Correctness = 1`** but **`Groundedness = 0`**.
2. Any latency jump greater than 10% is immediately caught and marked as **`REGRESS`**.
3. It prevents unsafe prompt or retrieval changes from being promoted to production.

---

## 3. How Each Functionality Is Implemented

### A. Pinned Benchmark Dataset & Knowledge Corpus
- **Files**: [`dataset/eval_set.json`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/dataset/eval_set.json) & [`src/data.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/data.py)
- **Implementation**:
  - Contains 36 multi-hop question records (HotpotQA-style). Each record stores `id`, `question`, `ground_truth_answer`, and `ground_truth_context_titles`.
  - [`src/data.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/data.py) provides a fixed corpus of 70 documents representing the mock vector database.
- **Reproducibility Guarantee**: [`config/eval_pins.json`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/config/eval_pins.json) pins the exact byte-level SHA-256 hash of the dataset so benchmarks cannot silently drift.

### B. Baseline Pipeline A (`Fixed-size Top-2`)
- **File**: [`src/pipelines.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/pipelines.py#L94-L97)
- **Implementation**:
  - Simulates top-2 chunk retrieval (`pipeline_a_top_k: 2`).
  - Generates concise, strictly factual answers directly matching the reference answer.
  - Achieves high groundedness (`1.0`), fast response latency (~2–3 ms), and ~83.3% average recall.

### C. Candidate Pipeline B (`Expanded Top-5 + Tangential Context`)
- **File**: [`src/pipelines.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/pipelines.py#L99-L102)
- **Implementation**:
  - Simulates an expanded top-k search with a reranker (`pipeline_b_top_k: 5`), retrieving all gold documents plus tangential documents (e.g., Apple, Microsoft, Earth).
  - Deliberately introduces an external, unsupported sentence to the answer:
    `"Ada Lovelace. The surrounding evidence also establishes an unrelated external claim."`
  - Demonstrates the failure mode: Recall improves to 100%, Correctness remains 1.0, but Groundedness collapses to 0.0 and latency jumps to ~16 ms.

### D. LLM-as-a-Judge Evaluation Logic
- **Files**: [`src/judge.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/judge.py) & [`prompts/judge_prompts.json`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/prompts/judge_prompts.json)
- **Implementation**:
  - [`score_correctness()`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/judge.py#L34-L40): Checks if the core answer tokens match the gold reference answer (binary `0` or `1`).
  - [`score_groundedness()`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/judge.py#L41-L49): Splits the generated answer into distinct sentences/claims and verifies each claim against the retrieved context tokens. **Fails closed (`0`)** if any claim cannot be verified directly from the retrieved context.

### E. Metric Aggregator & Direction-Aware Regression Detection
- **File**: [`src/metrics.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/metrics.py)
- **Implementation**:
  - Computes arithmetic means for quality metrics: Recall, Correctness, Groundedness.
  - Computes interpolated **P95 Latency** via [`percentile()`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/src/metrics.py#L15-L26).
  - Generates directional flags:
    - Quality metrics: `REGRESS` if candidate < baseline; `IMPROVE` if candidate > baseline.
    - Latency: `REGRESS` if candidate exceeds baseline by more than 10% (`candidate > baseline * 1.10`).

### F. Automated Contract Test Suite
- **File**: [`tests/contract_test.py`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/tests/contract_test.py)
- **Implementation**:
  - 8 contract checks that run in under 0.02s with `pytest`.
  - Asserts that dataset schemas, pins, prompt rubric constraints, detailed record JSONs, mathematical aggregates, and regression flags obey strict system invariant rules.

---

## 4. How Each Technology Used in This Project Helps

| Technology | Role in Project | How It Helps |
| :--- | :--- | :--- |
| **Python 3 Standard Library** | Core Execution Runtime | Enables fast, lightweight, and deterministic execution without bulky external ML dependencies or required GPU runtimes. Modules like `hashlib`, `statistics`, `dataclasses`, and `csv` power the entire engine. |
| **Pytest** | Automated Contract Testing | Serves as the CI/CD gatekeeper. Ensures that any code modification preserves dataset schemas, mathematical definitions of percentiles/deltas, and regression-flagging rules. |
| **Docker & Dockerfile** | Containerization | Packages the application into a reproducible Linux environment (`python:3.11-slim`), ensuring identical execution across macOS, Linux, and Windows machines without dependency drift. |
| **Docker Compose** | Multi-Service Orchestration | Coordinates the evaluation container alongside vector database dependencies, managing volume mounts (`.:/workspace`) so results generated inside containers land directly on the host machine. |
| **Qdrant Vector Database** | Production Integration Boundary | Provides a real-world vector database service container with an HTTP healthcheck (`http://localhost:6333/healthz`), documenting how production vector search engines connect into this evaluation harness. |
| **JSON & CSV** | Audit & Reporting Formats | Produces question-level traces ([`eval_A_details.json`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/results/eval_A_details.json), [`eval_B_details.json`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/results/eval_B_details.json)) and a tabular regression report ([`regression_report.csv`](file:///home/shamshu/GPP/Week38/LLM-Evaluation-Regression-Harness-for-RAG-Pipelines/results/regression_report.csv)) for easy human inspection and integration with CI/CD dashboards. |
