================================================================================
Quick Start: Run Evaluation & Regression Harness
================================================================================

--------------------------------------------------------------------------------
STEP 1: Setup Environment
--------------------------------------------------------------------------------
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Expected Output:
# Successfully installed iniconfig packaging pluggy pytest-8.3.3


--------------------------------------------------------------------------------
STEP 2: Run End-to-End Evaluation
--------------------------------------------------------------------------------
# Evaluates Pipeline A (baseline) vs Pipeline B (candidate) across all 36 questions
python run_eval.py

# Expected Output:
# metric,baseline,candidate,delta,flag
# Recall,0.833,1.000,0.167,IMPROVE
# Correctness,1.000,1.000,0.000,NEUTRAL
# Groundedness,1.000,0.000,-1.000,REGRESS
# P95_Latency,3.166,16.171,13.005,REGRESS


--------------------------------------------------------------------------------
STEP 3: Run Contract Tests
--------------------------------------------------------------------------------
# Verifies dataset schema, pins, judge rubric, results schema, and regression flags
pytest -v tests/contract_test.py

# Expected Output:
# ============================== 8 passed in 0.01s ===============================


--------------------------------------------------------------------------------
STEP 4: View Regression Report & Artifacts
--------------------------------------------------------------------------------
# 4A. View the summary regression report table
cat results/regression_report.csv

# Expected Output:
# metric,baseline,candidate,delta,flag
# Recall,0.833333,1.000000,0.166667,IMPROVE
# Correctness,1.000000,1.000000,0.000000,NEUTRAL
# Groundedness,1.000000,0.000000,-1.000000,REGRESS
# P95_Latency,3.165754,16.171084,13.005330,REGRESS

# 4B. View sample per-question evaluation details (Pipeline B failure mode)
python -c "import json; print(json.dumps(json.load(open('results/eval_B_details.json'))[0], indent=2))"

# Expected Output:
# Shows id, generated answer with unsupported claim, retrieved titles, and scores:
# "recall": 1.0, "correctness": 1, "groundedness": 0


--------------------------------------------------------------------------------
STEP 5: (Optional) Run with Docker Compose
--------------------------------------------------------------------------------
# Prepares env, spins up Qdrant with healthcheck, and runs evaluator
cp .env.example .env
docker compose up --build

# Teardown when finished:
docker compose down -v


--------------------------------------------------------------------------------
All-in-One Execution (Single Command)
--------------------------------------------------------------------------------
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python run_eval.py && pytest -v tests/contract_test.py && cat results/regression_report.csv
================================================================================
