# Verification Checklist & Tests

## Automated Test Commands

```bash
# Full test suite (404 tests, fake LLM provider, no real API calls)
SBA_NO_DOTENV=1 python -m pytest tests/ -v

# LLM client, model router, provider fallback
SBA_NO_DOTENV=1 python -m pytest tests/test_llm_client.py -v

# MetaEvolutionAgent and experience memory
SBA_NO_DOTENV=1 python -m pytest tests/test_memory_and_meta.py -v

# Multi-seed & sandbox validation
SBA_NO_DOTENV=1 python -m pytest tests/test_multiseed_agent.py -v

# Runtime Monitor (Module 1)
SBA_NO_DOTENV=1 python -m pytest tests/test_runtime_monitor_agent.py -v

# Full pipeline (slow — real training runs with fake LLM)
SBA_NO_DOTENV=1 python -m pytest tests/test_supervisor_pipeline.py -v

# Weight-Space Programming (WSP) unit tests
SBA_NO_DOTENV=1 python -m pytest tests/test_wsp.py -v
```

**Expected success indicator:** `X passed` with no `FAILED` or `ERROR` lines.

## Manual Verification Paths

### Path A: Static Scan
1. `python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py`
2. Expected: at least findings for `BUG_MISSING_ZERO_GRAD`, `BUG_SOFTMAX_BEFORE_CE`, or `BUG_TRAIN_EVAL_MODE`

### Path B: One-Shot Fix
1. `python -m silent_bug_agent fix examples/sample_scripts/buggy_mnist.py`
2. Expected: pipeline runs through all 5 stages; at least one patch accepted; diff printed to stdout

### Path C: Weight-Space Programming (WSP)
1. `python -m silent_bug_agent wsp examples/sample_scripts/benchmark_06_dead_relu.py --dry-run --yes`
2. Expected: simultaneous code & weight perception, dual solution synthesis, Gate 1 & Gate 2 validation passing

### Path D: Doctor Check
1. `python -m silent_bug_agent doctor` — offline check (no real LLM)
2. `python -m silent_bug_agent doctor --live` — requires real API keys in `.env`
3. Expected offline: all structural checks pass; Expected live: LLM provider responds within 5 s

### Path E: Benchmark Suite
1. `python examples/evaluate_pipeline.py`
2. Expected: repair rate reported per benchmark script; damage rate = 0 on clean scripts

## Last Run Results
- Date: 2026-11-07 | Model: Gemini 3.8 Flash
- Output Proof:
```text
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\Projects\DL\v2
configfile: pyproject.toml
collected 6 items

tests/test_wsp.py::test_inspector_detects_bias_collapse PASSED           [ 16%]
tests/test_wsp.py::test_inspector_detects_rank_collapse PASSED           [ 33%]
tests/test_wsp.py::test_inspector_detects_weight_outliers PASSED         [ 50%]
tests/test_editor_bias_center_patch PASSED                               [ 66%]
tests/test_editor_subspace_excision PASSED                               [ 83%]
tests/test_editor_outlier_clamp PASSED                                   [100%]

======================= 6 passed in 6.78s ========================
Full Regression (WSP + CLI + Static + Schemas): 48 passed in 5.61s
```

## Test Infrastructure Notes
- All tests use `SBA_NO_DOTENV=1` — real `.env` keys are NEVER loaded in tests.
- Fake LLM server speaks Groq/Cerebras protocol; started per test session in `conftest.py`.
- Multi-seed tests run real PyTorch training in isolated subprocesses — may take 2-5 min on CPU.
- `validation/multiseedagent.py` is a dead file (duplicate of `multiseed_agent.py`) — do not add tests for it.
