# Verification Checklist & Tests

## Automated Test Commands

```bash
# Full test suite (426 tests, fake LLM provider, no real API calls). Takes about 8 minutes on a laptop CPU.
# Run ALL of it before pushing. A subset passing says nothing about the rest.
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

# Read-only weight inspector
SBA_NO_DOTENV=1 python -m pytest tests/test_wsp.py -v

# Dry run, apply, protected examples, and the event stream the TUI reads
SBA_NO_DOTENV=1 python -m pytest tests/test_apply_and_events.py -v

# The terminal UI has no tests of its own; check that it still compiles
cd tui && npm ci && npm run build
```

On Windows PowerShell, set the variable first: `$env:SBA_NO_DOTENV="1"; python -m pytest tests/ -v`.

The same suite runs in CI on Linux and Windows (`.github/workflows/tests.yml`) for every pull request.

**Expected success indicator:** `X passed` with no `FAILED` or `ERROR` lines.

## Manual Verification Paths

### Path A: Static Scan
1. `python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py`
2. Expected: findings that include `missing_zero_grad` and `softmax_before_cross_entropy` (bug ids are lower-case)

### Path B: Fix with approval (on a copy — never on a file under `examples/`)
1. `cp examples/sample_scripts/benchmark_04_softmax_before_crossentropy.py my_train.py`
2. `python -m silent_bug_agent fix my_train.py --dry-run --no-llm`
   - Expected: a validated patch that removes the softmax; "Dry run: the file was not changed"; `my_train.py` is byte-for-byte unchanged
3. `python -m silent_bug_agent apply my_train.py`
   - Expected: "Patch ... applied"; `my_train.py.orig` holds the original
4. `python -m silent_bug_agent fix examples/sample_scripts/benchmark_04_softmax_before_crossentropy.py --no-llm`
   - Expected: the patch is shown, then "The file was not changed. This is one of the project's bundled examples..."; `git status --short examples/` prints nothing

### Path C: Weight inspection (read-only)
1. `python -m silent_bug_agent wsp examples/sample_scripts/benchmark_06_dead_relu.py`
   - Expected: `[CRITICAL] bias_collapse in fc1.bias`, "32 of 32 biases (100.0%) are at or below -15"; exit code 1
2. `python -m silent_bug_agent wsp examples/sample_scripts/clean_mnist.py`
   - Expected: "No weight anomalies found."; exit code 0

### Path D: Doctor Check
1. `python -m silent_bug_agent doctor` — offline check (no real LLM)
2. `python -m silent_bug_agent doctor --live` — requires real API keys in `.env`
3. Expected offline: all structural checks pass; Expected live: LLM provider responds within 5 s

### Path E: Benchmark Suite
1. `python examples/evaluate_pipeline.py`
2. Expected: repair rate reported per benchmark script; damage rate = 0 on clean scripts

## Last Run Results
- Date: 2026-11-15 | macOS (Apple M2, CPU only), Python 3.12.13, torch 2.15.0, pytest 9.2.0
- Command: `SBA_NO_DOTENV=1 python -m pytest tests/ -q` — the complete suite
```text
........................................................................ [ 16%]
........................................................................ [ 33%]
........................................................................ [ 50%]
........................................................................ [ 67%]
..................................................................s..... [ 84%]
..................................................................       [100%]
425 passed, 1 skipped in 455.92s (0:07:35)
```
- `git status --short examples/` after the run: no modified example.
- TUI: `npm ci && npm run build` exits 0 (Node 24).
- **Not yet run anywhere:** the suite on Windows or Linux (CI will do this on the first pull request), and on Python 3.14.

### History
- 2026-11-14, commit `fedacc1` on `main`: **15 failed, 394 passed, 1 skipped.** Five example files had been overwritten. The run recorded here before (2026-11-07, "6 passed" and "48 passed") was a subset, which is why the failures were not seen.

## Test Infrastructure Notes
- All tests use `SBA_NO_DOTENV=1` — real `.env` keys are NEVER loaded in tests.
- Fake LLM server speaks Groq/Cerebras protocol; started per test session in `conftest.py`.
- Multi-seed tests run real PyTorch training in isolated subprocesses — may take 2-5 min on CPU.
- `validation/multiseedagent.py` is the real Gate 2 implementation and is what `tests/test_multiseed_agent.py` tests. `multiseed_agent.py` is only an alias for it. (An earlier version of this page had them the wrong way round.)
- The files under `examples/` are test fixtures: tests rely on the bugs planted in them. `fix` and `apply` refuse to edit them.
