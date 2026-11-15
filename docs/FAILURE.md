# System Failure & Resilience Map

Checked against the code on 2026-11-15. Where a failure is **not** handled, this page says so.

## Dependency Breakdown

### Groq / Cerebras LLM API (`semantic/client.py`, `semantic/router.py`)
- **Slow**: each request has a 90-second timeout. A timeout is retried up to 2 times (waiting 2 s, then 4 s) and then raises `LLMError`.
- **Rate limited (429)**: retried only if the server's `retry-after` is 20 seconds or less, at most 2 times. Otherwise the request fails over at once.
- **Server error (5xx) or connection error**: retried up to 2 times, then `LLMError`.
- **Failover**: for each task the router tries its models in order (Groq first, then Cerebras). If all fail it raises `LLMUnavailable`, and the agent that asked falls back to its rule-based path. The run report lists the failed calls.
- **Malformed JSON**: the client first asks for a strict JSON schema, then retries in plain JSON mode, and extracts the first JSON object from the reply. If that still fails it is an `LLMError`, handled as above.
- **No key at all**: every agent runs its deterministic path. The report says "offline".

### ChromaDB (semantic recall)
- **Missing, broken, or its embedding model cannot be downloaded**: recall falls back to word overlap over the SQLite rows, and a note is added to the report. It does not return an empty list.
- **Cold start**: no examples; the fix agent works zero-shot and the bandit has no history.
- There is no query timeout.

### SQLite (repair history, bandit, lessons)
- **The file cannot be opened or its schema cannot be created** (no permission, read-only folder): the store switches to an in-memory database for this run and adds a note. Nothing learned in that run is saved. Tested in `tests/test_memory_and_meta.py`.
- **Locked**: SQLite waits up to 10 seconds.
- **Corrupted, or disk full during a later write**: **not handled.** The `sqlite3` error propagates and the command fails. Delete `~/.silent_bug_agent/experience.sqlite3` to start again.

### PyTorch (runtime hooks)
- **A hook raises**: the exception is caught, monitoring switches itself off for the rest of the run, and the script continues.
- **No model or no optimizer step**: no runtime findings; static analysis still reports.
- **`F.relu` instead of `nn.ReLU`**: dead units are not measured (known gap).
- **The script itself crashes**: `fix` stops with outcome `script_error` and the traceback tail. Nothing is patched.

### Validation subprocesses (`validation/sandbox.py`, `multiseedagent.py`)
- **A patched run crashes, times out, or yields no finite loss on any seed**: the patch is rejected. One failed seed is enough; seeds are not dropped from the statistics.
- **The unpatched script itself fails or gives no finite loss on some seed** (for example it diverges to NaN): there is nothing to test against. If the patched script trains on every seed, the patch is accepted on that ground, the route is recorded as `baseline_failed`, and no p-value or effect size is reported.
- **No `__METRICS__` line**: the loss is taken from the values passed to `backward()`, which `script_runner.py` records itself. The script does not need to cooperate.
- **Out of memory**: the child process dies and that run counts as a failure.

### Weight inspection (`wsp`)
- **The script makes no optimizer step, crashes, or times out**: no weights are captured; the report says why and the exit code is 2.
- **A checkpoint that needs unpickling**: refused, with the reason.

### Terminal UI
- **The Python process dies without a final event**: the UI shows the last lines of its error output instead of waiting forever.
- **Python cannot be started**: reported as an error on screen.

## System Limits

| Limit | Value | Behaviour at the limit |
|---|---|---|
| Repair attempts per round | 3 (`--max-iter`) | status `failed`, with the last rejection reason; exit code 1 |
| Bugs fixed per `fix` command | 3 (`--max-rounds`) | stops; run again for more |
| Attempts per bug | 2 | the next hypothesis is tried |
| Patch size | 40 changed lines | larger patches are rejected before validation |
| Seeds for Gate 2 | 3 to 5 (`--seeds`) | fewer seeds means less statistical power |
| Per-run timeout | 180 s | the run counts as failed |
| Rank check in `wsp` | matrices up to 1024 on a side | larger ones are skipped, with a note |
| Weights captured by `wsp` | 50 million values | larger models are not written; use `--checkpoint` |

## Monitoring & Alerts
- There is no structured logging, Sentry or Prometheus integration.
- The CLI prints a report at the end of each command. `--json` gives the same data as JSON, and `--events` streams NDJSON events as the pipeline runs.
- `monitor` prints early alerts to stderr while the script is still running.
- A failed `fix` exits with code 1 and prints the reason.
