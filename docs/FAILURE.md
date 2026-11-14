# System Failure & Resilience Map

## Dependency Breakdown

### Dependency: Groq / Cerebras LLM API
- **Slow (Latency > 10 s)**: `semantic/client.py` raises `TimeoutError`; Supervisor logs and routes to rule-based fallback if available.
- **Down (5xx / connection refused)**: `semantic/router.py` attempts provider failover (Groq → Cerebras or vice versa); if both down, LLM-dependent agents emit empty results and pipeline continues with deterministic tools only.
- **Returning Corrupted JSON**: `agents/prompts.py` schema validation raises `ValidationError`; agent logs the raw response and returns an empty hypothesis/patch list; Supervisor retries up to `max_iterations`.
- **Rate Limited (429)**: Client applies exponential backoff (3 retries, 2× backoff); if exhausted, treated as Down.

### Dependency: ChromaDB (Vector Experience Store)
- **Slow (Query > 1 s)**: Few-shot retrieval times out; FixSynthesizerAgent proceeds with zero examples (zero-shot).
- **Down / Corrupted**: `learning/experience_store.py` catches `Exception`, logs, and returns empty list; pipeline continues without memory.
- **Empty (Cold Start)**: Normal condition; bandit uses uniform priors; LLM uses zero-shot prompting.

### Dependency: SQLite (Trajectory Store)
- **Slow**: Queries are fast (<10 ms expected); no timeout risk at current scale.
- **Corrupted / Locked**: `experience_store.py` catches `sqlite3.DatabaseError`; store fails open (logs error, skips write); trajectory is lost but pipeline does not crash.
- **Disk Full**: `sqlite3.OperationalError` caught; experience write skipped; warning logged.

### Dependency: PyTorch (Runtime Hooks)
- **Model has no `nn.Module` layers**: Hook attachment finds no targets; `RuntimeSession` returns empty telemetry; static analysis continues alone.
- **CUDA OOM during validation**: Subprocess exits with non-zero code; `script_runner.py` captures stderr; validation verdict: rejected with note "OOM".
- **Model uses `F.relu` instead of `nn.ReLU`**: Dead unit detection silently misses it (known gap; see CONSTRAINTS.md).

### Dependency: Subprocess (Multi-Seed Validation)
- **Seed subprocess crashes**: Exit code ≠ 0; that seed's metric is excluded from statistics; if fewer than 3 seeds complete, verdict is rejected with note "insufficient seeds".
- **Metric protocol absent** (script does not print `__METRICS__:{...}`): Validation falls back to loss curve comparison; warns user to add `__METRICS__` output.

## System Limits & Bottlenecks

| Limit | Value | Behavior at Limit |
|---|---|---|
| Repair loop retries | 3 (default `max_iterations`) | `AgentState.status = ABORTED` |
| Multi-seed count | 5 seeds default | Configurable; fewer seeds = less statistical power |
| Max patch size | No hard limit | Large diffs slow LLM generation and diff parsing |
| ChromaDB collection size | ~100K vectors (practical) | Retrieval degrades; run `meta_evolution` pruning |
| SQLite trajectory count | ~1M rows (practical) | No issue at current project scale |
| LLM context window | Model-dependent (~128K tokens) | Long source files may need chunking (not yet implemented) |

## Monitoring & Alerts

- **Structured logging**: All agents log to stdout with `[AGENT_NAME]` prefix; pipe to file for persistence.
- **Sentry integration**: Not yet integrated — open gap.
- **Prometheus metrics**: Not yet integrated — open gap.
- **Current alerting**: `runtime/alerts.py` prints real-time invariant violation alerts to stdout during `sba monitor`.
- **Pipeline abort signal**: `AgentState.status == ABORTED` with `abort_reason` string; CLI exits with code 1.
