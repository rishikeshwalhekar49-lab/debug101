# DEFENSE BRIEF: Silent Bug Detection & Repair Pipeline

- **Primary Purpose (2 sentences):** The system autonomously detects silent bugs in PyTorch training scripts — bugs that run without errors but cause models to silently underperform. It then synthesizes, empirically validates, and applies fixes through a five-stage multi-agent pipeline without requiring the user to understand the bug internals.

- **Execution Flow Summary:**
  1. User runs `sba fix <script.py>` → CLI builds `AgentState` and starts LangGraph graph.
  2. Detection: AST static rules (10 patterns) + PyTorch runtime hooks + LLM semantic review → unified `DetectionReport`.
  3. Diagnosis: Causal engine maps findings to 21 bug taxonomy entries via weighted symptom dampening → ranked `DiagnosticHypotheses`.
  4. Fix: Contextual MAB selects repair strategy → few-shot memory retrieval from ChromaDB → ReAct LLM generates unified diff `PatchCandidate`.
  5. Validation Gate 1: One-batch overfit oracle checks numeric sanity and gradient flow on patched script.
  6. Validation Gate 2: baseline and patched runs on 3 to 5 seeds in separate subprocesses → paired t-test + KS-test + Cohen's d → `ValidationVerdict`.
  7. Every attempt is stored in SQLite and the bandit reward updated; accepted repairs are also embedded in ChromaDB. If rejected: the rejection reason goes into the next prompt; up to 3 attempts.
  8. Separate from repair: `wsp` gives a read-only report on the trained weights. It never edits anything.

- **Core Rationale (Why X over Y):**
  - LangGraph over Celery: explicit auditable state machine; no server infrastructure needed.
  - Subprocess isolation over in-process multiprocessing: prevents CUDA context leaks across seeds.
  - Dual-layer memory (ChromaDB + SQLite) over pgvector: zero infrastructure; ChromaDB for semantic similarity, SQLite for structured reward lookups.
  - Fake LLM server in tests over mocking: environment-agnostic; validates request shape; deterministic.

- **Data Structure & Storage:**
  - `AgentState` dataclass: single shared state contract across all LangGraph nodes.
  - `Finding` → `DiagnosticHypothesis` → `PatchCandidate` → `ValidationVerdict`: typed pipeline contract defined in `schemas.py`.
  - Experience store: ChromaDB (384-dimension `all-MiniLM-L6-v2` embeddings of accepted repairs) + SQLite (`experience.sqlite3`: every attempt, bandit statistics, lessons).
  - Multi-seed metrics: communicated via `__METRICS__:{...}` stdout protocol from subprocess to validator.

- **Malicious Abuse Vectors & Controls:**
  - *Malicious training script*: Script runs in isolated subprocess with no additional sandboxing beyond OS process boundaries. Gap: no seccomp filter; malicious script has user-level filesystem access.
  - *Prompt injection via LLM response*: LLM output is schema-validated; `patch_guard` rejects patches that add process, network or file-deletion calls or change more than 40 lines; the patch runs on a temp copy first; the original is written only after both gates pass (or, with `fix --dry-run`, only when the user runs `apply`). There is no `--apply` flag.
  - *Poisoned experience store*: ChromaDB + SQLite live in `~/.silent_bug_agent/`; require local user access; `meta_evolution.py` ablates low-quality trajectories.
  - *Path traversal via CLI args*: Path resolved to absolute and must be an existing file. Gap: the extension is not checked and symlinks are followed.

- **Failure Modes & Alerting:**
  - First failure point: LLM API down → agents fall back to deterministic tools only; pipeline continues with reduced capability.
  - Second failure point: a validation subprocess crashes or runs out of memory → the patch is rejected. One failed seed is enough; seeds are not dropped.
  - Budget spent: `AgentState.status = FAILED` with the last rejection reason; CLI exits code 1.
  - Current monitoring: the CLI report, `--json`, and `--events` (NDJSON). Gap: no structured logging, Sentry or Prometheus.

- **Future Improvements:**
  - Add OS-level sandbox (Linux `seccomp` / Windows AppContainer) to the validation subprocess.
  - Extend dead ReLU detection to `F.relu` and `F.leaky_relu` calls (not just `nn.ReLU` modules).
  - Add cryptographic signing of experience trajectories to prevent poisoning.
  - Implement `--fast` flag to skip Gate 2 (multi-seed) for time-constrained runs.
  - Integrate Sentry and Prometheus for production observability.
  - Measure the live LLM repair and damage rates (`examples/evaluate_pipeline.py` with real keys); only the offline rates have been measured.
  - Evaluate on real training scripts, not only the eight synthetic benchmarks.
