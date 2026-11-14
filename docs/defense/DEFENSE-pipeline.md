# DEFENSE BRIEF: Silent Bug Detection & Repair Pipeline

- **Primary Purpose (2 sentences):** The system autonomously detects silent bugs in PyTorch training scripts — bugs that run without errors but cause models to silently underperform. It then synthesizes, empirically validates, and applies fixes through a five-stage multi-agent pipeline without requiring the user to understand the bug internals.

- **Execution Flow Summary:**
  1. User runs `sba fix <script.py>` → CLI builds `AgentState` and starts LangGraph graph.
  2. Detection: AST static rules (10 patterns) + PyTorch runtime hooks + LLM semantic review → unified `DetectionReport`.
  3. Diagnosis: Causal engine maps findings to 21 bug taxonomy entries via weighted symptom dampening → ranked `DiagnosticHypotheses`.
  4. Fix: Contextual MAB selects repair strategy → few-shot memory retrieval from ChromaDB → ReAct LLM generates unified diff `PatchCandidate`.
  5. Validation Gate 1: One-batch overfit oracle checks numeric sanity and gradient flow on patched script.
  6. Validation Gate 2: 5 isolated subprocesses run with different seeds → paired t-test + KS-test + Cohen's d → `ValidationVerdict`.
  7. If accepted: trajectory stored in SQLite + ChromaDB; bandit reward updated. If rejected: rejection reason injected into LLM prompt; retry up to 3 times.

- **Core Rationale (Why X over Y):**
  - LangGraph over Celery: explicit auditable state machine; no server infrastructure needed.
  - Subprocess isolation over in-process multiprocessing: prevents CUDA context leaks across seeds.
  - Dual-layer memory (ChromaDB + SQLite) over pgvector: zero infrastructure; ChromaDB for semantic similarity, SQLite for structured reward lookups.
  - Fake LLM server in tests over mocking: environment-agnostic; validates request shape; deterministic.

- **Data Structure & Storage:**
  - `AgentState` dataclass: single shared state contract across all LangGraph nodes.
  - `Finding` → `DiagnosticHypothesis` → `PatchCandidate` → `ValidationVerdict`: typed pipeline contract defined in `schemas.py`.
  - Experience store: ChromaDB (1536-dim embeddings of bug descriptions) + SQLite (full trajectory JSON + reward scalar).
  - Multi-seed metrics: communicated via `__METRICS__:{...}` stdout protocol from subprocess to validator.

- **Malicious Abuse Vectors & Controls:**
  - *Malicious training script*: Script runs in isolated subprocess with no additional sandboxing beyond OS process boundaries. Gap: no seccomp filter; malicious script has user-level filesystem access.
  - *Prompt injection via LLM response*: LLM output is strictly schema-validated; patch is applied to temp copy first; original modified only on `--apply`.
  - *Poisoned experience store*: ChromaDB + SQLite live in `~/.silent_bug_agent/`; require local user access; `meta_evolution.py` ablates low-quality trajectories.
  - *Path traversal via CLI args*: Path resolved to absolute; must be an existing `.py` file. Gap: symlinks not checked.

- **Failure Modes & Alerting:**
  - First failure point: LLM API down → agents fall back to deterministic tools only; pipeline continues with reduced capability.
  - Second failure point: Validation subprocess OOM → seed excluded; if <3 seeds complete, patch rejected.
  - Pipeline abort: `AgentState.status = ABORTED` after `max_iterations` retries; CLI exits code 1.
  - Current monitoring: stdout structured logging per agent. Gap: no Sentry, no Prometheus.

- **Future Improvements:**
  - Add OS-level sandbox (Linux `seccomp` / Windows AppContainer) to the validation subprocess.
  - Extend dead ReLU detection to `F.relu` and `F.leaky_relu` calls (not just `nn.ReLU` modules).
  - Add cryptographic signing of experience trajectories to prevent poisoning.
  - Implement `--fast` flag to skip Gate 2 (multi-seed) for time-constrained runs.
  - Integrate Sentry and Prometheus for production observability.
  - Resolve the `validation/multiseedagent.py` dead file (duplicate of `multiseed_agent.py`).
