# Architecture Decisions

## [D-001]: LangGraph StateGraph as the Supervisor Orchestrator
- Date: 2026-11-04 | Model: Claude Sonnet 4.6 Thinking
- Context: We need a durable, inspectable orchestrator that can route between 7+ specialist agents with conditional edges, retry loops, and clean state management.
- Options Considered:
  1) LangGraph StateGraph — explicit node/edge graph with typed shared state
  2) Celery task queue — distributed worker model
  3) Plain Python async coroutines — no framework overhead
- Chosen Option: 1 — LangGraph StateGraph
- Why: Explicit graph edges make control flow auditable; `AgentState` dataclass is a single-source-of-truth contract; LangGraph natively supports ReAct loops and human-in-the-loop pauses.
- Trade-offs: Adds LangGraph dependency; LangGraph version pinning can lag behind LangChain core.
- Revisit if: LangGraph introduces breaking API changes or the graph grows beyond 15 nodes and compile time becomes a bottleneck.

## [D-002]: Dual-Layer Experience Memory (ChromaDB + SQLite)
- Date: 2026-11-04 | Model: Claude Sonnet 4.6 Thinking
- Context: The fix synthesizer needs to retrieve similar past bug-fix trajectories for few-shot prompting, and the bandit needs fast scalar reward lookups.
- Options Considered:
  1) ChromaDB (vector) + SQLite (scalar trajectories) — hybrid
  2) PostgreSQL with pgvector — single DB
  3) In-memory dict only — no persistence
- Chosen Option: 1 — ChromaDB + SQLite
- Why: ChromaDB gives sub-10ms semantic similarity search with zero infrastructure; SQLite provides reliable ACID trajectory storage without a server process.
- Trade-offs: Two databases to maintain; ChromaDB has no built-in encryption.
- Revisit if: Team needs multi-process / multi-machine memory sharing or trajectory store exceeds 1 GB.

## [D-003]: Multi-Armed Bandit for Repair Strategy Selection
- Date: 2026-11-04 | Model: Claude Sonnet 4.6 Thinking
- Context: Multiple repair strategies exist (DIRECT_EDIT, REFACTOR, HYPERPARAM, FRAMEWORK_CONFIG). We need an adaptive selector that learns which strategy works best per bug class without exhaustive search.
- Options Considered:
  1) Contextual MAB (Thompson Sampling per bug_id) — adaptive
  2) Round-robin strategy rotation — no learning
  3) Always use LLM to pick strategy — full LLM overhead per selection
- Chosen Option: 1 — Contextual MAB
- Why: Converges to the best strategy per bug class with O(log N) regret; no extra LLM call per selection; reward signal comes naturally from ValidationVerdict.accepted.
- Trade-offs: Requires sufficient trajectory volume to converge; cold-start problem on new bug classes.
- Revisit if: Repair success rate plateaus below 70% after 100+ trajectories, suggesting the bandit prior is miscalibrated.

## [D-004]: Fake LLM Provider for Tests (No Real API Calls)
- Date: 2026-11-04 | Model: Claude Sonnet 4.6 Thinking
- Context: 404 automated tests must be fast, deterministic, and free. Real LLM calls are slow, expensive, and non-deterministic.
- Options Considered:
  1) Local fake HTTP server speaking Groq/Cerebras protocol — isolated, fast
  2) `unittest.mock.patch` on the client — brittle, couples tests to internals
  3) VCR cassette replay — requires initial real API recording
- Chosen Option: 1 — Fake HTTP server
- Why: Tests are environment-agnostic; `SBA_NO_DOTENV=1` ensures no real keys are accidentally used; the fake server validates request shape.
- Trade-offs: Fake responses may not reflect real model behavior; must be updated when API schema changes.
- Revisit if: A new LLM provider is added whose protocol the fake server does not yet simulate.

## [D-005]: Subprocess Isolation for Multi-Seed Validation
- Date: 2026-11-04 | Model: Claude Sonnet 4.6 Thinking
- Context: Running multiple seeds of a training script inside the same process risks CUDA context corruption and shared global state pollution.
- Options Considered:
  1) Subprocess per seed — full isolation, OS-level
  2) `multiprocessing.Process` with spawn — partial isolation
  3) Sequential in-process runs — no isolation
- Chosen Option: 1 — Subprocess per seed
- Why: Complete process isolation prevents CUDA context leaks; each seed gets a clean environment; crash of one seed does not kill the validator.
- Trade-offs: Higher startup overhead per seed (~1-2 s); IPC only via stdout (`__METRICS__:{...}` protocol).

## [D-006]: Dual Code-Space and Weight-Space Programming (WSP) Architecture
- Date: 2026-11-07 | Model: Gemini 3.8 Flash
- Context: Traditional debugging focuses strictly on source code ASTs, but deep learning models frequently suffer from failures living purely in parameter space (e.g. rank collapse, dead neuron subspaces, reasoning loops driven by kurtosis outliers). We need a system that inspects and repairs both spaces simultaneously.
- Options Considered:
  1) Dedicated WSP package (`silent_bug_agent/wsp/`) running simultaneous code-space and weight-space audits.
  2) Treat weight tensors as plain runtime hooks inside the existing AST supervisor.
  3) Completely separate standalone CLI package outside `silent_bug_agent`.
- Chosen Option: 1 — Dedicated WSP package inside `silent_bug_agent`
- Why: Keeps code-space detectors and weight-space inspectors modular; enables combined `DualSolution` objects (code diff + weight surgery $\Delta W$); shares validation infrastructure (Gate 1 Overfit Oracle + Gate 2 Multi-Seed Sandbox).
- Trade-offs: SVD and kurtosis calculations add tensor inspection latency (~2-5 seconds on CPU); requires PyTorch on the host machine.
- Revisit if: Distributed tensor parallelism (FSDP / Megatron) is required, necessitating distributed weight sharding.

## [D-007]: Human-in-the-Loop Interactive Approval Gate for Weight Surgery
- Date: 2026-11-07 | Model: Gemini 3.8 Flash
- Context: Weight-space modifications (subspace projection excision, low-rank delta injection, bias centering) directly alter model behavior without gradient descent. Blindly applying tensor edits to weights can corrupt model representations.
- Options Considered:
  1) Mandatory user confirmation prompt displaying both Code Diff and Weight $\Delta W$ metrics before entering empirical validation gates.
  2) Fully autonomous unsupervised modification and write-back to checkpoints.
  3) Offline-only dry-run reporting without repair capability.
- Chosen Option: 1 — Interactive Human-in-the-Loop confirmation gate with `--yes` override for automated test harnesses.
- Why: Provides full interpretability and safety for human developers; surfaces affected parameter percentages and Frobenius delta norms before execution; allows stopping or skipping unpromising proposals.
- Trade-offs: Requires user interaction in interactive shells unless `--yes` is passed.
- Revisit if: Completely autonomous continuous self-repair loops in production cluster environments are required.
- Revisit if: Seed startup time dominates total validation cost (>50% of wall time).
