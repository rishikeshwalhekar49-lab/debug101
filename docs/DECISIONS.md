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
  1) Contextual bandit with an exploration bonus, one context per bug category — adaptive
  2) Round-robin strategy rotation — no learning
  3) Always use LLM to pick strategy — full LLM overhead per selection
- Chosen Option: 1 — Contextual bandit
- What the code does (`learning/bandit.py`): each strategy is scored as average reward + an exploration bonus `sqrt(ln(total + 1) / count)` + a bonus for the strategy the hypothesis suggests. This is a UCB-style rule. It is **not** Thompson sampling: nothing is sampled from a posterior. An earlier version of this page said Thompson sampling; that was wrong.
- Why: no extra LLM call per selection; the reward comes directly from the validation verdict (+1 accepted, −0.5 rejected, −1 invalid patch).
- Trade-offs: needs many recorded repairs before it matters. With the handful of runs made so far the bandit has had almost no data, so its effect on results has not been measured. No regret bound is claimed.
- Revisit if: after 100+ recorded repairs the choice of strategy shows no effect on the acceptance rate.

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
- Trade-offs: Higher startup overhead per seed (~1-2 s). Results come back through a JSON result file written by `script_runner.py`; a `__METRICS__:{...}` line on stdout is optional.
- Revisit if: Seed startup time dominates total validation cost (>50% of wall time).

## [D-006]: Dual Code-Space and Weight-Space Programming (WSP) Architecture — SUPERSEDED by D-008
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

## [D-007]: Human-in-the-Loop Interactive Approval Gate for Weight Surgery — SUPERSEDED by D-008 and D-009
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

## [D-008]: Weight inspection is read-only; weight editing is removed
- Date: 2026-11-15
- Context: The WSP package of D-006 proposed "weight surgery" but never applied or tested it: validation retrains the script from scratch, so an edit to a tensor cannot change the outcome. It also loaded models by executing the user's script inside the tool's process, inspected untrained weights, raised false alarms on normalisation layers, and reported a fixed confidence of 0.92. Details are in `features/FEATURE-002_wsp_weight_space_programming.md`.
- Options Considered:
  1) Make weight editing real: apply edits to a checkpoint and validate by evaluating that checkpoint
  2) Keep only a read-only inspector over real trained weights
  3) Remove the package
- Chosen Option: 2
- Why: The project repairs **training code**. A bug in the code reappears on the next training run however the weights are edited, so the durable fix is a code patch, and `fix` already validates those. Inspection still earns its place: it shows what a bug did to the model (for example that every bias in a layer sits at −100), which helps explain a diagnosis.
- How: `editor.py` and `dual_supervisor.py` are deleted. Weights are captured from a sandboxed training run (`script_runner --save-weights`) or read from a checkpoint with `weights_only=True`.
- Trade-offs: no automatic repair of an already-trained checkpoint.
- Revisit if: there is a concrete need to repair checkpoints that cannot be retrained, with an evaluation set to validate against.

## [D-009]: Approval means nothing is written before the user agrees
- Date: 2026-11-15
- Context: The terminal UI showed an "Apply to File / Keep Dry-Run" menu after the pipeline had already written the patch, and then reported "Target file unchanged". Running it from the menu on the bundled examples silently removed the planted bugs from three benchmark files, which broke 15 tests on `main`.
- Chosen Option: `fix --dry-run` validates and stores the patch (`repair/pending.py`) without writing; `apply` writes exactly that patch, only if the file's hash is unchanged. The UI always uses this pair. The bundled examples under `examples/` are never edited by either command.
- Trade-offs: in the UI, one bug is fixed per run instead of up to three.
- Revisit if: users want multi-round fixing with a single approval.

## [D-010]: The approval menu has three real choices, and bundled examples get a fixed copy
- Date: 2026-11-15
- Context: D-009 made the UI refuse to apply anything to a bundled example and removed the old "Retry Fix" choice, because that choice did nothing. The team's demos start from the bundled examples, so the screen ended with "the file was not changed" and no choice at all. The team asked for the three choices back: apply, do not apply, try another fix.
- Options Considered:
  1) Let the UI edit the bundled examples again
  2) Keep refusing, and tell people to copy the file by hand
  3) Show the menu everywhere; for a bundled example write the fixed version to a copy; make "try another fix" a real second search
- Chosen Option: 3
- Why: Option 1 is what broke 15 tests. Option 2 makes the demo end without the step people want to see. Option 3 gives the full flow and the fixtures stay intact.
- How:
  - `apply` on a bundled example writes `fixed_examples/<name>.py` and leaves the example alone.
  - `fix --another` moves the pending patch to a list of declined patches and seeds the fix agent's existing `tried_patches` list with them, so the same result cannot be offered twice. The LLM receives the declined patch as feedback.
- Trade-offs: "another fix" depends on the LLM. Offline there is one rule per bug, so the honest answer is often "no other fix". An LLM can also return a patch that differs only in wording; it counts as different unless the resulting file is identical.
- Revisit if: users expect several alternatives side by side rather than one at a time.
