# Execution Flow

## Flow: Full Pipeline — `sba fix <script.py>`

### Call Sequence

1. `cli.py` → `main()`: Parses CLI args, calls `load_env()`, creates `run_id`, builds initial `AgentState(target_file=..., run_id=...)`.
2. `agents/supervisor.py` → `build_graph()`: Compiles LangGraph StateGraph with nodes: detect → diagnose → fix → validate → store → (loop or done).
3. **Detection Node**
   - `static/analyzer.py` → `StaticAnalyzer.analyze()`: Runs 10 AST visitor rules over the target file; emits `Finding` list.
   - `runtime/instrument.py` → `Instrumenter.attach()`: Registers PyTorch hooks (forward, backward) on model layers.
   - `runtime/session.py` → `RuntimeSession.run()`: Executes one monitored forward/backward pass; collects telemetry.
   - `agents/runtime_monitor_agent.py` → `RuntimeMonitorAgent.run()`: LLM consolidates telemetry + static findings into unified `DetectionReport`.
   - `semantic/analyzer.py` → `SemanticAnalyzer.review()`: LLM reviews source for intent vs. implementation mismatches.
4. **Diagnosis Node**
   - `diagnosis/bridge.py` → `build_taxonomy_graph()`: Loads `taxonomy/bug_taxonomy.yaml` into a weighted causal graph.
   - `diagnosis/inference/causal_engine.py` → `CausalEngine.rank()`: Applies symptom dampening matrix to findings; scores hypotheses.
   - `agents/diagnosis_agent.py` → `DiagnosisAgent.run()`: LLM performs chain-of-thought reasoning over ranked hypotheses; emits `List[DiagnosticHypothesis]`.
5. **Fix Node**
   - `learning/bandit.py` → `ContextualBandit.select()`: Picks `RepairStrategy` for the top hypothesis.
   - `learning/experience_store.py` → `ExperienceStore.retrieve()`: Fetches similar past trajectories from ChromaDB for few-shot context.
   - `repair/recipes.py` → `apply_recipe()`: Attempts rule-based patch first.
   - `agents/fix_synthesizer_agent.py` → `FixSynthesizerAgent.run()`: ReAct LLM generates unified diff `PatchCandidate`.
6. **Validation Node**
   - `repair/patch_utils.py` → `apply_patch()`: Applies diff to a temp copy of the target file.
   - `validation/overfit_agent.py` → `OverfitOracle.run()`: Gate 1 — single-batch overfit check; numeric sanity and gradient flow.
   - `validation/multiseed_agent.py` → `MultiSeedAgent.run()`: Gate 2 — 5 isolated subprocesses, paired t-test + KS-test + Cohen's d.
   - `validation/analyzer.py` → `ValidationAnalyzerAgent.run()`: LLM interprets statistics; emits final `ValidationVerdict`.
7. **Store / Loop Node**
   - If `verdict.accepted`: `learning/experience_store.py` → `ExperienceStore.store()`: Writes trajectory to SQLite + ChromaDB; bandit reward updated.
   - If `verdict.rejected` and `iteration < max_iterations`: loop back to Fix Node with rejection reason injected into prompt.
   - If `verdict.rejected` and budget exhausted: `AgentState.status = ABORTED`.

### Modifying In This Change:
- Track which file/function/lines you change here during active edits.

---

## Flow: Static Scan — `sba scan <script.py>`

### Call Sequence
1. `cli.py` → `scan_command()`: Builds minimal `AgentState`.
2. `static/analyzer.py` → `StaticAnalyzer.analyze()`: Runs 10 AST rules; returns findings.
3. `semantic/analyzer.py` → `SemanticAnalyzer.review()` (if `--semantic` flag): LLM reviews intent.
4. `cli.py` → renders findings table to stdout.

---

## Flow: Runtime Monitor — `sba monitor <script.py>`

### Call Sequence
1. `cli.py` → `monitor_command()`: Builds `AgentState`, launches model execution.
2. `runtime/instrument.py` → attaches hooks to all `nn.Module` layers.
3. `runtime/hooks.py` → collects per-layer tensor statistics on each forward/backward pass.
4. `runtime/anomaly.py` → `AnomalyDetector.check()`: Flags gradient explosions, dead activations, NaN/Inf.
5. `runtime/alerts.py` → emits alerts to stdout in real time.
6. `agents/runtime_monitor_agent.py` → LLM consolidates alerts into a natural-language report.

---

## Flow: Weight-Space Programming — `sba wsp <script.py> [--checkpoint ckpt.pt]`

### Call Sequence

1. `cli.py` → `run_wsp()`: Parses CLI args (`script`, `--checkpoint`, `--max-iter`, `--dry-run`, `--yes`, `--events`). Instantiates `DualSupervisorAgent(DualConfig)`.
2. `wsp/dual_supervisor.py` → `run()`: Starts iterative repair loop (up to `max_iterations`, default 3).
3. **Simultaneous Perception Phase** (`_simultaneous_perception`):
   - **Code-Space**:
     - `static/analyzer.py` → `StaticAnalyzer.analyze_file()` (AST rules).
     - `agents/runtime_monitor_agent.py` → `RuntimeMonitorAgent.monitor_script()` (live telemetry hooks).
     - Merges into `Report`.
   - **Weight-Space**:
     - `wsp/dual_supervisor.py` → `_try_extract_model()`: Instantiates model from script, or reads `--checkpoint`.
     - `wsp/inspector.py` → `WeightInspector.inspect_model()` / `inspect_checkpoint()`:
       - SVD singular value decomposition, condition number $\kappa = \sigma_1 / \sigma_k$.
       - Spectral entropy effective rank $r_{eff} = \exp(H(p_i))$.
       - Kurtosis / heavy-tailed outlier check on weights.
       - Extreme bias collapse / dead ReLU check ($bias \le -15.0$).
       - Emits `WeightReport` with `List[WeightFinding]`.
4. **Dual Solution Synthesis** (`_synthesize_dual_solution`):
   - Weight findings $\rightarrow$ `wsp/editor.py` synthesizes surgical `WeightPatch` (subspace excision, low-rank delta, bias centering, outlier clamping).
   - Code findings $\rightarrow$ `repair/recipes.py` synthesizes AST code diff.
   - Combines into `DualSolution`.
5. **Interactive User Approval Gate** (`_prompt_user_approval`):
   - Emits `wsp_proposal` event (for TUI/API).
   - In CLI interactive mode: prints rationale, code diff, and weight surgery details ($\Delta W$ norm, affected parameter count & percentage).
   - Prompts user: `Implement this proposed solution and enter validation gate? [y/N/stop]`.
   - If user rejects or chooses stop $\rightarrow$ aborts cleanly.
6. **Empirical Validation Gates**:
   - Applies code diff and/or weight patch.
   - **Gate 1**: `validation/overfit_agent.py` → `OverfitOracleAgent.check_code()` (single-batch numeric sanity).
   - **Gate 2**: `validation/multiseedagent.py` → `MultiseedAgent.evaluate_code_across_seeds()` (paired t-test / multi-seed comparison across seeds).
   - If both gates pass:
     - If not dry-run: writes patched code/checkpoint to disk.
     - Emits success event and exits 0.
   - If gates fail:
     - Feedback (e.g. `Gate 1 passed=False`, `loss difference`) recorded in `feedback_history`.
     - Loops back to Step 3 for next iteration (up to 3 iterations max).
