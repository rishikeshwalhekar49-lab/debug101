# Execution Flow

The function and file names below are the real ones, checked against the code on 2026-11-15.

## Flow: Full Pipeline — `fix <script.py>`

1. `cli.py` → `main()` → `run_fix()`: builds a `PipelineConfig` and a `SupervisorAgent`, then calls `run_until_clean()`, which runs the graph once per bug (up to `--max-rounds`, or once with `--dry-run`).
2. `agents/supervisor.py` → `SupervisorAgent._build_graph()`: a LangGraph `StateGraph` with the nodes `perception → diagnosis → repair → validation → memory`, and a `feedback → repair` loop after a rejection. One `AgentState` object flows through every node.
3. **Perception** (`_perception`)
   - `static/analyzer.py` → `StaticAnalyzer.analyze_file()`: 10 AST rules, no execution.
   - `semantic/analyzer.py` → `SemanticAnalyzer.analyze_file()`: LLM review of the source, only when a key is set.
   - `agents/runtime_monitor_agent.py` → `RuntimeMonitorAgent`: runs the script in a subprocess for a step budget with `runtime/instrument.py` (`AutoInstrumenter`) and `runtime/hooks.py` (`RuntimeHooks`) attached, and turns invariant violations into findings.
   - `_consolidate()`: an optional LLM summary that marks findings that are side effects of others.
   - If the script itself raises, the run stops here with outcome `script_error`.
4. **Diagnosis** (`_diagnosis`)
   - `agents/diagnosis_agent.py` → `DiagnosisAgent`: `diagnosis/bridge.py` (`engine_inputs`, `hypotheses_from_report`) feeds the findings to `diagnosis/inference/causal_engine.py` (`CausalEngine`), which ranks causes above their symptoms. With a key, an LLM reviews the ranking.
5. **Repair** (`_repair`)
   - `repair/loop.py` → `AutonomousRepairLoop.run()` → `agents/fix_synthesizer_agent.py` → `FixSynthesizerAgent`.
   - `learning/bandit.py` → `ContextualBandit.select_strategy()` picks a repair strategy; `learning/experience_store.py` → `ExperienceStore.recall()` supplies similar accepted repairs as examples.
   - With a key the LLM writes the patch; without one, or if its patch is unusable, `repair/recipes.py` → `recipe_for()` writes it from a rule.
   - `repair/patch_utils.py` → `patch_guard()` rejects unsafe or oversized patches before anything is run.
6. **Validation** (`_validation`)
   - Gate 1: `validation/overfit_agent.py` → `OverfitOracleAgent`: the patched script must be able to overfit one batch.
   - Gate 2: `validation/multiseedagent.py` → `MultiseedAgent.verify_patch()`: baseline and patched runs on 3 to 5 seeds in subprocesses (`validation/sandbox.py`, `validation/script_runner.py`), compared with a paired t-test, KS test, Cohen's d and a confidence interval. Skipped if Gate 1 rejected. (`multiseed_agent.py` is only an alias that re-exports this module.)
   - `_bug_gone()`: the detectors are run again on the patched code.
   - `validation/analyzer.py` → `ValidationAnalyzerAgent.analyze()`: accepts on statistics, or when the bug is gone and results are not worse (warning-level evidence or above). An LLM may veto an acceptance; it can never overturn a rejection.
7. **Feedback or memory**
   - Rejected, budget left: `_feedback()` adds the rejection reason to the next repair prompt.
   - `_memory()`: records the attempt and the bandit reward in `ExperienceStore`. If a patch was accepted:
     - normal run: `apply_patch_to_file()` writes the script and keeps `<name>.py.orig`;
     - `--dry-run`: nothing is written; `repair/pending.py` → `save_pending()` keeps the patch for `apply`;
     - a bundled example under `examples/`: nothing is written.
   - No patch accepted: status `failed`, with the last rejection reason. Exit code 1.
   - Every fifth recorded attempt, `agents/meta_evolution.py` → `MetaEvolutionAgent.run()` prunes unhelpful examples and updates lessons.

## Flow: `apply <script.py>`

1. `cli.py` → `run_apply()` → `repair/pending.py` → `apply_pending()`.
2. Refuses if there is no pending patch, if the file's SHA-256 differs from the one recorded when the patch was validated, or if the file is a bundled example.
3. Otherwise `apply_patch_to_file()` writes the patch, keeps the `.orig` backup, and the pending record is deleted.

## Flow: `scan <path>`

1. `cli.py` → `run_scan()` → `StaticAnalyzer.analyze_file()` or `analyze_directory()`.
2. With `--semantic` and a key: `SemanticAnalyzer`, merged with `findings.merge_reports()`.
3. Prints `Report.format_text()` or JSON. Exit code 1 if anything is warning-level or above.

## Flow: `monitor <script.py>`

1. `cli.py` → `run_monitor()` → `RuntimeMonitorAgent.monitor_script()`.
2. The script runs in a subprocess with the hooks attached; early alerts are streamed while it runs.
3. The findings are printed as a report. No LLM is involved.

## Flow: `diagnose <script.py>`

`cli.py` → `run_diagnose()`: the perception steps above, then `DiagnosisAgent`. Prints the ranked hypotheses. Nothing is patched.

## Flow: `wsp [script.py] [--checkpoint FILE]` (read-only)

1. `cli.py` → `run_wsp()`.
2. With `--checkpoint`: `wsp/inspector.py` → `WeightInspector.inspect_checkpoint()` loads the file with `weights_only=True` (or `safetensors`).
3. With a script: `wsp/inspector.py` → `inspect_script()` runs it for `--max-steps` optimizer steps through `SubprocessSandbox.run_measured(save_weights=...)`. Inside the child, `script_runner.py` finds the models an optimizer trained and `Recorder.save_weights()` writes their tensors to a temp file, which is then inspected and deleted.
4. `inspect_state_dict()` checks every floating-point tensor: non-finite values, exploded values, all-zero matrices, collapsed biases, heavy outliers and collapsed rank. Normalisation statistics and zero biases are not flagged.
5. Prints a `WeightReport`. Exit code 1 for a warning or critical finding, 0 if clean, 2 if no weights could be read. Nothing is ever written to the script or the checkpoint.

## Flow: `ui`

`cli.py` → `_launch_ui()` starts `tui/dist/cli.js` with Node. Each screen runs one of the commands above with `--events` and renders the NDJSON events from `events.py`. The Fix screen runs `fix --dry-run`, shows the validated patch, and runs `apply` only if the user confirms.
