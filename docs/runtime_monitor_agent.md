# Runtime Monitor Agent

**Owner:** Person 2 (Rishikesh) · **Code:** `silent_bug_agent/agents/runtime_monitor_agent.py` plus tools in
`silent_bug_agent/runtime/` · **Tests:** `tests/test_runtime_monitor_agent.py` · **Benchmark:**
`examples/benchmark_hooks.py`, results in [`overhead_benchmark.md`](overhead_benchmark.md)

## The task

> Develop the Runtime Monitor Agent that acts as an active observer during model training. It subscribes to
> telemetry events, continuously runs invariant checks (Module 0 invariant suite), detects silent anomalies (e.g.
> dead units, exploding gradients, frozen parameters), and constructs contextual Finding alert signals for the
> Supervisor Agent. Concurrently profile and benchmark observation overhead on heavy deep learning workloads
> (ResNet-50, Transformer, MLP) to guarantee < 5% compute overhead with zero GPU memory leaks.
> Port live invariant evaluation, runtime anomaly detection, and overhead benchmarking from package101
> (`src/trainmedic/detect/invariants.py`, `probes.py`, and `bench/`).
> Deliverable: `silent_bug_agent/agents/runtime_monitor_agent.py`, `examples/benchmark_hooks.py`, integration tests
> with synthetic training loops, and formal overhead benchmark documentation.

## Integration contract (Person 1, Task 1.2)

```python
from pathlib import Path
from silent_bug_agent.schemas import AgentState
from silent_bug_agent.findings import Report

class RuntimeMonitorAgent:
    def monitor_script(self, path: Path) -> Report: ...      # attach hooks to a training script, return Findings
    def __call__(self, state: AgentState) -> AgentState: ... # Supervisor protocol: fills state.detection_report
```

Implemented with exactly these names. Register it with `SupervisorAgent(detection_agent=RuntimeMonitorAgent())`.

- `monitor_script(path, args=None)` takes optional script arguments. `__call__` reads them from
  `state.extra["script_args"]` if the Supervisor sets that key.
- If `state.detection_report` already holds findings, `__call__` merges instead of overwriting. Findings from
  another detector (static, semantic) stay first, and a runtime finding of the same bug within 2 lines is dropped
  (`merge_reports`).
- The run summary is written to `state.extra["runtime_monitor"]`.
- The module imports without torch. Without torch, `monitor_script` returns a `Report` with a diagnostic
  instead of raising.

> **Note:** the contract says `SupervisorAgent`, `ExperienceAgent` and the CLI `monitor` command are live, but
> they were not in this repository when this was written (`main` at `fcb210b`). The agent was built and tested
> against the contract above and against `schemas.py` / `findings.py` as they are on `main`. Until the CLI
> command lands, run it with `python -m silent_bug_agent.agents.runtime_monitor_agent`.

## Usage

```bash
# a training script, unmodified (runs in a subprocess, at most 300 optimizer steps / 900 s)
python -m silent_bug_agent.agents.runtime_monitor_agent train.py
python -m silent_bug_agent.agents.runtime_monitor_agent train.py --max-steps 500 --json -- --epochs 3 --lr 0.1

# the planted-bug demo script
python -m silent_bug_agent.agents.runtime_monitor_agent examples/sample_scripts/planted_runtime_bugs.py -- --bug frozen
```

Exit code: 0 with no findings, 1 with a warning or critical finding, 2 if the script does not exist (the same
convention as `silent-bug-agent scan`). Early alerts are printed to stderr as they happen.

```python
from silent_bug_agent.agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent

# Supervisor-style: one call, a Report back; on_alert fires while the script is still running
agent = RuntimeMonitorAgent(MonitorConfig(max_steps=300), on_alert=lambda f: print("early:", f.bug_id))
report = agent.monitor_script(Path("train.py"))
print(report.format_text())

# Live, in-process, in your own loop (keeps monitoring overhead under 5% by default)
with agent.watch(model, optimizer) as session:
    for x, y in loader:          # unchanged training loop
        ...
    print(session.alerts)        # confirmed alerts so far, while training runs
report = session.report()
```

## How it works, step by step

1. **Start the script.** `monitor_script` runs the script unmodified, in a subprocess by default
   (`isolate=True`). The subprocess isolates crashes and timeouts, keeps global patches out of the
   Supervisor's process, and gives each run clean GPU memory. The script runs as `__main__` with its own
   `sys.argv`, and with a real `__main__` module so pickling and spawned DataLoader workers work.
2. **Find the model and optimizer.** `AutoInstrumenter` (`runtime/instrument.py`, ported from package101's
   tracer) adds temporary global hooks:
   - an `Optimizer.__init__` wrapper records each optimizer, plus the script line that built it;
   - global module forward hooks track call depth, so a module called at depth 0 is a root model;
   - on a root model's first forward, every optimizer whose parameters overlap that model is *bound* to it,
     before the forward runs.
   One model+optimizer pair is a *channel*. A GAN has two.
3. **Attach the sensors.** For each channel the session attaches the Module 1 `RuntimeHooks` and subscribes to
   its `SnapshotStream`. Those `TrainingState` snapshots are the telemetry events. The session also adds:
   - optional `OverheadGovernor` step hooks;
   - source-anchor hooks: where the step is called, where the model is called for training and for validation;
   - the gradient-lifecycle sensor, which watches `zero_grad()` calls and gradient tensor identity between
     steps.
4. **Run startup checks** at the first optimizer step:
   - **optimizer coverage:** trainable parameters that no optimizer updates (package101
     `OptimizerCoversTrainableParams`). It fires at step 1, long before the 20-step `frozen_weights` patience.
   - **device placement:** the model trains on the CPU although CUDA is available.
5. **Check every snapshot.** Training never waits: snapshots go into a bounded inbox. The consumer thread
   evaluates each one with `InvariantEvaluator` (`runtime/anomaly.py`):
   - the **Module 0 invariant suite** (`get_default_registry()`), unchanged. Each failure is mapped to a taxonomy
     `bug_id`: gradient bounds become `exploding_gradients` or `vanishing_gradients`, `frozen_weights` becomes
     `frozen_layer_no_grad`, `activation_health` becomes `dead_relu` or non-finite, and so on;
   - **runtime checks** ported from package101: non-finite loss (`FiniteLoss`) and loss spikes (`NoLossSpikes`,
     from the EMA z-score the hooks compute);
   - the **rolling-window trend analyzer** (`runtime/trends.py`), which catches trends before a threshold is
     crossed:
     - gradient norm growing exponentially (×100 over 8 snapshots);
     - loss divergence (median doubled, with the whole recent window above the earlier level, so noise does not
       trigger it);
     - **dead units:** `dead_unit_frac` ≥ 50% for 5 snapshots. This is a new hook statistic: the fraction of
       ReLU units that are zero for every sample (measured for batches of at least 16 samples). It catches
       partly dead layers that the 90% `zero_frac` threshold misses;
     - update ratio ‖ΔW‖/‖W‖ far too high or far too low (package101 `UpdateRatioSane`).
6. **Aggregate and confirm.** `AlertDispatcher` (`runtime/alerts.py`) folds the signals into **one alert per bug
   per channel**, recording first and last step, count per rule (once per snapshot), peak, offenders, and a trace
   of the first occurrence.
   - Critical signals and checks with their own patience confirm an alert at once. Noisy per-step checks
     (gradient bounds, activation health, loss spikes) need 2 occurrences, so a single transient reading is
     counted but not reported.
   - Confidence starts at the check's base confidence (package101 values) and grows with persistence. Several
     rules supporting the same bug are combined with a noisy-OR.
   - A loss plateau after the loss has already fallen to ≤50% of its start is treated as convergence, not as a
     `loss_trend` failure. This removed the clean-run false positive.
7. **Send early alerts.** The moment an alert is confirmed (and again if it escalates to critical), `on_alert`
   receives a `Finding`.
   - In subprocess mode the child streams each one as a JSON line, and the parent calls `on_alert` while the
     script is still running.
   - With `stop_on_critical=True` the run is stopped with `StopTraining` once a critical alert is confirmed.
8. **Stop and report.** The run ends when the script finishes or the budget is reached (`max_steps`,
   `max_seconds`). `StopTraining` is a `BaseException`, so the script's own `except Exception:` blocks do not
   swallow it. Everything is then flushed and detached, and `session.report()` returns the `Report`.

## The Findings it produces

Each confirmed alert becomes one `Finding` with `detector="runtime"`:

| field | content |
|---|---|
| `bug_id` | Module 0 taxonomy id: `exploding_gradients`, `vanishing_gradients`, `dead_relu`, `frozen_layer_no_grad`, `train_eval_mode_mismatch`, `missing_zero_grad`, `learning_rate_misconfigured`, `silent_device_fallback` |
| `file`, `line`, `evidence` | the user's code the bug is about, with the source line, like the static analyzer (see below) |
| `severity`, `confidence` | highest severity confirmed; noisy-OR confidence of the supporting rules |
| `message` | `[runtime, steps 1-200, 181 of 200 monitored snapshots] 2 trainable parameter(s) unchanged ...` |
| `rule` | the main rule, e.g. `invariant:frozen_weights`, `runtime:missing_zero_grad`, `trend:dead_units` |
| `extra` | `first_step`, `last_step`, `confirmed_step`, `occurrences`, `monitored_snapshots`, `phase`, `channel`, `rules` (per-rule counts, peaks, messages), `offenders` (top 10 parameters/layers with values), `alternative_bug_ids` (other taxonomy entries with the same symptom, for the diagnosis agent), `context` (trace of the first occurrence: loss, loss tail, EMA, global grad norm, update ratio, LRs, stats of the offending layers, trend summary) |

Where a finding points:

| finding | points at |
|---|---|
| frozen layer, learning rate | the line that builds the optimizer (LR, param groups) |
| exploding gradients, non-finite loss, missing `zero_grad` | the `optimizer.step()` line |
| dead units, vanishing gradients | the line in the model class that defines the layer (`self.act1 = nn.ReLU()`) |
| train/eval mode | the validation forward call (or the training forward) |

## Results on the planted bugs

From `examples/sample_scripts/planted_runtime_bugs.py`. It uses the same model and data as
`examples/planted_bugs_torch.py`, plus four new bugs. Each run is 200 steps, in-process, on CPU. All of these are
asserted in the tests.

| planted bug | findings (first step) |
|---|---|
| none (clean run) | **no findings** |
| `model.eval()` missing in validation | `train_eval_mode_mismatch` critical (50, first validation), pointing at the validation forward |
| `fc2` left out of the optimizer | `frozen_layer_no_grad` (**step 1**, optimizer coverage; confirmed by `frozen_weights` from 20) |
| learning rate 500 | `exploding_gradients` critical (2), `learning_rate_misconfigured` (1) |
| `fc1` bias −100 (all ReLUs dead) | `dead_relu` (1) at `self.act1 = nn.ReLU()`, `vanishing_gradients`, `frozen_layer_no_grad`, `learning_rate_misconfigured` |
| 20 of 32 ReLU units dead | `dead_relu` (5) via `trend:dead_units`; the Module 0 suite alone reports nothing |
| random labels | `learning_rate_misconfigured` (loss never improves, step 90), with `label_data_misalignment` among its alternatives |
| `optimizer.zero_grad()` missing | `missing_zero_grad` critical (**step 4**), then `exploding_gradients` (16) |
| NaN in the input from step 30 | `exploding_gradients` critical (30); the activation rule names `fc1` as the first non-finite layer |
| loss scale doubling every step | `exploding_gradients` critical (8) |

## Overhead and memory

Full methodology and results: [`overhead_benchmark.md`](overhead_benchmark.md). In short, on an Apple M2 GPU with
the default governor:

- **overhead** was ResNet-50 +2.1%, Transformer within noise, MLP +4.7%;
- **GPU memory** grew by 0 B during monitoring, with 0 B left after detach;
- **CUDA** has not been measured yet.

- **`watch()` (live training)** uses the `OverheadGovernor` with `overhead_budget=0.05`:
  - it monitors every step for the first 8 steps, then pauses monitoring for 6 to measure the plain step time;
  - it then monitors one step in N, with N the smallest value whose estimated overhead fits 60% of the budget
    (3%), leaving headroom for timing noise;
  - it re-measures every 200 steps;
  - heavy models keep N = 1; tiny models are sampled.
- **`monitor_script()`** monitors every step by default (`script_overhead_budget=None`), because a diagnostic run
  is short and detection matters more there. Set `script_overhead_budget=0.05` to govern it as well.
- **Memory:** `RuntimeHooks.detach()` now frees every device buffer the hooks allocated. The benchmark and the
  tests check, with exact allocated-byte counters on CUDA and MPS, that GPU memory does not grow during
  monitoring and that nothing is left after detach.

## Configuration (`MonitorConfig`)

| option | default | meaning |
|---|---|---|
| `max_steps`, `max_seconds` | 300, 900 | budget of a monitored script run (`None` = no limit) |
| `isolate` | True | run the script in a subprocess |
| `cwd`, `echo_output`, `timeout_grace` | current dir, False, 120 s | script working directory; show the script's output; extra time before a hung child is killed |
| `overhead_budget` | 0.05 | `watch()`: amortised monitoring cost cap (`None` = every step) |
| `script_overhead_budget` | None | the same for `monitor_script()` |
| `min_severity` | `"warning"` | findings below this are dropped |
| `confirm_after` | 2 | occurrences before a noisy per-step check becomes a finding |
| `stop_on_critical` | False | stop the run when a critical alert is confirmed |
| `grad_lifecycle` | True | missing-`zero_grad()` sensor |
| `hook_overrides` | {} | `HookConfig` fields, e.g. `{"histogram_bins": 16}`; the agent's defaults are `stream_capacity=256`, `max_loss_history=2048` |

## Files

| file | role (tool named in the task) |
|---|---|
| `agents/runtime_monitor_agent.py` | the agent: contract, `watch()`, subprocess runner with early-alert streaming, CLI |
| `runtime/session.py` | observe → check → alert loop: channels, consumer thread, startup checks, report |
| `runtime/anomaly.py` | **invariant evaluator**: Module 0 suite + package101 runtime checks → taxonomy signals |
| `runtime/trends.py` | **rolling window trend analyzer** |
| `runtime/alerts.py` | **early-alert callback dispatcher**: aggregation, confirmation, findings |
| `runtime/instrument.py` | script instrumentation, gradient-lifecycle sensor, source anchors |
| `runtime/profiling.py` | **overhead governor**, **memory leak detector**, **PyTorch profiler integration**, step timing |
| `examples/benchmark_hooks.py` | **synthetic benchmark runner** (ResNet-50, Transformer, MLP) |
| `examples/sample_scripts/planted_runtime_bugs.py` | demo/test script with 10 selectable planted bugs |
| `runtime/hooks.py` (changed) | new `dead_unit_frac` statistic, `pause()` / `resume()` / `set_every_n_steps()`, `detach()` frees device buffers |

## Interop with the diagnosis agent

`feat/root-cause-diagnosis-agent` takes runtime alerts as dicts with `id`, `type`, `message`, `step`,
`severity` and `metadata`. A runtime `Finding` maps to that format as `type=bug_id`, `step=extra["step"]`,
`metadata=extra`.

## Known limitations

- **No CUDA run yet.** CUDA has not been run on this machine. The benchmark ran on CPU and on an Apple GPU (MPS).
  Run `python examples/benchmark_hooks.py --device cuda` on an NVIDIA machine.
- **Manual in-place zeroing looks like a missing `zero_grad()`.** Clearing gradients with `p.grad.zero_()`
  instead of `zero_grad()` or `set_to_none` looks the same as a missing `zero_grad()` to the lifecycle sensor.
- **Non-finite values have no dedicated taxonomy entry.** They are reported as `exploding_gradients`, with
  `mixed_precision_overflow` and `non_normalized_inputs` among the alternatives. When the first non-finite layer
  is the model's first layer, the message says the input batch is likely the cause.
- **A sub-module called directly can be bound first.** A model is bound on its first root forward. If a script
  calls a sub-module directly before the full model, the sub-module is bound instead.
- **Tiny models keep a small fixed cost.** On models with sub-millisecond steps, a few microseconds of hook
  dispatch per step remain even when the governor samples rarely (see the benchmark document).
- **Inherited from Module 1:** activations inside `torch.compile` regions are not recorded; a layer called
  several times per step reports its last call.

## Provenance

Ported from package101 (https://github.com/rishikeshwalhekar49-lab/package101):

| from | to |
|---|---|
| `src/trainmedic/detect/invariants.py`: `FiniteLoss`, `NoLossSpikes`, `UpdateRatioSane`, `OptimizerCoversTrainableParams`, `GradsClearedBetweenSteps` (live version), base confidences | `runtime/anomaly.py`, `runtime/trends.py`, `runtime/session.py`, `runtime/instrument.py` |
| `src/trainmedic/detect/tracer.py` and `_bootstrap.py`: global hooks, optimizer discovery, `StopTraining` budget, subprocess execution | `runtime/instrument.py`, `agents/runtime_monitor_agent.py` |
| `src/trainmedic/detect/probes.py` | the batch-level probes (one-batch overfit, determinism) belong to Person 3's validation agents and were not duplicated here; the runtime monitor keeps to passive observation |
| `src/trainmedic/bench/`: interleaved clean/buggy runs, budgets, JSON reports | `examples/benchmark_hooks.py`, `runtime/profiling.py` |

Everything is rewritten against this repository's `TrainingState`, `Finding`/`Report`, `AgentState` and taxonomy.
