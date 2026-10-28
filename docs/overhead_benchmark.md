# Runtime Monitor Agent: overhead and GPU-memory benchmark

**Requirement (task):** the observation overhead of the Runtime Monitor Agent on heavy deep learning
workloads (ResNet-50, Transformer, MLP) must stay **below 5% of compute**, with **zero GPU memory
leaks**.

**Runner:** `examples/benchmark_hooks.py`. Building blocks: `silent_bug_agent/runtime/profiling.py`.

**Raw results:** `bench_results/` (JSON and Markdown written by the runner).

## 1. Summary

Measured on an Apple M2 GPU (MPS), torch 2.14.1, in the final run (5 repetitions):

| workload | baseline step | agent, every step | **agent, governed (default)** | governor N | GPU memory growth | left after detach | false alarms |
|---|---|---|---|---|---|---|---|
| ResNet-50 (25.6 M params) | 465 ms | +17.0% | **+2.1%** | 3 | 0 B | 0 B | none |
| Transformer LM (51.7 M params) | 582 ms | +5.7% | **−7.8%** (noise; see 3.2) | 6 | 0 B | 0 B | none |
| MLP (41.9 M params) | 114 ms | +25.3% | **+4.7%** | 9 | 0 B | 0 B | none |

- **Overhead: met.** With the default overhead governor, the observation overhead was below 5% on all three
  heavy workloads, in both GPU runs (section 3.2) and on CPU (section 3.3: +2.7% at most).
- **GPU memory leaks: none.** Exact allocated-byte counters show 0 B of growth during monitoring and 0 B left
  after detach, in every cycle, for every workload.
- **Every-step monitoring** costs 3–25% on this GPU. The governor therefore monitors one step in 3 to 9.
  CUDA has cheaper kernel launches and fused multi-tensor kernels, so it should need less sampling, but this
  is not yet measured (section 6).
- **Optimisation and a leak fix along the way.** Profiling during this work cut the every-step cost of
  ResNet-50 from **+95% to +17%**, and the memory check found and fixed a real leak (section 4).

## 2. What is measured

### 2.1 Workloads

| name | model | input per step | optimizer | parameters |
|---|---|---|---|---|
| `resnet50` | ResNet-50 (`examples/benchmark_hooks.py`, same layout as `torchvision.models.resnet50`) | B×3×224×224 images, 1000 classes; B = 32 on CUDA, 16 on MPS, 8 on CPU | SGD, momentum 0.9, weight decay 1e-4 | 25.6 M |
| `transformer` | Pre-norm Transformer encoder LM: 6 layers, d_model 512, 8 heads, FFN 2048, vocab 32,000 | B×255 tokens; B = 16 CUDA, 8 MPS, 4 CPU | AdamW | 51.7 M |
| `mlp` | MLP 1024-4096-4096-4096-1000, ReLU | batch 512 | AdamW | 41.9 M |

The data are synthetic random tensors created once on the device, so no data loading is timed.

### 2.2 Monitoring modes

| mode | what is attached |
|---|---|
| `baseline` | nothing |
| `hooks_every_step` | `RuntimeHooks` only (Module 1 sensors), every step monitored |
| `agent_every_step` | full agent `watch()` with `overhead_budget=None`. That is hooks, plus the consumer thread running the Module 0 invariants, runtime checks and trend analyzer, plus alert aggregation, the gradient-lifecycle sensor and source anchors; every step monitored |
| `agent_governed` | full agent with the default `overhead_budget=0.05`: the `OverheadGovernor` picks one monitored step in N |

`agent_governed` is the configuration the requirement applies to: it is the default for live monitoring
(`watch()`). The every-step modes show what monitoring costs per monitored step.

### 2.3 Protocol

1. Build the model, optimizer and data on the device, and run 3 untimed warm-up steps.
2. Repeat 3 times (repetitions):
   - run one block per mode, rotating the order of the modes in every repetition, so slow drift such as
     thermal throttling or allocator warm-up does not favour any mode;
   - each block attaches its mode, runs untimed warm-up steps, times its steps, then detaches:
     - every-step modes: 2 warm-up steps (the first monitored step allocates the staging buffers);
     - `agent_governed`: 16 warm-up steps, which covers the governor's 8-step calibration and 6-step probe.
3. Time each step with a device synchronisation before and after it (`torch.cuda.synchronize` /
   `torch.mps.synchronize`), so GPU time is fully counted.
4. Use 10 timed steps per block. For `agent_governed` the count is rounded to whole sampling cycles (a multiple
   of the chosen N), so the block contains exactly one monitored step per N steps.

**Overhead** of a mode = mean step time of its block ÷ mean step time of the baseline block of the same
repetition − 1. The reported value is the **median over the 3 repetitions**. Means, not medians, are used
within a block so the cost of the rare monitored step in a governed block is fully counted.

### 2.4 GPU memory leak check

Done for every workload on CUDA and MPS, where the allocator gives exact live bytes
(`torch.cuda.memory_allocated`, `torch.mps.current_allocated_memory`):

1. **Control:** run the same number of steps *without* monitoring, measured the same way, to show what the
   training step itself leaves behind (allocator or library caches).
2. **Two monitored cycles.** In each one: call `gc.collect()`, synchronise, and record allocated bytes; attach
   the full agent (`agent_every_step`) and run 10 steps, recording allocated bytes after every step; detach,
   call `gc.collect()` and synchronise again.

A leak in the monitor would show in every cycle; a one-off allocation (e.g. a library cache filled on first
use) shows in only one. The first benchmark run showed one such 1.5 MiB one-off on the Transformer. It did
not reproduce in two re-runs, and nine attach/detach cycles in a row left exactly 0 B each.

The reported figures:
- **growth:** allocated bytes after the last monitored step minus after step 3, the largest over the cycles.
  It must be **0 B**;
- **left after detach:** after detach minus before attach, the smallest over the cycles. It must be **0 B**;
- **monitoring overhead bytes:** the constant memory the hooks use while attached (staging buffers and
  weight samples). It is reported for information; it is not a leak.

Host memory (RSS) is reported for information only. Allocator and OS noise make it inexact. The bounded
host-side structures are tested separately with `tracemalloc` (`test_host_memory_stays_bounded`).

### 2.5 Profiler

`torch.profiler` records 2 baseline and 2 monitored steps (`hooks_every_step`). `profile_diff` lists the
operators monitoring adds, with their call counts and self CPU time per step. It is used to find the cost
drivers (section 4).

## 3. Results

### 3.1 Final run (Apple M2 GPU, 5 repetitions × 10 timed steps)

Raw data: `bench_results/mps_apple_m2.json` and `bench_results/mps_apple_m2.md`.

| workload | baseline step | hooks every step | agent every step | agent governed | N | governor's own estimate | cost of one monitored step |
|---|---|---|---|---|---|---|---|
| ResNet-50, 16×3×224×224, SGD+momentum | 465 ms | +16.7% | +17.0% | **+2.1%** | 3 | 2.5% | 50 ms |
| Transformer LM, 8×255 tokens, AdamW | 582 ms | +2.7% | +5.7% | **−7.8%** | 6 | 2.5% | 71 ms |
| MLP 1024-4096³-1000, batch 512, AdamW | 114 ms | +25.1% | +25.3% | **+4.7%** | 9 | 2.8% | 28 ms |

GPU memory (exact allocated bytes; control = the same steps without monitoring):

| workload | growth while monitored | left after detach (cycle 1, 2) | control growth / residue | constant memory while attached |
|---|---|---|---|---|
| ResNet-50 | 0 B | 0 B, 0 B | 0 B / 0 B | 96 MiB |
| Transformer | 0 B | 0 B, 0 B | 0 B / −256 KiB | 89 MiB |
| MLP | 0 B | 0 B, 0 B | 0 B / 0 B | 17 MiB |

The constant memory is the weight samples used to measure weight deltas (up to `max_param_elems` = 1,048,576
elements per parameter tensor) plus the staging buffers. It is allocated on attach and freed on detach.
Lower it with `HookConfig(max_param_elems=...)`. The negative control residue shows that the training step itself
can release cached memory, which is why the control block is part of the check.

### 3.2 Earlier run, and how much the numbers vary

The same code was run before the last governor tuning (3 repetitions; margin 0.8, 4-step probe):

| workload | hooks every step | agent every step | agent governed | N |
|---|---|---|---|---|
| ResNet-50 | +19.5% | +19.2% | **+3.5%** | 5 |
| Transformer | +11.2% | +11.0% | **+2.3%** | 4 |
| MLP | +21.8% | +11.9% | **+1.0%** | 7 |

Between runs, the every-step cost of the Transformer moved from +11% to +3–6%. Its baseline step also moved,
from 437 ms to 582 ms, under the same code. That is the run-to-run variation of a laptop GPU (thermal state,
background load). The final run's −7.8% for the governed Transformer is the same noise: in that run the
governed block happened to be faster than the baseline blocks. The governor's own estimate for that run is
2.5%. Because of this variation the governor aims at 3% rather than 5% (section 5).

### 3.3 CPU (Apple M2, 4 threads, 3 repetitions × 8 timed steps)

Raw data: `bench_results/cpu_apple_m2.json` and `bench_results/cpu_apple_m2.md`. CPU tensors have no exact
allocation counter, so the GPU memory columns do not apply.

| workload | baseline step | hooks every step | agent every step | agent governed | N | governor's own estimate |
|---|---|---|---|---|---|---|
| ResNet-50, 8×3×224×224 | 614 ms | +12.6% | +11.7% | **−0.8%** (noise) | 6 | 2.5% |
| Transformer LM, 4×255 tokens | 506 ms | +10.0% | +6.6% | **+1.7%** | 4 | 2.6% |
| MLP, batch 512 | 213 ms | +10.9% | +12.0% | **+2.7%** | 4 | 2.7% |

**One finding on a healthy workload, investigated and fixed.** The CPU run reported a `dead_relu` warning on
ResNet-50. From step 14, 55–90% of the units of `layer4.2.relu` were zero for every sample. That measurement is
correct for this setup: the benchmark trains on one fixed random batch, and on CPU that batch has only 8
images. But "zero for all 8 images of the same batch" is weak evidence that a unit is dead. The GPU run, with
16 images per batch, reported nothing. `dead_unit_frac` is therefore now measured only for batches of at
least 16 samples. A fully dead layer is still caught at any batch size by the Module 0 90%-zeros rule. After
the change, the same 25-step CPU run reports no finding. The raw JSON above is from the run before the fix and
still lists the warning. Every other healthy workload in both runs produced no finding.

### 3.4 Where the time goes (PyTorch profiler, ResNet-50)

The profiler records 15,475 operators in a monitored step against 3,576 for training alone. Most of the
added ones are free view/slice bookkeeping; the kernels that do work are the copies into the staging buffers,
one sum per sampled tensor, the batched statistics and the parameter norms.
`aten::copy_` heads the profiler's list, with about 900 ms of self CPU time. Most of that is the one
synchronous device-to-host copy waiting for the GPU to finish the step: on MPS the copy is synchronous, and
the profiler charges the wait to it. On CUDA that copy is asynchronous (`non_blocking`, pinned buffer, event).

## 4. What the profiling found, and what was changed

The first benchmark run measured `hooks_every_step` at **+95%** on ResNet-50 (Apple M2 GPU, batch 16,
447 ms baseline step). That number made `agent_governed` sample only one step in 20. Profiling showed
that the cost was the **number of kernel launches**, not arithmetic:

| finding | measurement (ResNet-50, MPS) | change |
|---|---|---|
| per-layer statistics launched ~25 tiny kernels per layer per hook | 61,534 ops per monitored step vs 3,576 for training itself | hooks now only copy a strided sample into a staging buffer row (one kernel); all layers' statistics are computed together at the end of the step in a few dozen batched kernels; ops per monitored step: **15,475** |
| the parameter collector looped over 161 parameters, several kernels each | parameters alone: +53% | fused multi-tensor kernels (`torch._foreach_norm`, `_foreach_copy_`, `_foreach_sub_`) |
| on MPS, `torch.linalg.vector_norm` is ~20× slower than a multiply-and-sum (103 ms for 161 gradient norms vs 4.8 ms) | parameters: +74% even with fused calls | norms by per-tensor multiply-and-sum (MPS) / dot product (CPU, 3× faster than `foreach_norm`); fused `foreach_norm` kept on CUDA |
| result | | `hooks_every_step` **+95% → +19%** on ResNet-50/MPS; parameter part **+74% → +4.8%** |
| remaining cost: the exact NaN/Inf reduction over every sampled tensor | +18.8% → +8.0% without it | new option `HookConfig(nonfinite_check="sampled")`; the default stays `"exact"` |

The memory check also found a real leak. With two models monitored at once (a GAN), the two `backward()`
wrappers were restored in install order rather than reverse order, which left one wrapper installed. It then
kept recording every later loss: **256 B of GPU memory per step**, plus a small time cost. Fix: one shared
`backward()` wrapper that monitors register with and unregister from in any order. Regression test:
`test_overlapping_monitors_restore_patches_in_any_order`. The same restore-only-if-still-ours rule now
applies to the `zero_grad()` and `Optimizer.__init__` patches.

## 5. How the 5% guarantee works

`OverheadGovernor` (`runtime/profiling.py`) keeps the amortised cost under the budget during a live run:

1. **Calibrate:** monitor every step for 8 steps and measure step durations between optimizer post-hooks.
   That duration includes snapshot packing and publishing.
2. **Probe:** pause monitoring for 6 steps; t_u is the median of the last 5 unmonitored steps.
3. **Choose N:**

   ```
   cost = median(monitored step) − t_u  [+ median(step right after a monitored one) − t_u, for deferred CUDA copies]
   N    = ceil(cost / (0.6 × budget × t_u))
   ```

   The 0.6 margin aims at 3%, so that noise in the step timings (±2 points on a single repetition on MPS)
   stays under 5%. N is capped at 100: a model is always monitored at least once every 100 steps.
4. **Repeat** the probe every 200 steps, so N follows changes in the workload.

Medians make the estimate robust to occasional slow steps (validation passes, data loading). When every-step
monitoring already fits the budget, N stays 1.

Diagnostic script runs (`monitor_script`) monitor every step by default (`script_overhead_budget=None`):
they are short, bounded runs whose purpose is detection. Set `script_overhead_budget=0.05` to govern them too.

## 6. Limits of this benchmark

- **No CUDA measurements.** No NVIDIA GPU was available for these runs. On CUDA, kernel launches are much
  cheaper than on MPS, and the parameter statistics use fused multi-tensor kernels, so lower every-step
  overhead is expected. This is not measured. Run:
  ```bash
  python examples/benchmark_hooks.py --device cuda --json bench_results/cuda.json --markdown bench_results/cuda.md
  ```
  The runner exits with code 1 if the governed overhead is above the budget or GPU memory leaks, so it can
  serve as a CI gate on a GPU runner.
- **Small models are a different regime.** On models whose step takes well under a millisecond, a fixed cost
  of a few microseconds per step remains even when the governor samples rarely: hook dispatch, the
  `backward()` wrapper and the step hooks. `--quick` uses such models as a smoke test and does not apply the
  budget. The governor is capped at N = 100.
- **Synthetic data, single process.** There is no data loading, no distributed training and no mixed
  precision. Real input pipelines make steps longer, which lowers the relative overhead.
- **Timed steps synchronise the device.** This makes every mode's GPU time count fully. Real training
  loops usually synchronise at least once per step anyway, e.g. with `loss.item()`.

## 7. Reproduce

```bash
pip install -e ".[torch]"
python examples/benchmark_hooks.py                                  # best device: cuda > mps > cpu
python examples/benchmark_hooks.py --device cpu --workloads mlp     # one workload
python examples/benchmark_hooks.py --steps 20 --reps 5 --json out.json --markdown out.md
python examples/benchmark_hooks.py --quick                          # tiny models, seconds (smoke test)
python -m pytest tests/test_runtime_monitor_agent.py -k "memory or detach or governor or profiler or benchmark"
```
