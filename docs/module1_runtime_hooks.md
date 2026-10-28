# Module 1: runtime hooks (perception layer of the Runtime Monitor Agent)

**Owner:** Person 2 (Rishikesh) · **Code:** `silent_bug_agent/runtime/` · **Tests:** `tests/test_runtime_hooks.py`

## The task

> Implement non-intrusive forward, backward, and optimizer hooks that serve as sensory perception tools for the
> Runtime Monitor Agent. Capture gradient norms, activation distributions, weight deltas, and loss dynamics into
> `TrainingState` snapshots without modifying user model source code. Adapt the PyTorch hook collectors and
> zero-copy streaming logic from package101 (`src/trainmedic/detect/tracer.py`, `numerics.py`).
> Deliverable: `silent_bug_agent/runtime/hooks.py` with standalone validation across standard neural architectures.

## What it does

`RuntimeHooks` attaches to an existing PyTorch model and optimizer. The user's model code and training loop do not
change. At every monitored training step it produces one `TrainingState` (the Module 0 contract in
`silent_bug_agent/invariants/base.py`), which the five existing invariants check unchanged. It also produces one
snapshot with `is_eval_phase=True` for each validation pass.

Every hook is removed by `detach()`. Training is bit-for-bit identical with and without the hooks, as the validation
below checks.

## Usage

```python
from silent_bug_agent import get_default_registry
from silent_bug_agent.runtime import HookConfig, RuntimeHooks

hooks = RuntimeHooks(model, optimizer, HookConfig(every_n_steps=1),
                     invariants=get_default_registry(),           # optional: check every snapshot
                     on_failure=lambda state, result: print(state.step, result))
hooks.attach()                      # or: with RuntimeHooks(model, optimizer) as hooks:

for step, (x, y) in enumerate(train_loader):   # unchanged training loop
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()                # <- a training TrainingState is produced here

    if step % 500 == 0:             # unchanged validation pass
        model.eval()
        with torch.no_grad():
            model(x_val)            # <- an is_eval_phase=True TrainingState is produced here
        model.train()

hooks.detach()
hooks.failures                      # [(step, InvariantResult), ...] when invariants=... was given
```

The Runtime Monitor Agent can consume the snapshots from another thread. The stream never blocks training:

```python
for state in hooks.stream:          # blocks until the next snapshot; ends after hooks.stream.close()
    ...
hooks.stream.get(timeout=1.0)       # one snapshot, or None
hooks.stream.subscribe(callback)    # push style; returns an unsubscribe function
hooks.latest                        # newest TrainingState
```

Without an optimizer (manual weight updates), call `hooks.end_step()` after each update. To report a loss
explicitly, call `hooks.observe_loss(loss)`; otherwise losses are captured from `backward()` automatically.

The monitoring rate can change while attached: `hooks.set_every_n_steps(n)`, `hooks.pause()` and `hooks.resume()`
(the Runtime Monitor Agent's overhead governor uses them). `detach()` removes every hook and also frees every device
buffer the hooks allocated (weight samples, pinned host buffers), so nothing stays in GPU memory after it.

## What goes into `TrainingState`

| field | how it is filled |
|---|---|
| `step` | number of optimizer steps completed (1 for the first step) |
| `loss_history` | every scalar passed to `backward()` (mean per step), most recent `max_loss_history` (10,000) values |
| `grad_norms` | `{param: L2 norm of its gradient}` taken just before `optimizer.step()` (exact; fused `torch._foreach_norm` on CUDA, per-tensor dot products on CPU/MPS) |
| `weight_deltas` | `{param: ||W_after - W_before||}` for that step. Covers *every* model parameter, so a trainable layer left out of the optimizer shows `0.0` |
| `requires_grad` | `{param: bool}` |
| `activation_stats` | `{"name[Class]": {...}}` from forward hooks on every leaf layer (see below) |
| `module_modes` | `{"name[Class]": "train" or "eval"}` for every module, at snapshot time |
| `is_eval_phase` | `True` for snapshots taken from a forward under `torch.no_grad()` |
| `extra` | new optional field; see below |

**`activation_stats` keys per layer:**
- The contract's keys: `mean`, `std`, `nan_count`, `inf_count`, `zero_frac`.
- Distribution extras: `min`, `max`, `abs_mean`, `saturated_frac` (tanh/sigmoid layers: fraction at |y|≥0.99, or y≤0.01 / y≥0.99), `norm`, `numel`, `calls`.
- `dead_unit_frac` (ReLU / ReLU6 layers, batches of at least 16 samples): the fraction of units, meaning every non-batch position, that are zero for every sample in the batch. With fewer samples, "zero for every sample" is weak evidence, so it is not measured. A partly dead layer keeps its overall `zero_frac` below the 90% `activation_health` threshold, but this shows the dead units. Added for the Runtime Monitor Agent.
- Gradient flowing back through the layer, from a backward hook on its output: `grad_norm`, `grad_abs_mean`, `grad_abs_max`, `grad_nan_count`, `grad_inf_count`.

**`extra` (training snapshots):**
- `loss`: `{value, ema, z, spike, n_backward}`. `spike` comes from the EMA z-score detector.
- `global_grad_norm`, `global_weight_norm`, `global_delta_norm`, `update_ratio` (‖ΔW‖/‖W‖).
- `weight_norms` and `grad_abs_max` (exact max \|gradient\|) per parameter.
- `lrs` (one per parameter group).
- `histograms` per layer, only when `histogram_bins > 0`.

Labels use the convention of `examples/torch_adapter.py` (`"bn1[BatchNorm1d]"`), so `train_eval_mode_mismatch` can
match "batchnorm" and "dropout".

Layers are selected as leaf modules, plus `nn.MultiheadAttention` and RNN modules (LSTM/GRU/RNN), each treated as a
single unit. `nn.Identity`, `nn.Flatten` and loss modules are skipped. A custom predicate can be passed as
`HookConfig(layers=lambda name, module: ...)`.

## Configuration (`HookConfig`)

| option | default | meaning |
|---|---|---|
| `every_n_steps` | 1 | monitor one step in N (non-monitored steps cost nothing) |
| `layers` | `"auto"` | layer selection rule (see above) |
| `max_layers` | 256 | cap on hooked layers |
| `max_elems` | 16,384 | activations/gradients larger than this are sampled with a strided view (was 65,536 before the batched statistics) |
| `nonfinite_check` | `"exact"` | `"exact"`: one extra reduction over every sampled tensor, so a NaN/Inf anywhere is counted; `"sampled"`: only inside the sample (cheaper on GPUs, see below) |
| `max_param_elems` | 1,048,576 | weight elements sampled per tensor to measure the delta |
| `histogram_bins` | 0 | > 0 adds normalised histograms to `extra["histograms"]` |
| `track_activation_grads` | True | backward hooks on layer outputs |
| `capture_backward_loss` | True | wrap `backward()` to record losses |
| `eval_snapshots` | `"first"` | `"first"` = one per validation pass, `"every"` = every no-grad forward, `"off"` |
| `max_loss_history` | 10,000 | length of `loss_history` (`None` = unlimited) |
| `stream_capacity` | 1,024 | snapshots kept in the ring buffer before the oldest are dropped |
| `max_pending` | 4 | asynchronous device-to-host copies in flight before waiting |
| `loss_z`, `loss_warmup` | 6.0, 10 | spike threshold and warm-up of the loss detector |

## How it works

**Sensors.** There are five kinds:
- `register_forward_hook` on each selected layer.
- `Tensor.register_hook` on each layer's output, which sees the gradient and never changes it.
- `optimizer.register_step_pre_hook` and `register_step_post_hook`.
- A wrapper around `torch.Tensor.backward`, installed only while attached.
- A forward hook on the root model, which detects validation passes.

**Low overhead.** These ideas are adapted from package101's tracer:
- **Statistics stay on the device, and are computed in one batch.** On a GPU, the cost of monitoring is the number of kernel launches, not the arithmetic. So a hook only copies a strided sample of its tensor into its own row of a staging buffer that is allocated once (one kernel). The sample is at most `max_elems` elements and keeps the batch dimension so dead units can be measured. At the end of the step the statistics of all layers are computed together in a few dozen batched kernels, instead of about 25 kernels per layer. Activations themselves are never kept or moved.
- **Large tensors are sampled.** Above `max_elems`, statistics come from the strided sample, and norms are scaled by the sampling ratio. With `nonfinite_check="exact"` (the default), one extra reduction over the whole tensor still catches a NaN or Inf outside the sample (if both occur only outside the sample, only `nan_count` is raised).
- **Fused parameter statistics.** Gradient norms, max |gradient|, weight norms and weight deltas for all parameters use a handful of multi-tensor kernels (`torch._foreach_*` on CUDA). On MPS, whose norm kernel is about 20× slower than a multiply-and-sum, and on CPU, per-tensor dot products are used. ResNet-50 on an Apple M2 GPU: the parameter part went from +74% to +4.8% of step time, and every-step monitoring from +95% to +19% (+8% with `nonfinite_check="sampled"`). Measured in [overhead_benchmark.md](overhead_benchmark.md).
- **No GPU synchronisation inside hooks.** No `.item()`, `float()` or `.tolist()`. At the end of a monitored step all statistics are packed into one small vector and copied to the host once, asynchronously, into a reused pinned buffer. The `TrainingState` is built when the copy lands, at most `max_pending` steps later.
- **One extra copy.** A strided sample of each weight tensor is taken just before `optimizer.step()`, into a buffer allocated once (one fused copy), so the delta can be measured afterwards. The old adapter keeps a full copy of every parameter and calls `.item()` several times per layer each step.
- **Non-blocking stream.** `SnapshotStream` is a bounded, thread-safe ring buffer. If the consumer falls behind, the oldest snapshots are dropped and counted in `stream.dropped`.

**Loss dynamics.** `runtime/numerics.py` is a pure-Python port of package101's EMA z-score detector. It flags a loss
spike without letting the spike inflate its own statistics.

**torch.compile.** Hooks inside a region being compiled return immediately, so they add nothing to the compiled
graph. Optimizer, loss and gradient-norm signals keep working for compiled models; this is validated.

## Validation

```bash
python -m silent_bug_agent.runtime.validate --steps 20 --json report.json   # add --device cuda on a GPU machine
python -m pytest tests/test_runtime_hooks.py -v
```

These are the results on CPU (torch 2.14.1, 20 steps per architecture).

**Architectures.** Every architecture passed every check:
- Training is bit-identical with and without hooks.
- One snapshot is produced per step.
- The loss, every per-parameter gradient norm and every per-parameter weight delta match an independent measurement (relative tolerance 1e-3).
- Activation and gradient statistics cover all selected layers.
- No critical invariant failure is reported on the clean run.
- `detach()` leaves no hook behind.

| architecture | hooked layers | result |
|---|---|---|
| MLP (Linear, ReLU, Dropout) | 6 | pass |
| ResNet (Conv, BatchNorm, in-place ReLU, skip connections) | 15 | pass |
| LSTM, 2 layers | 3 | pass |
| GRU, 2 layers | 3 | pass |
| Transformer encoder (`nn.TransformerEncoder`) | 18 | pass |
| GPT-style causal decoder (`nn.MultiheadAttention` + mask) | 16 | pass |
| Autoencoder (tanh / sigmoid) | 4 | pass |

**Planted bugs.** These are the scenarios of `examples/planted_bugs_torch.py`, run through `RuntimeHooks` with
the default invariants. The verdicts and invariant counts are identical to the old adapter's:

| scenario | expected | fired (count) | result |
|---|---|---|---|
| none | no critical failure | loss_trend 5 (warning) | pass |
| eval_mode | train_eval_mode_mismatch | train_eval_mode_mismatch 4, loss_trend 13 | pass |
| frozen | frozen_weights | frozen_weights 181, loss_trend 8 | pass |
| high_lr | gradient_norm_bounds or activation_health | gradient_norm_bounds 3, activation_health 1 | pass |
| dead_relu | activation_health | activation_health 204, gradient_norm_bounds 200, frozen_weights 181, loss_trend 114 | pass |
| random_labels | loss_trend | loss_trend 113 | pass |

**Extra signals.** All passed:
- A NaN in the input raises `nan_count` and a critical `activation_health` failure.
- Saturated tanh gives `saturated_frac` ≥ 0.95.
- A scaled batch is flagged as a loss spike at exactly that step.
- A learning rate of 1e4 gives `update_ratio` > 1.
- Histograms are normalised.
- A consumer thread receives every snapshot.
- A `torch.compile`d model still produces correct snapshots.

**Overhead.** Median per-step time on CPU with 4 threads, relative to no monitoring:

| model | step time | old adapter (`examples/torch_adapter.py`) | hooks, every step | hooks, every step, no activation gradients | hooks, every 10th step |
|---|---|---|---|---|---|
| CNN, 7 conv layers × 64 ch, 32×32, batch 64 | 433 ms | 1.34× | 1.15× | 1.08× | 1.00× |
| MLP 784-2048-2048-10, batch 256 | 11 ms | 1.75× | 1.98× | 1.79× | 1.01× |
| validation's toy models (sub-millisecond steps) | – | – | 1.3–5.9× | – | – |

On tiny models, the fixed cost per hook dominates. For long runs use `every_n_steps` of 10 or more. On a GPU, the
hooks should compare better still, because unlike the adapter they never stall the GPU. This has not been measured
yet.

## Change to a shared contract

`TrainingState` gained one optional field, `extra: Dict[str, Any]`, with an empty-dict default. It holds
producer-specific data the invariants don't need: loss dynamics, global norms, learning rates and histograms.
Existing code and tests that build `TrainingState` are unaffected; all 69 pre-existing tests pass. This follows the
`extra` fields already used in `schemas.py`.

## Relation to `examples/torch_adapter.py`

The adapter was the stopgap that "Module 1 will replace". It is left in place so existing instructions keep working.
`RuntimeHooks` produces the same keys and labels, so the invariants behave the same, and it adds:
- no manual `snapshot()` call
- automatic validation-phase snapshots
- per-layer gradient flow
- loss dynamics
- no GPU synchronisation
- a stream for the agent

## Known limitations

- **Not yet run on a CUDA GPU.** Run `python -m silent_bug_agent.runtime.validate --device cuda` on a GPU machine before relying on it there.
- **Activations inside compiled regions are not recorded.** Hooks are skipped while `torch.compile` traces.
- **Last call wins for repeated modules.** A module called several times in one step (shared layers, gradient accumulation) reports its last call; `calls` counts them.
- **One optimizer.** Only the optimizer passed to `RuntimeHooks` marks step boundaries. Multi-optimizer setups such as GANs should use one `RuntimeHooks` per model/optimizer pair.
- **`backward()` is patched globally while attached.** It is restored on `detach()`.

## Runtime Monitor Agent (built on these hooks)

The agent is built: [runtime_monitor_agent.md](runtime_monitor_agent.md).
- It runs in its own thread and subscribes to `hooks.stream`.
- It checks every snapshot with the Module 0 suite, the package101 runtime checks and a trend analyzer.
- It turns violations into `Finding`s for `AgentState.detection_report`. Each finding carries the step and a trace
  of that moment (the relevant `activation_stats` and `extra` entries).
- Overhead and memory are benchmarked in [overhead_benchmark.md](overhead_benchmark.md).

## Provenance

The design is adapted from package101 (https://github.com/rishikeshwalhekar49-lab/package101):
- `src/trainmedic/detect/tracer.py`: hook installation, compile-awareness, loss capture via `backward()`, and optimizer step hooks.
- `src/trainmedic/detect/numerics.py`: the EMA z-score spike detector.
- The package101 branch `silent-bug-agent-runtime-hooks`: the on-device statistics, async packing and ring buffer.

All of it is rewritten here against this repository's `TrainingState` contract, labels and Python 3.9 baseline.
