# New Silent DL Bug Benchmarks (`examples/new/`)

This directory contains 10 newly generated benchmarks covering silent deep learning bugs from the project's taxonomy (`silent_bug_agent/taxonomy/bug_taxonomy.yaml`).

Each benchmark script is self-contained, runs on CPU using synthetic tensors without external dataset downloads, accepts `--seed`, `--fix`, and `--steps`, and logs standard `Final Loss: ...` and `__METRICS__:{...}` machine-readable outputs.

---

## Benchmark Suite Overview

| File | Bug Name | Taxonomy ID | Category | Detection Method |
|---|---|---|---|---|
| `benchmark_09_lr_scheduler_step_order.py` | LR Scheduler Step Order | `lr_scheduler_step_order_bug` | `api_misuse` | AST (`LrSchedulerStepOrderRule`) + Runtime |
| `benchmark_10_gradient_accumulation_miscount.py` | Gradient Accumulation Miscount | `gradient_accumulation_miscount` | `training_dynamics` | AST (`GradientAccumulationMiscountRule`) |
| `benchmark_11_duplicate_augmentation.py` | Duplicate Augmentation Applied Twice | `duplicate_augmentation_applied_twice` | `data` | AST (`DuplicateAugmentationRule`) |
| `benchmark_12_silent_device_fallback.py` | Silent Device Fallback | `silent_device_fallback` | `environment` | AST (`SilentDeviceFallbackRule`) |
| `benchmark_13_optimizer_state_leak_across_folds.py` | Optimizer State Leak Across Folds | `optimizer_state_leak_across_folds` | `training_dynamics` | AST (`OptimizerStateLeakAcrossFoldsRule`) |
| `benchmark_14_label_data_misalignment.py` | Label/Data Desync Shuffling | `label_data_misalignment` | `data` | Semantic LLM + Runtime Loss Monitor |
| `benchmark_15_exploding_gradients.py` | Exploding Gradients (Unclipped) | `exploding_gradients` | `training_dynamics` | Invariant (`gradient_norm_bounds`) + Recipe |
| `benchmark_16_vanishing_gradients.py` | Vanishing Gradients (Deep Sigmoids) | `vanishing_gradients` | `training_dynamics` | Invariant (`gradient_norm_bounds`, `frozen_weights`) |
| `benchmark_17_mismatched_loss_task.py` | Mismatched Loss for Task (MSE on Classification) | `mismatched_loss_task` | `model_logic` | Semantic LLM + Causal Engine |
| `benchmark_18_data_leakage_train_test.py` | Train/Test Data Leakage in Preprocessing | `data_leakage_train_test` | `data` | Semantic LLM |

---

## Detailed Descriptions

### 1. `benchmark_09_lr_scheduler_step_order.py`
- **Bug**: `scheduler.step()` is called before `optimizer.step()`.
- **Why it is silent**: The training loop runs without throwing an exception. However, PyTorch skips the initial scheduled learning rate, skewing the entire decay curve forward by 1 step.
- **Fix**: Call `optimizer.step()` before `scheduler.step()`.

### 2. `benchmark_10_gradient_accumulation_miscount.py`
- **Bug**: `optimizer.zero_grad()` is placed unconditionally outside the accumulation interval check `if (step + 1) % accum_steps == 0:`.
- **Why it is silent**: The model continues to train and compute backward passes, but gradient accumulation is discarded on every sub-step, applying only the final mini-batch's gradient.
- **Fix**: Place gradient clearing inside the accumulation block directly after `optimizer.step()`.

### 3. `benchmark_11_duplicate_augmentation.py`
- **Bug**: Duplicate `RandomHorizontalFlip(p=0.5)` is added twice in `transforms.Compose`.
- **Why it is silent**: Runs without error, but double random flipping flips 50% twice, reverting 25% of images back to original orientation and altering the expected augmentation probability distribution.
- **Fix**: Remove the redundant transform from the composition pipeline.

### 4. `benchmark_12_silent_device_fallback.py`
- **Bug**: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` is constructed, but neither the model nor input tensors are moved to `device`.
- **Why it is silent**: Tensors remain silently on CPU default without raising an error, causing unexpected slow execution on GPU systems.
- **Fix**: Explicitly move model and data tensors via `.to(device)`.

### 5. `benchmark_13_optimizer_state_leak_across_folds.py`
- **Bug**: In a cross-validation loop over folds, optimizer and model weights are initialized outside the fold loop and never reset.
- **Why it is silent**: All folds train and loss decreases, but momentum buffers and weight updates from previous folds contaminate subsequent fold evaluations.
- **Fix**: Reinitialize model and optimizer state freshly at the start of each fold.

### 6. `benchmark_14_label_data_misalignment.py`
- **Bug**: Features and labels are shuffled independently using two distinct `torch.randperm()` calls.
- **Why it is silent**: Tensor dimensions match and loss converges slightly to mean baseline, but true input-output relationship is destroyed and generalization fails.
- **Fix**: Use a single shared permutation index tensor for both `x` and `y`.

### 7. `benchmark_15_exploding_gradients.py`
- **Bug**: Deep linear network without normalization or clipping causes gradient norms to grow exponentially during backpropagation.
- **Why it is silent**: Operates fine for tiny initial steps, then gradients explode to Inf/NaN.
- **Fix**: Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

### 8. `benchmark_16_vanishing_gradients.py`
- **Bug**: Deep 8-layer MLP using `nn.Sigmoid()` without residual connections or normalization.
- **Why it is silent**: Code trains without errors, but gradients decay by $(0.25)^8 \approx 1.5 \times 10^{-5}$, starving initial layers of gradient updates.
- **Fix**: Replace saturated sigmoids with `nn.ReLU()` activations.

### 9. `benchmark_17_mismatched_loss_task.py`
- **Bug**: Using `nn.MSELoss()` on discrete integer class labels $\{0, 1, 2, 3\}$.
- **Why it is silent**: PyTorch computes numeric MSE between scalar prediction and integer class index without errors, but treats categorical identities as arbitrary Euclidean distances.
- **Fix**: Output 4-class logits and optimize with `nn.CrossEntropyLoss()`.

### 10. `benchmark_18_data_leakage_train_test.py`
- **Bug**: Computing feature normalization (mean and std) over the combined train + test dataset before partitioning.
- **Why it is silent**: Code executes cleanly, but evaluation metrics are artificially optimistic because the test partition's distribution leaks into training normalization.
- **Fix**: Fit feature statistics exclusively on the training split, and apply those fitted statistics to transform the test split.

---

## Running the Benchmarks

Run any benchmark in buggy mode:
```bash
python examples/new/benchmark_09_lr_scheduler_step_order.py
```

Run with the verified fix:
```bash
python examples/new/benchmark_09_lr_scheduler_step_order.py --fix
```

Run static analysis scan:
```bash
python -m silent_bug_agent scan examples/new/
```
