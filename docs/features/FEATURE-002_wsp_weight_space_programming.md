# FEATURE-002: Weight-Space Programming (WSP)

## Goal & Research Context
Implements the paradigm introduced by Takahashi, Teoh, Zhang, and Khattab (MIT CSAIL, Nov 2026): *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*. 

Instead of debugging models exclusively through Python source code (AST) and gradient descent, WSP enables direct, gradient-free inspection, diagnosis, and surgical editing of raw neural network parameters ($W$) alongside code space.

---

## Architectural Pillars

### 1. Simultaneous Dual Perception
Runs code-space analysis and weight-space tensor perception simultaneously:
- **Code Space**: AST static rules (`static/`) + live runtime hook telemetry (`runtime/`).
- **Weight Space**:
  - **SVD / Spectral Decomposition**: Computes singular value distributions $\sigma_i$, condition numbers $\kappa = \sigma_{max} / \sigma_{min}$, and Shannon-entropy effective rank:
    $$p_i = \frac{\sigma_i}{\sum \sigma_j}, \quad H = -\sum p_i \ln(p_i), \quad r_{eff} = \exp(H)$$
  - **Kurtosis & Outlier Profiling**: Heavy-tailed outliers ($kurtosis > 30$) responsible for attention/reasoning loops.
  - **Neuron & Bias Collapse**: Catastrophic negative biases (e.g. $bias \le -15.0$) causing dead ReLUs.

### 2. Surgical Weight-Space Operations (`wsp/editor.py`)
- **Subspace Excision**:
  $$W' = W(I - v v^T) = W - (W v) v^T$$
  Removes an identified toxic/loop projection vector $v$ without perturbing orthogonal features.
- **Low-Rank Delta Injection**:
  $$W' = W + \alpha (u v^T)$$
  Injects low-rank capacity to revive collapsed rank.
- **Bias Re-centering**: Surgically centers depressed bias vectors.
- **Outlier Clamping**: Clamps extreme weights above $k \times \sigma$.

### 3. Human-in-the-Loop Approval & Empirical Validation Gate
1. Consolidates code findings and weight findings into a single `DualSolution`.
2. Interactively prompts the user with both code diffs and weight surgery cards ($\Delta W$ norm, parameters affected %).
3. Upon user approval, runs **Gate 1 (Overfit Oracle)** and **Gate 2 (Multi-Seed Statistical Sandbox)**.
4. If validation fails, collects empirical feedback and adapts across up to 3 iterations max.

---

## In-Scope
- Inspection of all PyTorch `nn.Module` models, `.pt`, `.pth`, `.bin`, `.safetensors`.
- SVD spectral analysis, condition number bounding, effective rank collapse detection.
- Subspace projection excision, low-rank delta injection, bias centering, outlier clamping.
- CLI subcommand: `sba wsp <script.py> [--checkpoint] [--dry-run] [--yes] [--events]`.
- Pastel + Ink TUI interactive screen: `tui/commands/wsp.tsx`.

## Out-of-Scope
- JAX / TensorFlow checkpoints.
- Distributed FSDP parameter sharding across multi-node clusters.

---

## Timeline Log
- 2026-11-07 19:40 - Designed WSP architecture (`schemas.py`, `inspector.py`, `editor.py`, `dual_supervisor.py`).
- 2026-11-07 19:50 - Implemented unit tests `tests/test_wsp.py` (6/6 passed, 48/48 full regression passed).
- 2026-11-07 19:58 - Integrated into Pastel + Ink TUI (`tui/commands/wsp.tsx` and `index.tsx`), built and verified exit 0.

