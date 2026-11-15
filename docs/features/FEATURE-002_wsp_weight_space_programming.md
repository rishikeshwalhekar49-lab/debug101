# FEATURE-002: Weight inspection (`wsp`)

## Status: reduced to a read-only inspector on 2026-11-15

The first version (2026-11-07) was described as "Weight-Space Programming": inspect weights, propose "weight surgery", and validate it. A review found that it did not do what this page claimed, so it was cut back to the part that works and can be defended. The reasons are listed below so the decision can be checked.

## What it does now

`python -m silent_bug_agent wsp [script.py] [--checkpoint FILE]` reads a model's weights and reports anomalies. It never changes a script, a model or a checkpoint.

| Check | Severity | Meaning |
|---|---|---|
| NaN or infinite values | critical | training diverged |
| Values above 10,000 | warning | an exploding update, usually just before NaN |
| A weight matrix that is all zero | warning | the layer passes no signal |
| Biases at or below −15 | critical if half or more of the layer, warning from 5%, otherwise a note | a ReLU after this layer cannot fire |
| Effective rank below 15% of the possible rank | warning (critical below 5%) | most units compute the same thing |
| A few values far outside the rest (kurtosis above 30) | warning if one value is 100 or more robust deviations out, otherwise a note | one huge update or a hand-written initialisation |

Effective rank is the exponential of the entropy of the normalised singular values: $p_i = \sigma_i / \sum_j \sigma_j$, $r_{eff} = \exp(-\sum_i p_i \ln p_i)$.

Weights come from a checkpoint (loaded with `weights_only=True`, so a pickle cannot run code), or from the script itself after `--max-steps` (default 60) optimizer steps in the validation sandbox.

## Why the first version was cut back

1. **The "surgery" was never applied or tested.** Weight patches were built and displayed, but the validation gates retrain the script from scratch, so an edit to a weight tensor could not affect the result. The gates were only testing the code diff.
2. **Models were loaded by executing the whole user script inside the tool's own process**, with no time limit and no isolation. A script that trains at import time would train there.
3. **It inspected freshly initialised weights**, which say nothing about a training bug. For the documented example (`benchmark_06_dead_relu.py`) it could not build the model at all and inspected nothing.
4. **False alarms.** Every model with BatchNorm, LayerNorm or zero-initialised biases was flagged ("zeroed out"), and healthy square layers were flagged as "ill-conditioned" purely by chance: 2% of fresh 64-wide `nn.Linear` layers, 4% at 256 and 7% at 512 have a condition number above the old threshold (measured over 100 to 200 layers each).
5. **Its "Gate 2" had no statistics.** It compared two means with no test, and accepted outright when the script printed no loss.
6. **Confidence was hard-coded to 0.92.**
7. **Outlier clamping is harmful as a default.** Large-magnitude weights in trained networks are often the ones that matter most.
8. `torch.load(..., weights_only=False)` on a user-supplied file can execute code.
9. One editing routine used `math` without importing it and would crash on any rank-collapse finding.

## The research reference

The first version cited "Takahashi, Teoh, Zhang, and Khattab (MIT CSAIL)" for "Sparks of Weight-Space Programming". Corrections:

- The first author is Rikiya **Takehi**, and the piece is a [blog post](https://rikiyat.github.io/blog/weight-space-programming/), not a peer-reviewed paper.
- Its claims have been [disputed](https://arxiv.org/html/2611.08070v1).
- It describes an LLM reading and writing weights. This package uses no LLM at all: it computes fixed statistics. It should not be presented as an implementation of that idea.

## Out of scope
- Editing weights. A repair belongs in the training code, where `fix` can test it against both gates.
- JAX / TensorFlow checkpoints.
- Rank checks on matrices larger than 1024 on a side (skipped, with a note).

## Tests
`tests/test_wsp.py` (13 tests): each anomaly type, a healthy model with normalisation layers that must stay clean, checkpoint formats, a malicious pickle that must not execute, weights captured from a script without changing it, and the CLI exit codes.

## Timeline
- 2026-11-07 — first version (`schemas.py`, `inspector.py`, `editor.py`, `dual_supervisor.py`).
- 2026-11-15 — `editor.py` and `dual_supervisor.py` removed; inspector rewritten; weights now captured from a sandboxed run.
