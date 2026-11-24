# The trace classifier: a small trained model for diagnosis

The pipeline's LLMs are not what makes a run slow: offline, validation takes most of the
time (see `Time:` at the end of every `fix` run). So the trained model does not replace
them. It does the job a small model is good at: recognising, from the numbers a training
run produces, which known bug the run looks like. It is a softmax regression of 350
weights, a 6.8 KB file inside the package, needs no key and no network, and one
prediction takes about 10 µs on a laptop CPU.

![Leaderboard: every model ranked on the same runs](images/leaderboard.png)

## What it reads

`script_runner --trace` (`silent_bug_agent/learned/trace.py`) records, for at most 60
optimizer steps, only numbers, never tensors. `learned/features.py` turns them into 34
scale-free numbers (ratios, fractions and logs).

| Signal | Points to |
|---|---|
| gradient norm per step, and its growth | missing `zero_grad()` (gradients pile up) |
| `zero_grad()` calls per step | missing `zero_grad()` |
| share of parameters without a gradient, how far each step moved the weights, tensors that never moved | frozen layer, dead units, learning rate |
| share of the model's trainable parameters that **no optimizer updates** | a layer left out of the optimizer |
| learning rate | learning rate |
| model outputs inside [0, 1] whose rows sum to 1; `cross_entropy` given probabilities | softmax before the loss |
| share of ReLU outputs that are zero (`nn.ReLU`, `F.relu`, `torch.relu`) | dead ReLUs |
| scale of the inputs | inputs not normalised |
| validation passes in training mode, **counted only when the model has dropout or batch norm** | `model.eval()` missing |
| every loss-function call: its `reduction`, the training loss against the loss of guessing (ln of the number of classes), the validation loss, its trend and its gap to the training loss | `reduction="sum"`, shuffled labels, and a healthy run that simply learns slowly |

## The data

There is no public dataset of buggy training runs, so the training data is generated, and
the test data is written separately.

**Training: generated scripts** (`silent_bug_agent/mutation/`). `templates.py` writes 80
healthy training scripts that differ in what changes a trace (task, MLP or CNN,
`nn.Sequential` or a class, depth, width, activation, dropout, batch norm, four
optimizers, batch size, DataLoader or manual slicing; some leave out `train()`/`eval()`
when the model has no dropout or batch norm, as real code does), plus a slower-learning
healthy version of each. `operators.py` then plants one bug at a time by rewriting the
syntax tree, several of them in two ways (a frozen layer through `requires_grad` or by
leaving it out of the optimizer; a softmax inside the model or in the loss call).
781 traced runs: 160 healthy, 621 with one bug.

**Tuning and test: hand-written scripts** (`examples/holdout/`). Ten training scripts
written separately from the generator, each in a different style: training functions, a
CNN with a scheduler, residual regression, an EmbeddingBag text model, a `Trainer` class,
binary classification with logits, an autoencoder, a GRU, raw tensors with no `nn.Module`,
and parameter groups. Each has two healthy versions and its bugs planted by hand-written
edits (`variants.py`), often in ways the operators never use: `zero_grad()` called once
before the loop, labels reversed, `train()` mode during validation, gradients cleared by
hand. Scripts 01 to 04 (39 runs) are for **tuning**: model settings are chosen on them,
and only their mistakes are studied. Scripts 05 to 10 (58 runs) plus blind copies of the
project's 18 benchmarks and their healthy twins (26 runs) are the **test set**: 84 runs,
29 of them healthy, never used to choose anything.

They were written by the same author as the generator, in deliberately different styles.
That makes them a fair test of generalisation, not an independent one.

```bash
python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl   # about 15 minutes
python -m silent_bug_agent.learned.train --data traces.jsonl
```

## The model

`train.py` tries a softmax regression and networks with one tanh hidden layer of 16 or
32 units, each with two strengths of L2 regularisation, and keeps the settings with the
best accuracy on the tuning scripts, averaged over three seeds (on a tie, the simpler
model). The plain softmax regression tied for best (92.3%) and was kept. It is numpy only
(`learned/model.py`); the weights are plain JSON in `silent_bug_agent/learned/weights.json`.

## Results

Every model gets the same 34 features and chooses its settings on the same tuning
scripts, from a small grid each; every learned model is retrained with 5 seeds, and the
scores are means. `examples/compare_models.py` produces all of it, and
`examples/plot_leaderboard.py` and `examples/plot_comparison.py` draw the images.

| Rank | Model | Hand-written test (84 runs) | Generated (781 runs) | Healthy flagged (of 29) | Size | Per prediction |
|---|---|---|---|---|---|---|
| 1 | **Trace classifier (ours, shipped)** | **85.7%** | 91.7% | 6 | **4.8 KB** | **11 µs** |
| 2 | Trace classifier with a hidden layer (ours) | 83.8% | 92.0% | 7 | 8.3 KB | 13 µs |
| 3 | Random forest (300 trees) | 83.3% | 96.6% | 8 | 2.1 MB | 6.1 ms |
| 4 | SVM (RBF kernel) | 83.3% | 89.6% | 9 | 167 KB | 144 µs |
| 5 | k-nearest neighbours | 82.1% | 87.8% | 8 | 217 KB | 374 µs |
| 6 | Hand-written rules (the project's detectors) | 59.5% | 67.9% | 10 | no model file | 7 s, including a 150-step run |
| 7 | Gradient boosting | 58.3% | 96.8% | 16 | 1.7 MB | 19.9 ms |
| 8 | Always the commonest label | 34.5% | 20.5% | 0 | none | none |

Data: `images/comparison.json`. The weights file is 6.8 KB on disk because it also stores how it
was trained; 4.8 KB is the model itself.

![Accuracy by model, and accuracy against size](images/model_comparison.png)

What it says:

- On the hand-written test the shipped model is first, at 85.7%, ahead of the random
  forest and the SVM (83.3%) while 35 to 450 times smaller. With 84 runs, 2 to 4 points
  is within the noise (the 95% interval around 85.7% is about 77% to 92%): say "on par or
  slightly ahead", not "better".
- On generated scripts the random forest and gradient boosting are clearly better (about
  97% to 92%). Gradient boosting then collapses on hand-written scripts (58%): it learned
  the generator's habits, not the bugs.
- It still flags about 1 in 5 healthy runs. That is why, in the pipeline, it only advises.

### How it got here

| Version | What changed | Tuning scripts (39) | Hand-written test (58) | Project benchmarks (26) |
|---|---|---|---|---|
| 1 | 24 features, trained on 701 generated runs | not measured | not measured | 81.5% |
| 2 | + 8 features from the loss calls, + slow healthy runs | 76.9% | 65.5% | 84.6% |
| 3 (shipped) | eval() signal only where it matters, + parameters outside the optimizer, + more ways to plant two bugs, settings chosen on the tuning scripts | 92.3% | 82.8% | 92.3% |

The fixes in version 3 come from the mistakes on the **tuning** scripts: healthy scripts
without dropout flagged for a missing `eval()`, and layers left out of the optimizer
going unseen. To be exact about it, the test set was scored twice, after version 2 and
after version 3, and the same two mistakes were visible there too. One mistake seen only
on the test set was deliberately not fixed, so that the test number stays honest:
**gradients cleared by hand** (`p.grad = None`, test script 09) look like a missing
`zero_grad()`, because the tracer counts `zero_grad()` calls. That is the first thing to
fix next, checked against a new test set.

## How the pipeline uses it, and why only as advice

While the runtime monitor runs, the script is traced once more in parallel
(`learned/prior.py`). What happens to the guess depends on the mode
(`fix --learned`, `diagnose --learned`, `PipelineConfig(learned=...)`):

| Mode | What the classifier does |
|---|---|
| `advisory` (default) | its top guesses are printed in the report (`Trace classifier: ...`) and kept in the JSON result; nothing else |
| `active` | also: a guess with p ≥ 0.9 of a bug no detector reported becomes a runtime finding, its probabilities nudge the offline ranking of causes by at most 0.3, and the diagnosis LLM is given them as evidence |
| `off` | no traced run |

`active` was measured end to end with version 1 on the blind benchmarks, and it made
things worse (8 of 18 repaired instead of 9; right cause first 13 instead of 14), because
a confident wrong guess pushed the detectors' right answer down the list. That has not
been re-measured with version 3. Until it is, advisory stays the default. In any mode the
classifier cannot change a file by itself: every patch still has to pass both gates.

## Limits

- It knows nine bugs and "clean". A bug outside that list gets whichever of them its
  trace looks most like, or "clean".
- It flags about 1 in 5 healthy runs on the hand-written test.
- Gradients cleared by hand look like a missing `zero_grad()` (see above).
- The hand-written scripts and the generator have the same author.
- The trace needs one extra run of 60 steps (about 4 s, in parallel with the monitor).
- LLMs (gpt-oss) have not been compared yet: `compare_models.py --llm` needs a Groq key.
