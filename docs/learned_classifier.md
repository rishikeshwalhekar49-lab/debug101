# The trace classifier: a small trained model for diagnosis

The pipeline's LLMs are not what makes a run slow: offline, validation takes most of
the time (see `Time:` at the end of every `fix` run, and the summary line of
`examples/evaluate_pipeline.py`). So the trained model does not replace them. It does
the job a small model is good at: recognising, from the numbers a training run
produces, which known bug the run looks like. It is fast, needs no key and no network,
and adds a few kilobytes to the package. It is honest to say what came of it: it is
accurate on generated scripts, less so on hand-written ones, and letting it steer the
pipeline made the results worse, so by default it only advises (see the end of this page).

## What it reads

`script_runner --trace` (`silent_bug_agent/learned/trace.py`) records, for at most 60
optimizer steps, only numbers, never tensors:

| Signal | Recorded as | Points to |
|---|---|---|
| gradient norm per step, and its growth | `grad_norms` | missing `zero_grad()` (gradients pile up) |
| share of parameters without a gradient | `no_grad_fraction` | frozen layer, dead units |
| how far each step moved the weights | `update_ratios`, `frozen_fraction` | frozen layer, learning rate |
| `zero_grad()` calls per step | `zero_grad_calls` | missing `zero_grad()` |
| learning rate | `lr` | learning rate |
| model outputs inside [0, 1] whose rows sum to 1 | `unit_outputs`, `rowsum_one_outputs` | softmax before the loss |
| `cross_entropy` given probabilities | `ce_probability_inputs` | softmax before the loss |
| share of ReLU outputs that are zero (`nn.ReLU` and `F.relu`) | `relu_zero_fraction` | dead ReLUs |
| scale of the inputs | `input_abs_mean`, `input_std` | inputs not normalised |
| forward passes under `no_grad()` while the model is in training mode | `no_grad_training_forwards` | `model.eval()` missing |
| mean bias, loss curve | `bias_mean`, `losses` | dead ReLUs, divergence, label noise |

`learned/features.py` turns one trace into 24 scale-free numbers (ratios, fractions and
logs), so a model trained on small scripts can say something about other scripts.

## Where the training data comes from

There is no labelled dataset of buggy training runs, and 18 benchmarks are far too few
to learn from, so the data is generated (`silent_bug_agent/mutation/`):

1. `templates.py` writes healthy training scripts that differ in what changes a trace:
   classification or regression, MLP or small CNN, `nn.Sequential` or a class with
   `forward()`, depth and width, ReLU/Tanh/GELU/LeakyReLU, dropout, batch norm,
   SGD/momentum/Adam/AdamW and their learning rates, batch size, DataLoader or manual
   slicing. Every script trains on synthetic data from a fixed teacher, so nothing is
   downloaded.
2. `operators.py` plants one bug at a time by rewriting the script's syntax tree: it
   removes `zero_grad()` or `eval()`, puts a softmax in the model, sets
   `reduction="sum"`, multiplies the learning rate, freezes the first layer, sets the
   biases to −8, scales the inputs to pixel range, or shuffles the labels on their own.
   An operator that does not apply (no ReLU, no dropout or batch norm for `eval()` to
   matter, no CrossEntropyLoss) plants nothing, so every label is exact.
3. `dataset.py` traces every version, healthy ones too, and writes one JSON line per run.

```bash
python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl
python -m silent_bug_agent.learned.train --data traces.jsonl
```

## The model

A softmax regression and a network with one tanh hidden layer of 32 units are both
trained (numpy only, `learned/model.py`), and the one with the better accuracy on
held-out templates is kept and refitted on all the data. The weights are plain JSON in
`silent_bug_agent/learned/weights.json`.

## How good it is (trained 24 November 2026)

Training data: 80 generated templates, 701 traced runs (80 healthy, 621 with one planted
bug). The shipped model is the hidden-layer network (32 units): 12.8 KB of JSON, about
12 microseconds a prediction on a laptop CPU.

| Test | What it measures | Accuracy |
|---|---|---|
| 5-fold cross-validation, whole templates held out | a planted bug in a script it has not seen, written the same way | **93.0%** (softmax regression: 91.3%) |
| The project's benchmarks, blind copies | hand-written scripts it never trained on, 9 buggy + 17 healthy | **80.8%** (21 of 26) |

Recall per bug on held-out templates: 100% for missing `zero_grad()`, learning rate,
unnormalised inputs and softmax before the loss; 97 to 98% for dead ReLUs and a frozen
layer; 92% for a missing `eval()`; 88% for `reduction="sum"`; 84% for shuffled labels;
76% for healthy scripts. Its usual mistake is calling a slow-learning healthy script
"labels shuffled" or "wrong loss reduction": all three have a loss that barely falls.

On the benchmarks it got 7 of the 9 buggy scripts right. It missed benchmark 03 (frozen
layer: that benchmark leaves the layer out of the optimizer instead of setting
`requires_grad=False`, which the operator never does) and benchmark 07 (`reduction="sum"`,
called shuffled labels). It also flagged 3 of 17 healthy twins, two of them with high
confidence (the healthy twin of benchmark 15, at p = 1.0).

That is why a guess only becomes a finding at **p ≥ 0.9**. On held-out templates:

| Threshold | Healthy runs given a finding | Bugs found correctly |
|---|---|---|
| 0.5 | 23.8% | 94.8% |
| 0.8 | 10.0% | 89.2% |
| **0.9** | **3.8%** | **83.1%** |
| 0.95 | 2.5% | 76.8% |

A wrong finding cannot change a healthy file by itself: a patch for a bug that only the
classifier reported can pass only on statistics (the "bug removed and not worse" route
is closed to it), and Gate 2 accepts a patch that does nothing at most 5% of the time.

## Compared with other models

![Accuracy by model, and accuracy against size](images/model_comparison.png)

`examples/compare_models.py` asks every model the same question for the same runs, and
`examples/plot_comparison.py` draws the chart above (data: `images/model_comparison.json`).
The learned models all get the same 24 features; the hand-written rules are the project's
own detectors, run offline exactly as `diagnose --no-llm` runs them.

| Model | Generated scripts (n = 701) | Hand-written (n = 26) | Healthy flagged (of 17) | Size | Per prediction |
|---|---|---|---|---|---|
| **Trace classifier (shipped)** | **93.0%** | 80.8% | 17.6% | 11.7 KB | 14 µs |
| Same, no hidden layer | 91.3% | 84.6% | 5.9% | 3.5 KB | 12 µs |
| Random forest (300 trees) | 94.9% | 80.8% | 23.5% | 6.0 MB | 6.3 ms |
| Gradient boosting | 94.3% | 38.5% | 76.5% | 2.0 MB | 17.8 ms |
| SVM (RBF) | 85.6% | 88.5% | 5.9% | 119 KB | 139 µs |
| k-nearest neighbours (k = 5) | 85.0% | 88.5% | 5.9% | 140 KB | 319 µs |
| Hand-written rules | 60.1% | 73.1% | 35.3% | no model file | 7 s, including a 150-step run |
| Always the commonest label | 11.4% | 65.4% | 0% | none | none |

What it says:

- On generated scripts the shipped model is within 2 points of the best (the random
  forest), at 1/500 of its size and 1/450 of its time per prediction.
- On hand-written scripts the simpler models generalise better (SVM and k-NN 88.5%, the
  no-hidden-layer version 84.6%). The test is small: 21 of 26 has a 95% interval of about
  62% to 92%, so a few points either way is noise.
- The rules miss whole bug types the operators plant (frozen layers 5%, unnormalised
  inputs 7%, shuffled labels 0%: there is no rule for it), but on the hand-written
  benchmarks they name 8 of the 9 bugs to the classifier's 7, at twice the false alarms.
- The times compare the models alone. The classifier also needs one 60-step traced run of
  the script (about 4 s), done in parallel with the runtime monitor.
- LLMs were not compared: no API key was set on the test machine. With a key in `.env`,
  `compare_models.py --llm` asks gpt-oss-120b and gpt-oss-20b the same question, once
  given the trace statistics and once given the source code.

```bash
python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl
pip install scikit-learn matplotlib            # for the baselines and the chart only
python examples/compare_models.py --data traces.jsonl --rules --json comparison.json
python examples/plot_comparison.py comparison.json model_comparison.png
```

## How the pipeline uses it, and why only as advice

While the runtime monitor runs, the script is traced once more in parallel
(`learned/prior.py`). What happens to the guess depends on the mode
(`fix --learned`, `diagnose --learned`, `PipelineConfig(learned=...)`):

| Mode | What the classifier does |
|---|---|
| `advisory` (default) | its top guesses are printed in the report (`Trace classifier: ...`) and kept in the JSON result; nothing else |
| `active` | also: a guess with p ≥ 0.9 of a bug no detector reported becomes a runtime finding, its probabilities nudge the offline ranking of causes by at most 0.3, and the diagnosis LLM is given them as evidence |
| `off` | no traced run |

`active` was measured end to end on the blind benchmarks (offline, 5 seeds) and it made
things worse, so it is not the default:

| | advisory (= no classifier) | active |
|---|---|---|
| Buggy scripts repaired (of 18) | 9 | 8 |
| Right cause ranked first (of 18) | 14 | 13 |
| Healthy scripts changed (of 17) | 2 | 2 |
| Total time, 35 runs | 422 s | 441 s |

What went wrong: on benchmark 15 the classifier was sure (p = 1.0) of
`wrong_loss_reduction`, which pushed the detectors' learning-rate finding down; the
repair for its guess failed Gate 1 and the attempt that used to succeed never ran. On
benchmark 16 a confident wrong finding (`label_data_misalignment`) went to the top. The
classifier is right 81% of the time on these scripts, but the detectors are right more
often on exactly the cases where the two disagree. A learned signal has to beat the rules
it overrides, not just be accurate on average, and this one does not yet.

In `active` mode it can never change a file by itself: every patch still has to pass both
gates, and a patch for a bug that only the classifier reported can pass only on statistics.

What would make it worth turning on: training scripts that look like hand-written code
(the templates are uniform), a frozen-layer operator that leaves the layer out of the
optimizer, and using the probabilities only to break ties between hypotheses whose
confidence is close.

## Limits

- It knows nine bugs and "clean". A bug outside that list gets whichever of them its
  trace looks most like, or "clean".
- It was trained on generated scripts. The benchmark numbers above are the honest test,
  and they are a small sample.
- The trace needs one extra run of 60 steps. On a machine with one or two cores that
  run is not free.
