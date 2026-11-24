"""The trace classifier: features, the numpy model, the shipped weights and how the pipeline uses them."""

import numpy as np
import pytest

from silent_bug_agent.learned.features import FEATURES, vectorize
from silent_bug_agent.learned.model import WEIGHTS, TraceClassifier, load_default
from silent_bug_agent.learned.prior import FINDING_THRESHOLD, apply_prior, finding_from
from silent_bug_agent.mutation.operators import OPERATORS
from silent_bug_agent.schemas import DiagnosticHypothesis


def test_features_are_finite_and_named():
    report = {"losses": [2.0, 1.5, 1.0, float("nan")],
              "trace": {"grad_norms": [1.0] * 5 + [4.0] * 5, "no_grad_fraction": [0.0, 0.0, 0.5], "update_ratios": [0.01],
                        "zero_grad_calls": 0, "lr": 0.1, "frozen_fraction": 0.25, "forwards": 10,
                        "no_grad_forwards": 2, "no_grad_training_forwards": 2, "measured_outputs": 4,
                        "unit_outputs": 4, "rowsum_one_outputs": 4, "ce_calls": 3, "ce_probability_inputs": 3,
                        "input_abs_mean": 100.0, "input_std": 50.0, "relu_zero_fraction": 0.9, "bias_mean": -8.0,
                        "mode_layers": True, "outside_optimizer_fraction": 0.25}}
    values = dict(zip(FEATURES, vectorize(report)))
    assert len(values) == len(FEATURES) and all(np.isfinite(list(values.values())))
    assert values["loss_nonfinite"] == 1.0 and values["zero_grad_per_step"] == 0.0
    assert values["rowsum_one_outputs"] == 1.0 and values["ce_on_probabilities"] == 1.0
    assert values["eval_missing_where_it_matters"] == 1.0 and values["log_input_scale"] == pytest.approx(2.0)
    assert values["outside_optimizer"] == 0.25 and values["has_mode_layers"] == 1.0
    # without dropout or batch norm, validating in training mode changes nothing, so it is not a signal
    plain = dict(zip(FEATURES, vectorize(dict(report, trace=dict(report["trace"], mode_layers=False)))))
    assert plain["eval_missing_where_it_matters"] == 0.0
    assert values["bias_mean"] < -0.99 and values["grad_growth"] > 0
    assert vectorize({}) == [0.0] * len(FEATURES)  # nothing measured: all zeros, no crash
    diverged = dict(report, trace=dict(report["trace"], bias_mean=float("nan")))
    assert all(np.isfinite(vectorize(diverged)))  # NaN weights after a divergence must not poison the model


@pytest.mark.parametrize("hidden", [0, 8])
def test_the_model_learns_and_round_trips(tmp_path, hidden):
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 3, (3, len(FEATURES)))
    X = np.vstack([c + rng.normal(0, 0.3, (40, len(FEATURES))) for c in centres])
    y = [label for label in ("a", "b", "clean") for _ in range(40)]
    model = TraceClassifier.fit(X, y, hidden=hidden, epochs=400)
    predicted = [model.classes[i] for i in model.probabilities(X).argmax(axis=1)]
    assert np.mean([p == t for p, t in zip(predicted, y)]) > 0.97
    path = model.save(tmp_path / "w.json")
    again = TraceClassifier.load(path)
    assert np.allclose(again.probabilities(X[:5]), model.probabilities(X[:5]), atol=1e-4)


def test_the_shipped_weights_are_small_and_cover_every_planted_bug():
    model = load_default()
    assert model is not None, "silent_bug_agent/learned/weights.json is missing or stale: retrain it"
    assert model.features == FEATURES and set(model.classes) == set(OPERATORS) | {"clean"}
    assert WEIGHTS.stat().st_size < 64 * 1024
    assert model.meta["benchmarks"]["n"] > 0 and model.meta["cross_validated"]["accuracy"] > 0.5


def _hypothesis(bug, confidence):
    return DiagnosticHypothesis(bug_id=bug, description=bug, confidence=confidence)


def test_the_prior_nudges_the_ranking_and_only_confident_guesses_become_findings():
    prediction = {"top": "learning_rate_misconfigured", "p": 0.9, "seconds": 1.0,
                  "probabilities": {"learning_rate_misconfigured": 0.9, "wrong_loss_reduction": 0.05}}
    ranked = apply_prior([_hypothesis("wrong_loss_reduction", 0.7), _hypothesis("learning_rate_misconfigured", 0.6)],
                         prediction)
    assert [h.bug_id for h in ranked] == ["learning_rate_misconfigured", "wrong_loss_reduction"]
    assert ranked[0].extra["learned_probability"] == 0.9
    # a much stronger detector result is not overturned: the nudge is at most 0.3
    kept = apply_prior([_hypothesis("wrong_loss_reduction", 0.95), _hypothesis("learning_rate_misconfigured", 0.4)],
                       prediction)
    assert kept[0].bug_id == "wrong_loss_reduction"

    finding = finding_from(prediction, "train.py", known=[])
    assert finding.bug_id == "learning_rate_misconfigured" and finding.rule == "trace_classifier"
    assert finding_from(prediction, "train.py", known=["learning_rate_misconfigured"]) is None
    assert finding_from(dict(prediction, p=FINDING_THRESHOLD - 0.01), "train.py", known=[]) is None
    assert finding_from(dict(prediction, top="clean"), "train.py", known=[]) is None


def test_a_traced_run_of_a_real_mutant_is_classified():
    pytest.importorskip("torch")
    from silent_bug_agent.mutation.dataset import trace_run
    from silent_bug_agent.mutation.operators import mutants
    from silent_bug_agent.mutation.templates import Spec, render

    code = mutants(render(Spec(optimizer="sgd", lr=0.1, style="sequential")))["missing_zero_grad"]
    report = trace_run(code, max_steps=40, timeout=120)
    assert report["status"] == "stopped" and report["trace"]["zero_grad_calls"] == 0  # stopped by the step budget
    assert len(report["trace"]["grad_norms"]) == 40
    model = load_default()
    assert model is not None and next(iter(model.predict_report(report))) == "missing_zero_grad"


def test_an_advisory_classifier_is_shown_but_takes_no_part():
    from silent_bug_agent.agents.diagnosis_agent import DiagnosisAgent
    from silent_bug_agent.findings import Report
    from silent_bug_agent.schemas import AgentState

    prediction = {"top": "dead_relu", "p": 0.99, "probabilities": {"dead_relu": 0.99}, "seconds": 1.0}
    agent = DiagnosisAgent()
    agent.run(AgentState(detection_report=Report(), extra={"learned": dict(prediction, mode="advisory")}))
    assert agent.learned is None
    agent.run(AgentState(detection_report=Report(), extra={"learned": dict(prediction, mode="active")}))
    assert agent.learned["top"] == "dead_relu"


def test_loss_call_features_separate_the_cases_that_look_alike():
    import math

    def report(train_values, val_values, reduction="mean", classes=5, batch=32):
        calls = [{"fn": "cross_entropy", "train": True, "reduction": reduction, "value": v, "batch": batch,
                  "classes": classes} for v in train_values]
        calls += [{"fn": "cross_entropy", "train": False, "reduction": reduction, "value": v, "batch": 200,
                   "classes": classes} for v in val_values]
        return dict(zip(FEATURES, vectorize({"losses": train_values, "trace": {"loss_calls": calls}})))

    chance = math.log(5)
    shuffled = report([chance] * 20, [chance] * 4)
    healthy = report([chance * 0.97 ** i for i in range(20)], [chance * 0.9 ** i for i in range(4)])
    summed = report([chance * 32 * 0.97 ** i for i in range(20)], [chance * 200] * 4, reduction="sum")
    assert abs(shuffled["end_vs_chance"]) < 0.01 and abs(shuffled["val_slope"]) < 0.01
    assert healthy["end_vs_chance"] < -0.2 and healthy["val_slope"] < -0.1
    assert summed["sum_reduction"] == 1.0 and summed["start_vs_chance"] > 1.4
    assert shuffled["has_chance"] == 1.0 and shuffled["has_validation_loss"] == 1.0
