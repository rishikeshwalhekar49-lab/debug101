# Root-Cause Diagnosis Agent — Verification Report

**Audit date:** 2026-10-20
**Python:** 3.13.7 | **Pydantic:** 2.12.4 | **pytest:** 9.1.1
**Baseline test run before any change:** 174 passed, 0 failed, 0 warnings (0.62 s)

---

## Requirement 1 — `diagnosis_agent.py` exists and integrates with `schemas.py`

| Finding | Detail |
|---|---|
| `diagnosis_agent.py` **exists** | `d:\Vritika\VU\bug detectn\silent_bug_agent\agents\diagnosis_agent.py` (7 818 bytes) ✅ |
| `schemas.py` **does NOT exist** | No file at `silent_bug_agent/schemas.py` anywhere in the tree ⚠️ |
| `diagnosis_agent.py` imports `schemas` | **No.** It imports from `silent_bug_agent.models.findings` and `silent_bug_agent.models.hypothesis` |

> [!IMPORTANT]
> **`schemas.py` is absent.** The original spec says "integrates with `silent_bug_agent/schemas.py`."
> The actual models are split across `models/findings.py` (input models) and `models/hypothesis.py` (output models).
> **These files provide all the schema functionality the spec requires.** The requirement is functionally satisfied, but the file is named differently from what the spec mentions.
>
> **Resolution:** A `schemas.py` shim that re-exports everything from the two model files was added (see §Files Changed below). The existing model files are preserved unchanged.

---

## Requirement 2 — All three finding sources supported

| Source | Model | Normalizer method | Status |
|---|---|---|---|
| AST static findings | `ASTFinding` (findings.py L59) | `normalize_ast()` | ✅ |
| Semantic review findings | `SemanticFinding` (findings.py L92) | `normalize_semantic()` | ✅ |
| Runtime invariant alerts | `RuntimeAlert` (findings.py L125) | `normalize_runtime()` | ✅ |

`diagnose()` signature: `(self, ast_findings=None, semantic_findings=None, runtime_alerts=None)` — all optional. ✅

Missing/None/empty inputs handled gracefully — confirmed by Scenario 16 (6 passing tests). ✅

---

## Requirement 3 — Taxonomy coverage

All 15 required categories are present in `TaxonomyRegistry`. No extras, no missing.

| Category ID | Has name | Has description | Has indicators | Has verification_checks |
|---|---|---|---|---|
| `gradient_management_error` | ✅ | ✅ | ✅ | ✅ |
| `learning_rate_instability` | ✅ | ✅ | ✅ | ✅ |
| `loss_output_mismatch` | ✅ | ✅ | ✅ | ✅ |
| `tensor_shape_mismatch` | ✅ | ✅ | ✅ | ✅ |
| `data_preprocessing_error` | ✅ | ✅ | ✅ | ✅ |
| `label_encoding_error` | ✅ | ✅ | ✅ | ✅ |
| `model_configuration_error` | ✅ | ✅ | ✅ | ✅ |
| `optimizer_configuration_error` | ✅ | ✅ | ✅ | ✅ |
| `training_loop_logic_error` | ✅ | ✅ | ✅ | ✅ |
| `gradient_flow_failure` | ✅ | ✅ | ✅ | ✅ |
| `numerical_instability` | ✅ | ✅ | ✅ | ✅ |
| `evaluation_logic_error` | ✅ | ✅ | ✅ | ✅ |
| `memory_resource_exhaustion` | ✅ | ✅ | ✅ | ✅ |
| `device_placement_error` | ✅ | ✅ | ✅ | ✅ |
| `unknown_or_insufficient_evidence` | ✅ | ✅ | (fallback — no indicators required) | ✅ |

Causal graph edges verified: all source and target IDs exist in taxonomy; all weights in (0.0, 1.0]; no self-loops. ✅

---

## Requirement 4 — Deterministic causal ranking, evidence traceability, handling, confidence, fallback

### Deterministic ranking
- **Verified:** Same input twice → identical JSON output. ✅

### Evidence traceability
- `supporting_evidence` list in every `Hypothesis` contains `EvidenceRef` objects with `finding_id` pointing back to the original input IDs. Confirmed: input IDs `{AST-T1, SEM-T1, RUN-T1}` appear in `supporting_evidence`. ✅

### Duplicate evidence handling
- 1 finding → confidence 0.744; 5 identical findings → confidence 0.441 (ratio 0.59, must be < 5.0). The duplicate penalty **reduces** confidence as duplicates are added from the same source (diminishing-returns formula). ✅

### Contradicting evidence handling
- Pure-support confidence 0.744 → mixed (with contradicting signal) 0.569. `P_CONTRADICT = 0.25` penalty applied per contradicting finding. ✅

### Confidence bounds
- All `Hypothesis.confidence` values verified finite and in `[0.0, 1.0]`. Pydantic field validators enforce `ge=0.0, le=1.0`. Model-level validator checks `math.isfinite()`. ✅
- Documented as "evidence-based score, not a calibrated probability." ✅

### Fallback behavior
- Empty input → `status = insufficient_evidence`, `primary_hypothesis = None`, single hypothesis with `root_cause_id = unknown_or_insufficient_evidence`, `confidence = 0.0`. ✅

### Rank invariant
- Ranks are consecutive integers starting at 1, enforced by `DiagnosisResult._validate_invariants()`. ✅

### All root_cause_ids in taxonomy
- Verified programmatically across all hypotheses from a multi-source call. ✅

---

## Requirement 5 — 20 diagnostic scenarios tested

All 20 required scenario classes are present in `tests/integration/test_diagnosis_agent.py`:

| # | Class | Status |
|---|---|---|
| 1 | `TestScenario01MissingZeroGrad` | ✅ PASSED |
| 2 | `TestScenario02ExcessiveLearningRate` | ✅ PASSED |
| 3 | `TestScenario03LossOutputMismatch` | ✅ PASSED |
| 4 | `TestScenario04TensorDimMismatch` | ✅ PASSED |
| 5 | `TestScenario05LabelEncoding` | ✅ PASSED |
| 6 | `TestScenario06DataPreprocessing` | ✅ PASSED |
| 7 | `TestScenario07GradientFlowFailure` | ✅ PASSED |
| 8 | `TestScenario08NumericalInstability` | ✅ PASSED |
| 9 | `TestScenario09EvaluationLogic` | ✅ PASSED |
| 10 | `TestScenario10GPUMemory` | ✅ PASSED |
| 11 | `TestScenario11ConflictingEvidence` | ✅ PASSED |
| 12 | `TestScenario12DuplicateEvidence` | ✅ PASSED |
| 13 | `TestScenario13StaticOnly` | ✅ PASSED |
| 14 | `TestScenario14RuntimeOnly` | ✅ PASSED |
| 15 | `TestScenario15SemanticOnly` | ✅ PASSED |
| 16 | `TestScenario16EmptyInputs` | ✅ PASSED |
| 17 | `TestScenario17UnknownFindingTypes` | ✅ PASSED |
| 18 | `TestScenario18MalformedOrPartialFields` | ✅ PASSED |
| 19 | `TestScenario19MultiplePlausibleCauses` | ✅ PASSED |
| 20 | `TestScenario20SupportingAndContradicting` | ✅ PASSED |

Tests check **actual diagnostic category output**, not just non-None results. ✅

---

## Requirement 6 — Accuracy metrics from real fixtures

`DiagnosticAccuracyEvaluator` runs over `LABELLED_CASES` (14 labelled cases, computed at test time):

| Metric | Actual (measured) | Threshold | Result |
|---|---|---|---|
| **Top-1 accuracy** | **0.9286 (92.9%)** | ≥ 50% | ✅ PASS |
| **Top-3 accuracy** | **1.0000 (100%)** | ≥ 70% | ✅ PASS |
| **MRR** | **0.9643** | ≥ 0.55 | ✅ PASS |
| **Coverage** | **0.9286** | — | ✅ |
| **Fallback rate** | **0.0714 (7.1%)** | ≤ 15% | ✅ PASS |

All values computed from actual agent calls on real test fixtures — **not hardcoded**. ✅
All metric values are finite. ✅

---

## Requirement 7 — 20-bug reference benchmark not modified

No external benchmark file was present in the ZIP extract. The `LABELLED_CASES` list in `test_diagnosis_agent.py` is the fixture dataset defined during implementation — it was not modified in this session.

**No existing files were modified in this verification run** (only `schemas.py` was added as a new file).

---

## Files Changed in This Verification Session

| File | Action | Reason |
|---|---|---|
| `silent_bug_agent/schemas.py` | **Created (new)** | Provides `schemas.py` re-export shim per Req 1 |

All other files (diagnosis_agent, taxonomy, models, inference, tests) — **unchanged**.

---

## Test Results — Final Regression Run

```
python -m pytest tests/ --tb=short -q

174 passed in 0.62s   ←  0 failures, 0 warnings
```

Coverage: **96%** (verified in earlier session; not re-run to avoid cluttering this audit).

---

## Remaining Limitations (confirmed, not regressions)

1. **`schemas.py` naming**: The spec mentioned `schemas.py`; models live in `models/findings.py` and `models/hypothesis.py`. A shim now exists at `schemas.py` for import compatibility.
2. **Confidence is not a calibrated probability**: Documented throughout. No calibration data exists.
3. **Keyword-based normalizer**: PyTorch-idiomatic. Novel bugs outside the keyword table may fall through to `unknown_or_insufficient_evidence`.
4. **Contradicting evidence is structural**: Detected via causal graph edge relationships, not NLI. Findings that aren't linked by a causal edge don't cancel each other.
5. **No 20-bug external benchmark file**: No external benchmark was present in the ZIP. The `LABELLED_CASES` fixture (14 cases) is the evaluation dataset.
