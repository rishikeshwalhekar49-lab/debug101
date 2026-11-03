# Silent Bug Agent — Root-Cause Diagnosis Agent

An AI-powered, offline, deterministic root-cause diagnosis system for machine learning and deep learning bugs.

---

## Overview

The **Silent Bug Agent** accepts findings from three analysis sources and produces ranked, explainable root-cause hypotheses backed by a structured ML/DL bug taxonomy and a lightweight causal graph.

| Feature | Detail |
|---|---|
| Works offline | No LLM API, no external services |
| Deterministic | Same input always yields the same output |
| Explainable | Each hypothesis includes evidence refs, causal chain, verification steps |
| Fast | < 1 ms per diagnosis on a typical finding set |
| Python | 3.10 + |

---

## Project Structure

```
bug detectn/
├── silent_bug_agent/
│   ├── agents/
│   │   └── diagnosis_agent.py      ← Public DiagnosisAgent class
│   ├── knowledge/
│   │   └── taxonomy.py             ← 15-category taxonomy + causal graph edges
│   ├── inference/
│   │   ├── normalizer.py           ← Heterogeneous findings → NormalizedFinding
│   │   └── causal_engine.py        ← Scoring, ranking, hypothesis construction
│   └── models/
│       ├── findings.py             ← Input Pydantic models
│       └── hypothesis.py          ← Output Pydantic models
├── tests/
│   ├── unit/                       ← Taxonomy, normalizer, engine unit tests
│   └── integration/                ← 20-scenario + accuracy metrics tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

```powershell
# From the project root (bug detectn/)
pip install -r requirements.txt
```

Or install in editable mode:

```powershell
pip install -e ".[dev]"
```

---

## Quick Start

```python
from silent_bug_agent.agents.diagnosis_agent import DiagnosisAgent

agent = DiagnosisAgent()

result = agent.diagnose(
    ast_findings=[
        {
            "id": "AST-001",
            "type": "gradient_management_error",
            "message": "optimizer.zero_grad() may be missing",
            "file": "train.py",
            "line": 42,
            "severity": "high",
        }
    ],
    semantic_findings=[],
    runtime_alerts=[
        {
            "id": "RUN-001",
            "type": "gradient_instability",
            "message": "Gradient norm increased sharply",
            "metadata": {"gradient_norm": 18000},
        }
    ],
)

# Print full JSON
print(result.to_json())

# Access primary hypothesis
top = result.primary_hypothesis
if top:
    print(f"Root cause: {top.root_cause}")
    print(f"Confidence: {top.confidence:.2f}")
    print(f"Explanation: {top.explanation}")
    print("Verification steps:")
    for check in top.verification_checks:
        print(f"  - {check}")

# Access causal chain
print(agent.explain_causal_chain(top))
```

### Example Output

```json
{
  "status": "diagnosed",
  "primary_hypothesis": {
    "rank": 1,
    "root_cause_id": "gradient_management_error",
    "root_cause": "Gradient Management Error",
    "confidence": 0.72,
    "explanation": "...",
    "causal_chain": ["Primary issue: Gradient Management Error", "..."],
    "verification_checks": [
      "Confirm optimizer.zero_grad() is called before loss.backward() in every training step.",
      "..."
    ]
  },
  "hypotheses": [...],
  "evidence_counts": {"ast": 1, "semantic": 0, "runtime": 1},
  "warnings": []
}
```

---

## API Reference

### `DiagnosisAgent`

```python
agent = DiagnosisAgent()
```

#### `agent.diagnose(ast_findings, semantic_findings, runtime_alerts) → DiagnosisResult`

Accepts lists of plain `dict` objects or typed Pydantic model instances:
- `ASTFinding`: `id`, `type`, `message`, `file?`, `line?`, `severity?`, `metadata?`
- `SemanticFinding`: `id`, `type`, `message`, `file?`, `line?`, `severity?`, `context?`, `metadata?`
- `RuntimeAlert`: `id`, `type`, `message`, `step?`, `epoch?`, `severity?`, `metadata?`

All inputs are optional. Empty/`None` categories are handled gracefully.

#### `agent.lookup_taxonomy(category_id) → TaxonomyEntry`

Returns the taxonomy entry for a category ID. Raises `KeyError` for unknown IDs.

#### `agent.list_taxonomy_categories() → List[str]`

Returns all 15 taxonomy category IDs in sorted order.

#### `agent.explain_causal_chain(hypothesis) → str`

Returns a formatted narrative causal chain from a `Hypothesis`.

### `DiagnosisResult`

| Field | Type | Description |
|---|---|---|
| `status` | `DiagnosisStatus` | `diagnosed`, `ambiguous`, `insufficient_evidence` |
| `primary_hypothesis` | `Hypothesis \| None` | Top-ranked hypothesis |
| `hypotheses` | `List[Hypothesis]` | All ranked hypotheses |
| `evidence_counts` | `Dict[str, int]` | Per-source finding counts |
| `evidence_summary` | `str` | Human-readable summary |
| `warnings` | `List[str]` | Evidence quality warnings |

`.to_dict()` — returns JSON-serialisable dict
`.to_json(indent=2)` — returns JSON string

### `Hypothesis`

| Field | Type | Description |
|---|---|---|
| `rank` | `int` | 1 = highest confidence |
| `root_cause_id` | `str` | Stable taxonomy ID |
| `root_cause` | `str` | Human-readable name |
| `confidence` | `float` | Evidence-based score in [0.0, 1.0] |
| `explanation` | `str` | Narrative explanation |
| `supporting_evidence` | `List[EvidenceRef]` | Source findings |
| `contradicting_evidence` | `List[EvidenceRef]` | Counter-evidence |
| `causal_chain` | `List[str]` | Cause → symptom steps |
| `affected_files` | `Dict[str, List[int]]` | File → line numbers |
| `verification_checks` | `List[str]` | Actionable steps |
| `alternative_hypotheses` | `List[str]` | Related category IDs |

> **Note:** `confidence` is an evidence-based scoring metric, not a statistically calibrated probability.

---

## Taxonomy Categories

| Category ID | Name |
|---|---|
| `gradient_management_error` | Gradient Management Error |
| `learning_rate_instability` | Learning Rate Instability |
| `loss_output_mismatch` | Loss / Output Mismatch |
| `tensor_shape_mismatch` | Tensor Shape Mismatch |
| `data_preprocessing_error` | Data Preprocessing Error |
| `label_encoding_error` | Label Encoding Error |
| `model_configuration_error` | Model Configuration Error |
| `optimizer_configuration_error` | Optimizer Configuration Error |
| `training_loop_logic_error` | Training Loop Logic Error |
| `gradient_flow_failure` | Gradient Flow Failure |
| `numerical_instability` | Numerical Instability |
| `evaluation_logic_error` | Evaluation Logic Error |
| `memory_resource_exhaustion` | Memory / Resource Exhaustion |
| `device_placement_error` | Device Placement Error |
| `unknown_or_insufficient_evidence` | Unknown / Insufficient Evidence |

---

## Running Tests

```powershell
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests + accuracy metrics
python -m pytest tests/integration/ -v -s

# With coverage
python -m pytest tests/ --cov=silent_bug_agent --cov-report=term-missing
```

### Test Results (as of implementation)

```
174 passed in 0.55s
```

### Diagnostic Accuracy Metrics (14-case labelled test set)

| Metric | Value |
|---|---|
| Top-1 Accuracy | 92.9% |
| Top-3 Accuracy | 100.0% |
| MRR | 0.9643 |
| Coverage | 92.9% |
| Fallback Rate | 7.1% |

---

## Scoring Model

All weights are documented constants in [`causal_engine.py`](silent_bug_agent/inference/causal_engine.py):

| Parameter | Value | Rationale |
|---|---|---|
| `W_AST` | 0.30 | Structural syntactic evidence |
| `W_SEMANTIC` | 0.40 | Higher-level logical evidence |
| `W_RUNTIME` | 0.50 | Direct observed symptom |
| `W_CAUSAL` | 0.20 | Causal graph match bonus |
| `P_DUPLICATE` | 0.10 | Per extra same-source finding penalty |
| `P_CONTRADICT` | 0.25 | Per contradicting finding |
| `P_SINGLE_SOURCE` | 0.10 | Single-source evidence penalty |
| Severity LOW | ×0.6 | Low severity multiplier |
| Severity MEDIUM | ×1.0 | Default multiplier |
| Severity HIGH | ×1.3 | High severity multiplier |
| Severity CRITICAL | ×1.5 | Critical severity multiplier |

---

## Limitations

1. **Confidence is not a probability**: the score reflects evidence strength relative to the input, not statistical calibration.
2. **Keyword matching**: category detection relies on keyword rules in `normalizer.py`. Novel bug descriptions outside the keyword vocabulary may produce `unknown_or_insufficient_evidence`.
3. **No external tool calls**: the agent does not execute or import user code, read files, or call APIs.
4. **Single-language focus**: the taxonomy and keyword rules are tuned for PyTorch-style ML training loops. TensorFlow/JAX code should still work but may have lower keyword recall.
