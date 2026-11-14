# Data Models & Storage Architecture

## Entity / Schema: `Finding`
- Storage Medium: In-memory (session), SQLite trajectory (persisted via `ExperienceStore`)
- Expected Volume: ~5–50 findings per analysis run
- Access Pattern: Write-once per run; read by DiagnosisAgent and FixSynthesizerAgent

### Shape / Structure
```json
{
  "bug_id": "string (e.g., BUG_MISSING_ZERO_GRAD)",
  "severity": "enum(critical, high, medium, low, info)",
  "file": "string (absolute path)",
  "line": "integer | null",
  "message": "string",
  "evidence": "string",
  "source": "enum(static, runtime, semantic)"
}
```

### Access Control & Permissions
- Write Access: `static/analyzer.py`, `runtime/session.py`, `agents/runtime_monitor_agent.py`, `semantic/analyzer.py`
- Read Access: `agents/diagnosis_agent.py`, `agents/fix_synthesizer_agent.py`, `learning/experience_store.py`

---

## Entity / Schema: `DiagnosticHypothesis`
- Storage Medium: `AgentState.hypotheses` (in-memory); serialized to SQLite as JSON in trajectory
- Expected Volume: 1–10 hypotheses per run, sorted by confidence descending
- Access Pattern: Write-once (DiagnosisAgent); read by FixSynthesizerAgent

### Shape / Structure
```json
{
  "bug_id": "string",
  "description": "string",
  "confidence": "float (0.0–1.0)",
  "evidence_finding_keys": "list of [bug_id, file, line] tuples",
  "suggested_strategy": "enum(direct_edit, refactor, hyperparam, framework_config, unknown)",
  "extra": "object"
}
```

### Access Control & Permissions
- Write Access: `agents/diagnosis_agent.py`
- Read Access: `agents/fix_synthesizer_agent.py`, `agents/supervisor.py`

---

## Entity / Schema: `PatchCandidate`
- Storage Medium: `AgentState.patch_candidates` (in-memory); diff string stored in SQLite trajectory
- Expected Volume: 1–5 candidates per repair iteration
- Access Pattern: Write (FixSynthesizerAgent); read (ValidationAgents, ExperienceStore)

### Shape / Structure
```json
{
  "hypothesis_bug_id": "string",
  "diff": "string (unified diff, patch -p1 compatible)",
  "strategy": "enum(direct_edit, refactor, hyperparam, framework_config, unknown)",
  "source": "enum(rules, llm, memory, hybrid)",
  "confidence": "float (0.0–1.0)",
  "changed_files": ["string"],
  "changed_lines": "integer",
  "extra": "object"
}
```

### Access Control & Permissions
- Write Access: `agents/fix_synthesizer_agent.py`, `repair/recipes.py`
- Read Access: `validation/overfit_agent.py`, `validation/multiseed_agent.py`, `validation/analyzer.py`

---

## Entity / Schema: `ValidationVerdict`
- Storage Medium: `AgentState.validation_verdicts` (in-memory); serialized to SQLite trajectory
- Expected Volume: 1–3 verdicts per patch candidate (overfit oracle + multiseed + analyzer)
- Access Pattern: Write (ValidationAgents); read (Supervisor, ExperienceStore)

### Shape / Structure
```json
{
  "patch_candidate": "PatchCandidate object",
  "accepted": "boolean",
  "validator": "enum(overfit_oracle, multiseed_stats, validation_analyzer, temporary_patch_and_syntax_check)",
  "p_value": "float | null",
  "effect_size": "float | null",
  "seeds_run": "integer",
  "compute_budget_used": "float (0.0–1.0)",
  "notes": "string",
  "extra": "object"
}
```

### Access Control & Permissions
- Write Access: `validation/overfit_agent.py`, `validation/multiseed_agent.py`, `validation/analyzer.py`
- Read Access: `agents/supervisor.py`, `learning/experience_store.py`

---

## Entity / Schema: `ExperienceTrajectory` (SQLite + ChromaDB)
- Storage Medium: SQLite (`~/.silent_bug_agent/trajectories.db`) + ChromaDB (`~/.silent_bug_agent/chroma/`)
- Expected Volume: ~100–10,000 trajectories over lifetime; ChromaDB embeddings ~1536 dims
- Access Pattern: Write on accepted patch; read for few-shot retrieval (95% reads / 5% writes)

### Shape / Structure
```json
{
  "id": "uuid",
  "bug_id": "string",
  "target_file_hash": "string (sha256 of target script)",
  "hypothesis": "DiagnosticHypothesis JSON",
  "patch": "PatchCandidate JSON",
  "verdict": "ValidationVerdict JSON",
  "reward": "float (-1.0 to +1.0)",
  "created_at": "ISO 8601 timestamp"
}
```

### Access Control & Permissions
- Write Access: `learning/experience_store.py` ONLY
- Read Access: `learning/experience_store.py` (retrieval), `agents/meta_evolution.py` (ablation)
- Architectural Trade-off: Dual-store chosen over pgvector to avoid server infrastructure; ChromaDB handles semantic similarity; SQLite handles structured queries and bandit reward lookups.
