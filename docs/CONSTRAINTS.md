# Project Constraints & Rules

## Hard Rules (NEVER)

- NEVER commit secrets, API keys, or any values from `.env` to version control.
- NEVER modify `silent_bug_agent/taxonomy/bug_taxonomy.yaml` without updating the taxonomy graph tests — it is the seed of the entire causal inference engine.
- NEVER change `silent_bug_agent/agents/prompts.py` JSON schemas without updating ALL agent parsers that consume them — schema drift breaks inter-agent contracts silently.
- NEVER call a real LLM API in automated tests — always use `SBA_NO_DOTENV=1` and the fake provider.
- NEVER bypass the `ValidationVerdict` acceptance gate — patches must pass at least Gate 1 (overfit oracle) before being written to disk.
- NEVER add a new NPM or Pip package without explicit team approval and updating `pyproject.toml` + `requirements.txt` together.
- NEVER write directly to `~/.silent_bug_agent/` from tests — use a temp directory and `SBA_HOME` override.

## Mandatory Guardrails (ALWAYS)

- ALWAYS validate untrusted input (target script paths, CLI args) before passing to subprocess execution in `validation/script_runner.py`.
- ALWAYS use the `AgentState` dataclass as the single state contract — never pass raw dicts between agents.
- ALWAYS use the repository pattern: `learning/experience_store.py` is the ONLY writer to ChromaDB and SQLite.
- ALWAYS create atomic, single-purpose commits per AI task (one bug fix = one commit).
- ALWAYS run `python -m pytest tests/ -v` before marking any phase complete.
- ALWAYS set `SBA_NO_DOTENV=1` in CI/CD environments to prevent test pollution from local `.env` files.
- ALWAYS maintain backward-compatible `to_dict()` / `from_dict()` on `schemas.py` dataclasses — the experience store persists these as JSON.

## Protected Paths (Requires Explicit Approval Before Touching)

- `silent_bug_agent/agents/supervisor.py` — central state machine; edge changes affect the whole pipeline
- `silent_bug_agent/taxonomy/bug_taxonomy.yaml` — 21 seed bug patterns
- `silent_bug_agent/agents/prompts.py` — LLM prompt templates and JSON schemas
- `silent_bug_agent/schemas.py` — inter-agent data contracts
- `tests/` — test suite integrity must be preserved

## Dependency Limits

- Python runtime: ONLY packages listed in `pyproject.toml`
- External LLM providers: Groq and Cerebras ONLY (the semantic router knows these two protocols)
- Vector DB: ChromaDB ONLY — do not add pgvector, Pinecone, or Weaviate without a formal D-XXX decision
- PyTorch hooks: ONLY use `register_forward_hook` / `register_backward_hook` — never monkey-patch `nn.Module` internals
