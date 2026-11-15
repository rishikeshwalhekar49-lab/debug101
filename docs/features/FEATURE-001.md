# FEATURE-001: Multi-Agent Silent Bug Detection & Repair Pipeline

## Goal & Business Context
Automate the full detect → diagnose → fix → validate workflow for silent bugs in deep learning training scripts. Target users are ML researchers and engineers who waste hours debugging non-crashing training failures.

## In-Scope
- 21 bug taxonomy entries in `bug_taxonomy.yaml`
- 10 AST static rules in `static/`
- PyTorch runtime hooks (Module 1)
- LLM semantic review (Groq/Cerebras)
- Causal diagnosis engine
- Fix synthesizer with MAB bandit + ReAct LLM
- Gate 1 (overfit oracle) + Gate 2 (multi-seed) validation
- Experience memory (ChromaDB + SQLite)
- Meta-evolution agent
- CLI: `scan`, `monitor`, `diagnose`, `fix`, `apply`, `doctor`

## Out-of-Scope
- Support for JAX or TensorFlow (PyTorch only)
- Multi-GPU / DDP bug patterns beyond `FRAMEWORK_CONFIG` strategy
- Web UI
- Cloud deployment / API server mode

## Implementation Plan
1. Module 0: Taxonomy YAML + Invariant library ✅
2. Module 1: Runtime hooks + RuntimeMonitorAgent ✅
3. Module 2: AST static analyzer + LLM semantic reviewer + CLI ✅
4. Module 3: Overfit oracle + multi-seed sandbox + analyzer ✅
5. Module 4: Causal engine + DiagnosisAgent ✅
6. Module 5: Fix synthesizer + repair recipes + patch utils ✅
7. Module 6: Experience store (ChromaDB + SQLite) + bandit ✅
8. Module 7: MetaEvolutionAgent + prompt tuning ✅
9. LangGraph Supervisor orchestrating all modules ✅
10. 404-test suite with fake LLM server ✅

## Open Questions & Assumptions
- Assumed: Users run `sba fix` on a single training script file, not a full project directory.
- Open: What is the target repair rate on real (non-benchmark) training scripts?
- Decided 2026-11-15 (D-009): there is no `--apply` flag. `fix` writes a validated patch; `fix --dry-run` followed by `apply` is the review-first path, and the terminal UI always uses it.

### Timeline Log
- 2026-11-04 12:00 - Feature fully implemented; 404 tests passing; documentation suite initialized
