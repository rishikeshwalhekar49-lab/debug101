# Architecture Overview

> See also: [DATA.md](DATA.md) · [THREAT.md](THREAT.md) · [FLOW.md](FLOW.md) · [FAILURE.md](FAILURE.md)

## Modules & Responsibilities

| Module | Owns | Does NOT touch |
|---|---|---|
| `cli.py` | CLI entry point, argument parsing, session bootstrap | Agent internals |
| `agents/supervisor.py` | LangGraph StateGraph orchestration, node routing, retry budget | Individual agent logic |
| `runtime/` | PyTorch hooks, live telemetry, profiling, session management | Static analysis |
| `static/` | AST-based rule engine (10 rules), bug detection without execution | Runtime state |
| `semantic/` | LLM client, model router (Groq/Cerebras), semantic review | Deterministic rules |
| `invariants/` | Invariant check library (gradient, activation, loss, weight, mode) | Storage |
| `taxonomy/` | YAML bug taxonomy loader, 21 seed bug patterns | Inference |
| `diagnosis/` | Causal engine, symptom dampening graph, DiagnosisAgent | Patch generation |
| `repair/` | Patch application, AST recipes, FixSynthesizerAgent | Validation |
| `validation/` | Overfit oracle (Gate 1), multi-seed sandbox (Gate 2), ValidationAnalyzerAgent | Patch creation |
| `learning/` | ChromaDB vector store, SQLite trajectory DB, contextual bandit | Agent orchestration |
| `agents/meta_evolution.py` | Trajectory ablation, exemplar pruning, prompt tuning, bandit prior updates | Real-time decisions |
| `wsp/` | Weight-Space Programming: SVD spectral inspection, kurtosis outlier excision, direct weight surgery, dual supervisor | Low-level AST parsing (delegates to static/) |

## Trust Boundaries & Data Entry

- **Untrusted Input Points:**
  - `cli.py` — target script path, checkpoint path, and CLI flags from the user
  - `wsp/inspector.py` — untrusted checkpoint files (`.pt`, `.pth`, `.safetensors`, `.bin`) loaded from user disk
  - `validation/script_runner.py` — executes user-supplied Python scripts in a subprocess
  - `semantic/client.py` — receives LLM-generated JSON (untrusted structured output)
  - `runtime/hooks.py` — tensor data from user model hooks (numeric validation required)

- **Internal Boundaries:**
  - CLI/User → `AgentState` (typed, validated dataclass) → all agents
  - LLM responses → JSON schema validation (strict parse in `agents/prompts.py`) → `DiagnosticHypothesis` / `PatchCandidate`
  - Subprocess stdout → `__METRICS__:{...}` protocol parse → `ValidationVerdict`

## System Map

```mermaid
flowchart TD
    CLI["CLI (scan / monitor / diagnose / fix / doctor / wsp)"]
    SUP["Supervisor / DualSupervisor\n(LangGraph & WSP Engine)"]
    CODE_DET["Code-Space Perception\nAST Static + Runtime Hooks + LLM Semantic"]
    WSP_DET["Weight-Space Perception\nSVD Spectral + Kurtosis Outliers + Bias Profiling"]
    DIAG["Dual Causal Diagnosis\nCode & Parameter Fault Attribution"]
    SYNTH["Solution Synthesis\nCode Diff + Weight Surgery Delta"]
    HUMAN["Human-in-the-Loop Gate\nInteractive Proposal Approval"]
    VAL["Validation Layer\nGate1: Overfit Oracle\nGate2: Multi-Seed Sandbox"]
    MEM["Experience Store\nChromaDB + SQLite"]

    CLI --> SUP
    SUP --> CODE_DET
    SUP --> WSP_DET
    CODE_DET --> DIAG
    WSP_DET --> DIAG
    DIAG --> SYNTH
    SYNTH --> HUMAN
    HUMAN -->|Approved| VAL
    HUMAN -->|Rejected| SUP
    VAL -->|Accepted| MEM
    VAL -->|Failed (<= 3 iter)| SYNTH
```

## Key Design Principles

1. **Hybrid Nodes** — every agent pairs deterministic tools with an LLM for perception/reasoning
2. **Single State Contract** — `AgentState` dataclass flows through every node; no raw dicts between agents
3. **Empirical Gate** — no patch is committed without passing at least Gate 1 (overfit oracle)
4. **Test Isolation** — `SBA_NO_DOTENV=1` + fake LLM server; zero real API calls in 404 tests
