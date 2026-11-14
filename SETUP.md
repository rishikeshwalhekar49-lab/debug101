# Setup and Installation Guide

This guide covers everything required to install, configure, verify, and run the **Deep Learning Bug Detector (`silent-bug-agent`)**, including the Python multi-agent backend, the Weight-Space Programming (WSP) engine, and the Pastel + Ink terminal UI (TUI).

---

## 1. System Requirements

| Component | Minimum Version | Recommended | Notes |
|---|---|---|---|
| **Python** | 3.10+ | 3.11 or 3.12 | Core multi-agent pipeline & PyTorch |
| **Node.js** | 18.0.0+ | 20.x or 22.x | Required for the interactive Pastel + Ink TUI |
| **npm** | 9.x+ | 10.x+ | Package manager for `tui/` |
| **Operating System** | Windows 10/11, macOS, or Linux | Any | Fully CPU-compatible (no dedicated GPU required) |

---

## 2. Step-by-Step Installation

### Step 2.1: Clone the Repository

```bash
git clone https://github.com/shreyasshah707/AgenticDLDebugger.git
cd AgenticDLDebugger
```

*(Or navigate to your local working directory if already cloned).*

---

### Step 2.2: Setup Python Virtual Environment

Create and activate an isolated virtual environment:

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If script execution is restricted on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2.3: Install Python Dependencies

Install the core package in editable mode along with PyTorch, ChromaDB, and developer test dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[torch,semantic,dev]"
```

> [!NOTE]
> Everything runs on standard CPUs. If you want CUDA acceleration on an Nvidia GPU, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org) (e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

---

### Step 2.4: Build the Pastel + Ink Terminal UI (TUI)

The rich interactive dashboard is built using Node.js, React, Ink, and Pastel. It lives in the `tui/` directory:

```bash
cd tui
npm install
npm run build
cd ..
```

This compiles TypeScript source files in `tui/commands/` and `tui/components/` into `tui/dist/cli.js`.

---

## 3. Configuration & API Keys

The system is designed with **Zero-Key Fallbacks**: it functions completely offline using deterministic AST recipes, PyTorch runtime telemetry, and empirical statistical tests.

Adding free LLM API keys activates the cognitive reasoning layers (semantic code intent review, causal chain-of-thought diagnosis, ReAct patch generation):

### Step 3.1: Create `.env`

Copy the template:
```bash
# On Linux / macOS:
cp .env.example .env

# On Windows:
copy .env.example .env
```

### Step 3.2: Configure Providers

Open `.env` in your editor:

```dotenv
# --- Primary Provider: Groq (Free Tier) ---
# Sign up at: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# --- Backup Provider: Cerebras (Free Tier) ---
# Sign up at: https://cloud.cerebras.ai
CEREBRAS_API_KEY=your_cerebras_api_key_here

# Optional: Custom storage directory for SQLite trajectories and ChromaDB vector memory
# Defaults to ~/.silent_bug_agent or <repo>/.silent_bug_agent
SBA_HOME=
```

> [!TIP]
> **Provider Priority**: The router tries **Groq** $\rightarrow$ **Cerebras** $\rightarrow$ **Offline Rules**. When Groq's free rate limit is hit, requests seamlessly fail over to Cerebras without interrupting your debugging session.

---

## 4. Health Check & Diagnostics

Run the built-in diagnostic tool to verify dependencies, model configurations, and environment variables:

```bash
# Structural check (libraries, .env file, configured models)
python -m silent_bug_agent doctor
```

If you configured real API keys in `.env`, verify network connectivity with a live probe:

```bash
python -m silent_bug_agent doctor --live
```

---

## 5. Running the Application

### 5.1 Interactive Terminal UI (TUI)

Launch the full keyboard-driven Pastel + Ink dashboard:

```bash
python -m silent_bug_agent ui
```

* **Navigate**: Use `↑` / `↓` arrow keys
* **Select / Confirm**: Press `Enter`
* **Go Back**: Press `Esc` or `b`
* **Quit**: Press `q`

---

### 5.2 Command Line Interface (CLI)

The unified CLI provides dedicated subcommands:

#### 1. Static & Semantic Code Scan (`scan`)
Fast AST pattern audit and optional semantic intent review:
```bash
# Static AST scan
python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py

# Static + LLM Semantic Review
python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py --semantic

# Output structured JSON for CI/CD pipelines
python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py --json
```

#### 2. Live Runtime Monitoring (`monitor`)
Executes the script with PyTorch telemetry hooks to catch vanishing/exploding gradients, dead activations, and un-updated parameters:
```bash
python -m silent_bug_agent monitor examples/sample_scripts/benchmark_01_missing_zero_grad.py
```

#### 3. Root-Cause Diagnosis (`diagnose`)
Correlates findings through the causal dampening graph and ranks hypotheses:
```bash
python -m silent_bug_agent diagnose examples/sample_scripts/buggy_mnist.py
```

#### 4. Autonomous Repair Pipeline (`fix`)
Runs the full supervisor loop: Perception $\rightarrow$ Diagnosis $\rightarrow$ Patch Synthesis $\rightarrow$ Gate 1 (Overfit Oracle) $\rightarrow$ Gate 2 (Multi-Seed Sandbox) $\rightarrow$ Apply:
```bash
# Preview patch diff without writing to disk
python -m silent_bug_agent fix examples/sample_scripts/benchmark_06_dead_relu.py --dry-run

# Run full repair loop (applies verified patch to disk)
python -m silent_bug_agent fix examples/sample_scripts/benchmark_06_dead_relu.py
```

#### 5. Weight-Space Programming (`wsp`)
Simultaneously audits source code AST and raw weight matrices (SVD spectral condition number, kurtosis outliers, dead bias collapse) with an interactive human approval gate:
```bash
# Interactive mode (prompts before applying and entering validation gates)
python -m silent_bug_agent wsp examples/sample_scripts/benchmark_06_dead_relu.py

# Non-interactive / CI mode
python -m silent_bug_agent wsp examples/sample_scripts/benchmark_06_dead_relu.py --dry-run --yes
```

---

## 6. Running the Test Suite

The automated test suite runs against a local fake LLM server that simulates provider protocols—ensuring tests are deterministic, free, and require no live internet connection.

Run all tests:
```bash
# Windows (PowerShell):
$env:SBA_NO_DOTENV="1"; python -m pytest tests/ -v

# Linux / macOS:
SBA_NO_DOTENV=1 python -m pytest tests/ -v
```

Targeted test suites:
```bash
# Weight-Space Programming (WSP) unit tests
python -m pytest tests/test_wsp.py -v

# AST static analyzer & rules
python -m pytest tests/test_static.py -v

# CLI commands & argument handling
python -m pytest tests/test_cli.py -v

# Multi-seed statistical sandbox
python -m pytest tests/test_multiseed_agent.py -v
```

---

## 7. Troubleshooting & FAQ

### Q: `npm run build` fails with TypeScript errors
**Fix**: Ensure your Node.js version is $\ge 18$. Run `node -v` to check. In `tui/`, delete `node_modules` and re-run:
```bash
cd tui
rm -rf node_modules package-lock.json  # Windows: Remove-Item -Recurse -Force node_modules, package-lock.json
npm install
npm run build
cd ..
```

### Q: `sqlite3.OperationalError: unable to open database file` on Windows
**Fix**: In restricted sandbox environments, user directory permissions on `~/.silent_bug_agent/` may be locked. The system automatically falls back to an in-memory database and creates `<project_root>/.silent_bug_agent/`. You can also set `SBA_HOME` explicitly in `.env`:
```dotenv
SBA_HOME=./.silent_bug_agent
```

### Q: Does the multi-seed validation gate take too long?
**Fix**: Gate 2 runs multiple isolated subprocesses across distinct random seeds to compute statistical tests ($p < 0.05$). For rapid local prototyping, decrease seed counts:
```bash
python -m silent_bug_agent fix examples/sample_scripts/buggy_mnist.py --seeds 3 --no-llm
```

### Q: How do I pass script-specific arguments?
**Fix**: Append custom arguments for your training script after `--`:
```bash
python -m silent_bug_agent monitor train.py -- --batch-size 64 --lr 0.001
```
