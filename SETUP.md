# Setup and Installation Guide

This guide covers everything required to install, configure, verify, and run the **Deep Learning Bug Detector (`silent-bug-agent`)**, including the Python multi-agent backend, the read-only weight inspector (`wsp`), and the Pastel + Ink terminal UI (TUI).

---

## 1. System Requirements

| Component | Minimum Version | Recommended | Notes |
|---|---|---|---|
| **Python** | 3.10+ | 3.12 | Core multi-agent pipeline & PyTorch. 3.12 is the version in `.python-version`; CI tests 3.10 and 3.12. Python 3.14 has not been tested with the full suite. |
| **Node.js** | 18.0.0+ | 20.x or 22.x | Required for the interactive Pastel + Ink TUI |
| **npm** | 9.x+ | 10.x+ | Package manager for `tui/` |
| **Operating System** | Windows 10/11, macOS (Apple Silicon M1-M4 & Intel), or Linux | Any | Fully CPU-compatible (no dedicated GPU required) |

> [!TIP]
> **macOS Prerequisites via Homebrew**:
> If you don't have Python 3 or Node.js installed yet on macOS, install them with [Homebrew](https://brew.sh):
> ```bash
> brew install python node
> ```

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
> - **macOS / Apple Silicon**: Native `arm64` wheels for PyTorch and ChromaDB install automatically via pip. Apple Silicon Metal Performance Shaders (`mps`) and standard CPU execution are both supported out of the box with zero extra drivers.
> - **Linux / Windows with Nvidia GPU**: If you want CUDA acceleration, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org) (e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).
> - **CPU only**: Everything runs completely on CPU without any dedicated GPU required.

---

### Step 2.4: Build the Pastel + Ink Terminal UI (TUI)

The rich interactive dashboard is built using Node.js, React, Ink, and Pastel. It lives in the `tui/` directory:

```bash
cd tui
npm ci
npm run build
cd ..
```

`npm ci` installs the exact versions recorded in `package-lock.json`. This compiles TypeScript source files in `tui/commands/` and `tui/components/` into `tui/dist/cli.js`.

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

The **Fix** screen always validates a patch first and shows it while your file is still unchanged. It then asks what to do:

* **Apply the fix** writes exactly the patch shown, and keeps the original as `<name>.py.orig`.
* **Do not apply** leaves the file as it is.
* **Try another fix** turns this patch down and runs the pipeline again for a different one. Without an LLM key each bug has a single rule-based fix, so there may be no alternative; the screen says so.

For the bundled examples in the menu, "Apply the fix" writes the fixed version to `fixed_examples/<name>.py` and leaves the example itself alone, because the tests rely on its planted bug. Leaving a screen with `Esc`, `b` or `q` stops the command it was running.

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

#### 4. Autonomous Repair Pipeline (`fix`, then `apply`)
Runs the full supervisor loop: Perception $\rightarrow$ Diagnosis $\rightarrow$ Patch Synthesis $\rightarrow$ Gate 1 (Overfit Oracle) $\rightarrow$ Gate 2 (Multi-Seed Sandbox).

> [!IMPORTANT]
> **The bundled examples are never edited.** The tests and the published results depend on the bugs planted in `examples/`. For those files `fix` validates and shows the patch, and `apply` writes the fixed version to `fixed_examples/<name>.py` instead of changing the example. To have a file patched in place, work on your own script or a copy.

```bash
# Make a copy to work on
cp examples/sample_scripts/benchmark_06_dead_relu.py my_train.py     # Windows: copy examples\sample_scripts\benchmark_06_dead_relu.py my_train.py

# Validate a patch and show it. Nothing is written.
python -m silent_bug_agent fix my_train.py --dry-run

# Write exactly the patch that was validated. The original is kept as my_train.py.orig
python -m silent_bug_agent apply my_train.py

# Not happy with that patch? Turn it down and ask for a different one (best with an LLM key)
python -m silent_bug_agent fix my_train.py --dry-run --another

# Or validate and write in one step
python -m silent_bug_agent fix my_train.py
```

#### 5. Weight inspection (`wsp`, read-only)
Reads a model's weights and reports anomalies that the code does not show: NaN or exploded values, a zeroed weight matrix, biases too negative for a ReLU to fire, collapsed rank. It never changes a script or a checkpoint.
```bash
# Train the script for 60 steps in the sandbox, then inspect the weights it produced
python -m silent_bug_agent wsp examples/sample_scripts/benchmark_06_dead_relu.py

# Inspect a saved state_dict instead (loaded with weights_only=True)
python -m silent_bug_agent wsp --checkpoint model.pt
```
The exit code is 1 when a warning or critical anomaly is found, 0 when the weights look ordinary, and 2 when no weights could be read.

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
# Weight inspector
python -m pytest tests/test_wsp.py -v

# Dry run, apply and the TUI event stream
python -m pytest tests/test_apply_and_events.py -v

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
**Fix**: Ensure your Node.js version is $\ge 18$. Run `node -v` to check. In `tui/`, delete `node_modules` and reinstall from the lockfile:
```bash
cd tui
rm -rf node_modules  # Windows: Remove-Item -Recurse -Force node_modules
npm ci
npm run build
cd ..
```
Keep `package-lock.json`: it is what makes everyone build with the same dependency versions.

### Q: `sqlite3.OperationalError: unable to open database file` on Windows
**Fix**: In restricted environments `~/.silent_bug_agent/` may not be writable. The agent then uses `<project_root>/.silent_bug_agent/`. If that cannot be written either, it keeps the run's memory in RAM, says so in the report, and loses it when the run ends. To choose the folder yourself, set `SBA_HOME` in `.env`:
```dotenv
SBA_HOME=./.silent_bug_agent
```

### Q: macOS: `command not found: python` or points to system Python
**Fix**: On macOS, modern Python installs as `python3`. Always use `python3 -m venv .venv` when creating the environment. Once activated (`source .venv/bin/activate`), `python` will automatically resolve to the virtual environment's Python interpreter.

### Q: macOS: Terminal rendering in Terminal.app, iTerm2, Warp, Ghostty
**Fix**: The Pastel + Ink TUI uses standard 24-bit ANSI truecolor. It works out of the box across Terminal.app, iTerm2, Warp, Ghostty, Kitty, and Alacritty. For optimal visual display of spinners and step icons, ensure UTF-8 encoding is enabled in your terminal preferences (standard default on macOS).

### Q: Does the multi-seed validation gate take too long?
**Fix**: Gate 2 runs multiple isolated subprocesses across distinct random seeds to compute statistical tests ($p < 0.05$). For rapid local prototyping, decrease seed counts:
```bash
python -m silent_bug_agent fix my_train.py --seeds 3 --no-llm
```

### Q: How do I pass script-specific arguments?
**Fix**: Append custom arguments for your training script after `--`:
```bash
python -m silent_bug_agent monitor train.py -- --batch-size 64 --lr 0.001
```
