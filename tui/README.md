# silent-bug-agent TUI

A terminal UI for `silent-bug-agent` built using **Pastel** and **Ink** (React for the CLI).

## Features

- **Full Command Parity**: `scan`, `monitor`, `diagnose`, `fix`, `doctor`.
- **Live Event Streaming**: Subprocesses stream NDJSON events for real-time progress meters, spinners, and phase steppers.
- **Interactive Patch Approval**: Inspect syntax-colored diffs and choose between `[Apply to File]`, `[Keep Dry-Run]`, and `[Retry Fix]`.
- **Pastel Palette Theme**: Mint, cyan, lavender, and amber pastel accents with rounded border boxes.

## Setup

```bash
cd tui
npm install
npm run build
```

## Running the TUI

```bash
# Launch interactive menu
node dist/cli.js

# Or link globally:
npm link
sba-ui

# Scan a script:
sba-ui scan ../examples/sample_scripts/buggy_mnist.py --semantic

# Diagnose root causes:
sba-ui diagnose ../examples/sample_scripts/buggy_mnist.py

# Run the autonomous repair loop:
sba-ui fix ../examples/sample_scripts/buggy_mnist.py
```

## From the Python CLI

You can also launch the TUI directly from Python:

```bash
python -m silent_bug_agent ui
```

