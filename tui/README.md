# silent-bug-agent TUI

A terminal UI for `silent-bug-agent` built using **Pastel** and **Ink** (React for the CLI).

## Features

- **Screens**: `scan`, `monitor`, `diagnose`, `fix`, `wsp` (read-only weight inspection), `doctor`.
- **Live event streaming**: each screen runs `python -m silent_bug_agent <command> --events` and renders the NDJSON events it prints.
- **Real patch approval**: the Fix screen always runs the pipeline as a dry run, so your file is unchanged when the validated diff is shown. It then offers three choices:
  - **Apply the fix** runs `python -m silent_bug_agent apply <script>`, which writes exactly that patch and keeps the original as `<name>.py.orig`;
  - **Do not apply** leaves the file as it was;
  - **Try another fix** runs `fix --dry-run --another`: the patch is turned down, remembered, and the pipeline looks for a different one. If there is none, the screen says so.
- **Protected examples**: the bundled scripts under `examples/` are never edited. For them "Apply the fix" writes the fixed version to `fixed_examples/<name>.py`.
- **Leaving stops the work**: `Esc`, `b` or `q` stops the Python command the screen started.

A relative script path is resolved from the folder you launched from. There are no automated tests for the UI itself; CI checks that it builds, and the event stream is tested from Python (`tests/test_apply_and_events.py`).

## Setup

```bash
cd tui
npm ci
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

# Validate a patch for your own script, then choose whether to apply it:
sba-ui fix path/to/my_train.py

# Read-only weight inspection:
sba-ui wsp ../examples/sample_scripts/benchmark_06_dead_relu.py
```

## From the Python CLI

You can also launch the TUI directly from Python:

```bash
python -m silent_bug_agent ui
```

