# Threat Model & Security Controls

Every statement here was checked against the code on 2026-11-15. A "known gap" is a real, unmitigated risk.

## Threat Actors
- **The user's own training script**: it is executed, with the user's full permissions, by `monitor`, `diagnose`, `fix` and `wsp`.
- **A malicious or mistaken LLM response**: the model's reply becomes a patch, and that patch is executed during validation.
- **An untrusted checkpoint file** given to `wsp --checkpoint`.
- **A tampered memory store**: someone with file access to `~/.silent_bug_agent/` edits stored repairs that later become few-shot examples.

## Attack Surface & Mitigations

### CLI target path
- **Controls**: the path is resolved to an absolute path and must be an existing file.
- **Known gap**: the extension is not checked, and symlinks are followed. This is not a privilege boundary: the tool only does what the user running it could already do.

### Running the user's script (runtime monitor, validation gates, weight inspection)
- **Controls**: the script runs in a **separate process** with a time limit (`runtime_monitor_agent.py` with `isolate=True`, `validation/sandbox.py`). A crash or hang cannot take down the pipeline. Patched code is written to a temp folder and run from there; the user's file is not touched during validation.
- **Known gap**: this is isolation from crashes, **not a security sandbox**. The child process has the user's full file and network access. There is no `seccomp`, container or AppContainer. Do not run the tool on a script you do not trust.
- **Known gap**: memory limits (`RLIMIT_AS` on POSIX, Job Objects on Windows) exist in `sandbox.py` but are off unless `max_memory_mb` is set, and no test covers them.

### LLM response turned into a patch
- **Controls**:
  - replies must match a strict JSON schema (`agents/prompts.py`, `semantic/client.py`);
  - every patch passes `repair/patch_utils.patch_guard` before it is run: it must parse, must not add process, network or file-deletion calls (`subprocess`, `socket`, `os.system`, `eval`, `exec`, ...), must not remove a `backward()` or `step()` call, and must not delete most of the script;
  - a patch may change at most 40 lines;
  - the original file is only written after both validation gates pass, and a `.orig` backup is kept.
- **Approval**: `fix --dry-run` writes nothing and stores the validated patch; `apply` writes exactly that patch, and refuses if the file changed since. Plain `fix` (no `--dry-run`) writes a validated patch without asking. There is no `--apply` flag.
- **Known gap**: `patch_guard` is a deny-list of text patterns. It lowers the risk; it cannot prove a patch is harmless. A patch is still executed in the validation subprocess with the user's permissions.

### Checkpoint files (`wsp --checkpoint`)
- **Controls**: checkpoints are loaded with `torch.load(..., weights_only=True)` or `safetensors`, so a pickle cannot run code. A file that needs full unpickling is refused with an explanation. `tests/test_wsp.py` checks this with a pickle that would create a file if executed.
- **Known gap**: a very large checkpoint is loaded fully into memory.

### Experience store (SQLite + ChromaDB under `~/.silent_bug_agent/`)
- **Controls**: ordinary file permissions. `agents/meta_evolution.py` drops stored examples whose use correlates with worse outcomes.
- **Known gap**: no signing or integrity check. Anyone who can write the folder can plant examples that are shown to the LLM.

### Secrets
- **Controls**: keys are read from `.env`, which is in `.gitignore`. Tests set `SBA_NO_DOTENV=1` and remove key variables (`tests/conftest.py`), so they never load or send a real key.
- **Known gap**: nothing stops a user from committing or zipping `.env` by hand. A key that has been shared in any archive must be revoked.

## What the tool does and does not guarantee
- It never writes the target script before a patch has passed both gates; with `--dry-run` it never writes it at all.
- It never edits the bundled examples under `examples/`.
- `wsp` never modifies a script or a checkpoint.
- Network calls go only to the configured LLM endpoints, **from the tool itself**. The user's script can make any network call it likes.
