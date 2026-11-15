# Rollback Plan

- Last Known Good Commit: `_(update with git rev-parse HEAD after each stable release)_`
- Feature Flag Control: No feature flags implemented yet — rollback via git revert only.

## Step-by-Step Emergency Revert

### Scenario A: Bad Patch Applied to User's Training Script
During validation, patches run on temp copies only. The script is written by `fix` (without `--dry-run`) or by `apply`:
1. Before the first write, the original is copied to `<script>.py.orig` in the same folder. An existing `.orig` is never overwritten, so after several fixes it still holds the true original.
2. Restore: `cp train.py.orig train.py` (Windows: `copy train.py.orig train.py`)
3. Verify: `python train.py` — confirm it runs as before.

If a bundled example under `examples/` was changed (by an old version of the tool, or by hand), restore it from git: `git checkout -- examples/`. The current version refuses to edit those files.

### Scenario B: Corrupt Experience Memory Store
If ChromaDB or SQLite trajectories become corrupt:
1. Locate store: `echo $SBA_HOME` or default `~/.silent_bug_agent/`
2. Delete ChromaDB: `rm -rf ~/.silent_bug_agent/chroma/`
3. Delete SQLite: `rm ~/.silent_bug_agent/experience.sqlite3`
4. The pipeline will start fresh with an empty memory store (cold start).
5. Verification: `python -m silent_bug_agent doctor` — no errors. (`doctor` does not print a trajectory count.)

### Scenario C: Bad Code Change to the Package
1. Identify the last good commit: `git log --oneline -20`
2. Revert: `git revert <bad_commit_hash> -m 1`
3. Or hard reset (if no commit to preserve): `git checkout <good_commit_hash> -- silent_bug_agent/`
4. Reinstall in editable mode: `pip install -e .`
5. Run tests: `SBA_NO_DOTENV=1 python -m pytest tests/ -v`

### Scenario D: Broken Taxonomy
If `taxonomy/bug_taxonomy.yaml` is modified and breaks the causal engine:
1. `git diff HEAD silent_bug_agent/taxonomy/bug_taxonomy.yaml` — see changes
2. `git checkout HEAD -- silent_bug_agent/taxonomy/bug_taxonomy.yaml` — restore to last commit
3. Run: `SBA_NO_DOTENV=1 python -m pytest tests/ -k taxonomy -v`

## Post-Rollback Verification Checklist
- [ ] `SBA_NO_DOTENV=1 python -m pytest tests/ -v` — every test passes (the count is in TESTING.md)
- [ ] `git status --short examples/` — prints nothing (no example was modified)
- [ ] `python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py` — finds ≥1 bug
- [ ] `python -m silent_bug_agent doctor` — no structural errors
