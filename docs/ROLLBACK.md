# Rollback Plan

- Last Known Good Commit: `_(update with git rev-parse HEAD after each stable release)_`
- Feature Flag Control: No feature flags implemented yet — rollback via git revert only.

## Step-by-Step Emergency Revert

### Scenario A: Bad Patch Applied to User's Training Script
The pipeline applies patches to temp copies only during validation. If the user ran `sba fix --apply` and the patch was applied to their script:
1. The original file is backed up as `<script>.sba_backup.<timestamp>.py` in the same directory before any `--apply`.
2. Restore: `cp <script>.sba_backup.<timestamp>.py <script>.py`
3. Verify: `python <script>.py` — confirm it runs as before.

### Scenario B: Corrupt Experience Memory Store
If ChromaDB or SQLite trajectories become corrupt:
1. Locate store: `echo $SBA_HOME` or default `~/.silent_bug_agent/`
2. Delete ChromaDB: `rm -rf ~/.silent_bug_agent/chroma/`
3. Delete SQLite: `rm ~/.silent_bug_agent/trajectories.db`
4. The pipeline will start fresh with an empty memory store (cold start).
5. Verification: `python -m silent_bug_agent doctor` — should show 0 trajectories, no errors.

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
- [ ] `SBA_NO_DOTENV=1 python -m pytest tests/ -v` — all 404 tests pass
- [ ] `python -m silent_bug_agent scan examples/sample_scripts/buggy_mnist.py` — finds ≥1 bug
- [ ] `python -m silent_bug_agent doctor` — no structural errors
