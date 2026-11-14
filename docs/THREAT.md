# Threat Model & Security Controls

## Threat Actors
- **Anonymous User / Researcher**: Has shell access to the machine; runs `sba fix` on their own training scripts
- **Malicious Local Script**: A user-supplied Python training script that attempts to escape sandbox via subprocess calls, file writes, or environment manipulation
- **Compromised LLM Response**: The LLM provider returns a maliciously crafted JSON response designed to inject code or path traversal into the patch
- **Compromised Memory Store**: An attacker with file access to `~/.silent_bug_agent/` injects poisoned trajectories for future few-shot retrieval

## Attack Surface & Mitigations

### Entry Point: CLI Target Script Path (`--target / positional arg`)
- **Potential Abuse**: Path traversal to access `/etc/passwd` or system files; symlink attack to redirect analysis to sensitive files
- **Active Controls**: Path is resolved to absolute; must exist and be a `.py` file (file extension check in `cli.py`)
- **Known Gap**: No check for symlinks pointing outside the user's project directory

### Entry Point: User-Supplied Training Script Execution (Validation Subprocess)
- **Potential Abuse**: Script calls `os.system()`, writes to arbitrary paths, or spawns background processes during validation
- **Active Controls**: Executed in an isolated subprocess with a fresh Python environment; `SBA_HOME` isolates memory writes; stdout parsed for `__METRICS__` protocol only
- **Known Gap**: No `seccomp` or OS-level sandbox (e.g., `bubblewrap`); malicious scripts have full user-level filesystem access
- **Known Gap**: Windows Job Object memory limits implemented but only tested on macOS

### Entry Point: LLM API Response (Groq/Cerebras JSON)
- **Potential Abuse**: Prompt injection in the LLM response causes the patch to include `import subprocess; subprocess.call(...)` or overwrite unrelated files
- **Active Controls**: Strict JSON schema validation in `agents/prompts.py`; patch is applied to a temp file copy, not the original; diff is printed and requires explicit `--apply` flag (or is applied to temp only)
- **Known Gap**: No static analysis of the generated patch diff for malicious patterns before application

### Entry Point: ChromaDB / SQLite Experience Store
- **Potential Abuse**: An attacker with local file access poisons trajectories to bias future few-shot prompting toward harmful patches
- **Active Controls**: Store lives in `~/.silent_bug_agent/`; access requires local user permissions; `meta_evolution.py` runs trajectory ablation to prune low-quality exemplars
- **Known Gap**: No cryptographic signing of stored trajectories; no integrity check on load

### Entry Point: PyTorch Hook Telemetry (Runtime Monitor)
- **Potential Abuse**: A model that deliberately produces NaN gradients to fool the invariant checker into suppressing real bug signals
- **Active Controls**: `runtime/numerics.py` validates tensor dtype and finiteness; anomaly detection flags NaN/Inf regardless of source
- **Known Gap**: Hook-based telemetry can be bypassed by scripts that detach tensors before backward

## Security Properties We Guarantee
- Real LLM API keys are NEVER loaded in test environments (`SBA_NO_DOTENV=1`)
- Patches are diffed and applied to temp copies first; original files modified only on explicit acceptance
- No network calls outside of the LLM provider endpoints configured in `.env`
