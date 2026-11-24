"""Subprocess sandbox for running training scripts and patch candidates safely."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Telemetry Forwarder
# ---------------------------------------------------------------------------

class TelemetryForwarder:
    """Captures, buffers, and dispatches real-time training telemetry from sandboxed runs."""

    def __init__(self, callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None):
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = list(callbacks or [])
        self.history: List[Dict[str, Any]] = []

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a new listener for emitted telemetry metrics."""
        self.callbacks.append(callback)

    def forward(self, metrics: Dict[str, Any]) -> None:
        """Buffer and broadcast a telemetry payload to all registered callbacks."""
        if not metrics:
            return
        payload = dict(metrics)
        self.history.append(payload)
        for cb in self.callbacks:
            try:
                cb(payload)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Results & Config Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of running a script or code snippet in the sandbox."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    run_report: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass
class SandboxConfig:
    """Configuration for subprocess isolation and resource limits."""
    timeout_seconds: float = 60.0
    max_memory_mb: Optional[int] = None
    max_cpu_cores: Optional[int] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    python_executable: str = sys.executable
    cwd: Optional[str] = None


# ---------------------------------------------------------------------------
# Windows Job Object Limit Helper
# ---------------------------------------------------------------------------

def _apply_windows_memory_limit(proc: subprocess.Popen, max_memory_mb: int) -> Optional[Any]:
    """Apply a hard process memory limit to a Windows process using Job Objects."""
    if sys.platform != "win32" or not proc or not hasattr(proc, "_handle"):
        return None

    try:
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        # Explicit 64-bit handle typing
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]

        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]

        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryLimit", ctypes.c_size_t),
                ("PeakJobMemoryLimit", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
        limits.ProcessMemoryLimit = max_memory_mb * 1024 * 1024
        limits.JobMemoryLimit = max_memory_mb * 1024 * 1024
        JobObjectExtendedLimitInformation = 9

        ok = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        if ok:
            kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
        return job
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Subprocess Sandbox
# ---------------------------------------------------------------------------

class SubprocessSandbox:
    """Runs Python training workloads in an isolated child process with resource limits."""

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        telemetry_forwarder: Optional[TelemetryForwarder] = None,
    ):
        self.config = config or SandboxConfig()
        self.telemetry_forwarder = telemetry_forwarder or TelemetryForwarder()

    def _build_env(self, seed: Optional[int] = None) -> Dict[str, str]:
        env = os.environ.copy()
        # Ensure project root is on PYTHONPATH
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else repo_root
        )

        if seed is not None:
            env["PYTHONHASHSEED"] = str(seed)
            env["DL_SEED"] = str(seed)
            env["TORCH_SEED"] = str(seed)

        # Child output is decoded as UTF-8, so a Windows code page cannot break it
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Enforce CPU core / thread limits via environment
        if self.config.max_cpu_cores is not None:
            core_str = str(self.config.max_cpu_cores)
            env["OMP_NUM_THREADS"] = core_str
            env["MKL_NUM_THREADS"] = core_str
            env["OPENBLAS_NUM_THREADS"] = core_str
            env["TORCH_NUM_THREADS"] = core_str
            env["VECLIB_MAXIMUM_THREADS"] = core_str
            env["NUMEXPR_NUM_THREADS"] = core_str

        # Enforce memory hint via environment
        if self.config.max_memory_mb is not None:
            env["DL_MAX_MEMORY_MB"] = str(self.config.max_memory_mb)

        env.update(self.config.env_overrides)
        return env

    def _extract_telemetry(self, stdout: str) -> Dict[str, Any]:
        """Extract JSON telemetry from lines starting with '__METRICS__:'."""
        metrics: Dict[str, Any] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("__METRICS__:"):
                payload = line[len("__METRICS__:"):].strip()
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        metrics.update(data)
                except json.JSONDecodeError:
                    pass
        return metrics

    def run_script(
        self,
        script_path: str | Path,
        args: Optional[List[str]] = None,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a Python file in a sandboxed subprocess with resource limits."""
        path = Path(script_path).resolve()
        if not path.is_file():
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr="",
                error_message=f"Script not found: {path}",
            )

        cmd = [self.config.python_executable, str(path)]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if args:
            cmd.extend(args)
        return self._execute(cmd, seed, timeout, cwd or self.config.cwd or str(path.parent))

    def run_measured(
        self,
        code: str,
        *,
        seed: int = 0,
        mode: str = "run",
        max_steps: Optional[int] = None,
        script_args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        as_file: Optional[str] = None,
        save_weights: Optional[str] = None,
        trace: bool = False,
    ) -> ExecutionResult:
        """Run any training code under `script_runner`, which needs no help from the code.

        `save_weights` is a file the child writes the trained models' tensors to."""
        temp_dir = Path(tempfile.mkdtemp(prefix="dl_sandbox_"))
        try:
            script_file = temp_dir / "workload.py"
            script_file.write_text(code, encoding="utf-8")
            result_file = temp_dir / "result.json"
            cmd = [self.config.python_executable, "-m", "silent_bug_agent.validation.script_runner", str(script_file),
                   "--mode", mode, "--seed", str(seed), "--result", str(result_file)]
            if max_steps is not None:
                cmd += ["--max-steps", str(max_steps)]
            if as_file:
                cmd += ["--as-file", str(as_file)]
            if save_weights:
                cmd += ["--save-weights", str(save_weights)]
            if trace:
                cmd += ["--trace"]
            if script_args:
                cmd += ["--", *script_args]
            effective_cwd = cwd or self.config.cwd or (str(Path(as_file).parent) if as_file else str(temp_dir))
            result = self._execute(cmd, seed, timeout, effective_cwd, extra_path=str(Path(effective_cwd).resolve()))
            if result_file.is_file():
                report = json.loads(result_file.read_text(encoding="utf-8"))
                result.run_report = report
                if report.get("status") == "error" and report.get("error"):
                    result.error_message = str(report["error"])[-1500:]
            return result
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _execute(
        self,
        cmd: List[str],
        seed: Optional[int],
        timeout: Optional[float],
        cwd: Optional[str],
        extra_path: Optional[str] = None,
    ) -> ExecutionResult:
        timeout_sec = timeout if timeout is not None else self.config.timeout_seconds
        env = self._build_env(seed=seed)
        if extra_path:
            env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + extra_path
        start_time = time.perf_counter()

        preexec_fn = None
        # On POSIX, apply hard address-space limits via resource.setrlimit
        if sys.platform != "win32" and self.config.max_memory_mb is not None:
            def _posix_limits():
                try:
                    import resource
                    bytes_lim = self.config.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (bytes_lim, bytes_lim))
                except Exception:
                    pass
            preexec_fn = _posix_limits

        proc = None
        win_job = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=cwd,
                preexec_fn=preexec_fn,
            )

            # On Windows, assign process to Job Object with memory limit
            if sys.platform == "win32" and self.config.max_memory_mb is not None:
                win_job = _apply_windows_memory_limit(proc, self.config.max_memory_mb)

            stdout, stderr = proc.communicate(timeout=timeout_sec)
            duration = time.perf_counter() - start_time
            metrics = self._extract_telemetry(stdout)
            if metrics:
                self.telemetry_forwarder.forward(metrics)

            return ExecutionResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                duration_seconds=duration,
                metrics=metrics,
                error_message=None if proc.returncode == 0 else f"Process exited with code {proc.returncode}",
            )
        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
                stdout, stderr = proc.communicate()
            else:
                stdout, stderr = "", ""
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                returncode=-1,
                stdout=stdout or "",
                stderr=stderr or "",
                timed_out=True,
                duration_seconds=duration,
                error_message=f"Execution timed out after {timeout_sec} seconds",
            )
        except Exception as exc:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=str(exc),
                timed_out=False,
                duration_seconds=duration,
                error_message=f"Subprocess launch failed: {exc}",
            )
        finally:
            if win_job and sys.platform == "win32":
                try:
                    ctypes.windll.kernel32.CloseHandle(win_job)
                except Exception:
                    pass

    def run_code(
        self,
        code: str,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """Write Python code to a dedicated temporary directory and run it inside the sandbox."""
        temp_dir = Path(tempfile.mkdtemp(prefix="dl_sandbox_"))
        script_file = temp_dir / "workload.py"
        try:
            script_file.write_text(code, encoding="utf-8")
            effective_cwd = cwd or str(temp_dir)
            return self.run_script(
                script_path=script_file,
                seed=seed,
                timeout=timeout,
                cwd=effective_cwd,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
