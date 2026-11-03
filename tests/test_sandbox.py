"""Unit tests for SubprocessSandbox."""

import sys
import textwrap
from pathlib import Path
import pytest

from silent_bug_agent.validation.sandbox import (
    ExecutionResult,
    SandboxConfig,
    SubprocessSandbox,
)


def test_sandbox_run_code_success():
    sandbox = SubprocessSandbox()
    code = "print('Hello from sandbox!')"
    result = sandbox.run_code(code)
    assert result.success
    assert result.returncode == 0
    assert "Hello from sandbox!" in result.stdout
    assert not result.timed_out
    assert result.duration_seconds > 0


def test_sandbox_extract_telemetry():
    sandbox = SubprocessSandbox()
    code = textwrap.dedent("""
        import json
        print("Starting training...")
        print('__METRICS__:{"loss": 0.42, "final_loss": 0.15, "accuracy": 0.96}')
        print("Finished.")
    """)
    result = sandbox.run_code(code)
    assert result.success, f"Failed with stderr: {result.stderr}"
    assert result.metrics["loss"] == 0.42
    assert result.metrics["final_loss"] == 0.15
    assert result.metrics["accuracy"] == 0.96


def test_sandbox_seed_propagation():
    sandbox = SubprocessSandbox()
    code = textwrap.dedent("""
        import os
        import sys

        seed_env = os.environ.get("TORCH_SEED")
        seed_arg = None
        if "--seed" in sys.argv:
            seed_arg = sys.argv[sys.argv.index("--seed") + 1]

        print(f"ENV_SEED:{seed_env}")
        print(f"ARG_SEED:{seed_arg}")
    """)
    result = sandbox.run_code(code, seed=1337)
    assert result.success, f"Failed with stderr: {result.stderr}"
    assert "ENV_SEED:1337" in result.stdout
    assert "ARG_SEED:1337" in result.stdout


def test_sandbox_timeout_handling():
    config = SandboxConfig(timeout_seconds=0.5)
    sandbox = SubprocessSandbox(config=config)
    code = textwrap.dedent("""
        import time
        time.sleep(3.0)
    """)
    result = sandbox.run_code(code)
    assert not result.success
    assert result.timed_out
    assert result.returncode == -1
    assert "timed out" in (result.error_message or "").lower()


def test_sandbox_nonzero_exit_code():
    sandbox = SubprocessSandbox()
    code = textwrap.dedent("""
        import sys
        print("Error occurred!", file=sys.stderr)
        sys.exit(3)
    """)
    result = sandbox.run_code(code)
    assert not result.success
    assert result.returncode == 3
    assert "Error occurred!" in result.stderr


def test_sandbox_nonexistent_script():
    sandbox = SubprocessSandbox()
    result = sandbox.run_script("non_existent_script_xyz.py")
    assert not result.success
    assert result.returncode == -1
    assert "Script not found" in (result.error_message or "")


def test_sandbox_telemetry_forwarder():
    from silent_bug_agent.validation.sandbox import TelemetryForwarder

    forwarded = []
    forwarder = TelemetryForwarder(callbacks=[lambda m: forwarded.append(m)])
    sandbox = SubprocessSandbox(telemetry_forwarder=forwarder)

    code = textwrap.dedent("""
        print('__METRICS__:{"step": 1, "loss": 0.88}')
        print('__METRICS__:{"step": 2, "loss": 0.44}')
    """)
    res = sandbox.run_code(code)
    assert res.success
    assert len(forwarded) >= 1
    assert any(item.get("loss") == 0.44 for item in forwarded)
    assert len(forwarder.history) >= 1


def test_sandbox_resource_limits():
    # Test setting CPU thread limits and memory limits in config
    config = SandboxConfig(
        max_cpu_cores=2,
        max_memory_mb=512,
    )
    sandbox = SubprocessSandbox(config=config)
    code = textwrap.dedent("""
        import os
        print("OMP:", os.environ.get("OMP_NUM_THREADS"))
        print("TORCH:", os.environ.get("TORCH_NUM_THREADS"))
        print("MEM:", os.environ.get("DL_MAX_MEMORY_MB"))
    """)
    res = sandbox.run_code(code)
    assert res.success, f"Failed with stderr: {res.stderr}"
    assert "OMP: 2" in res.stdout
    assert "TORCH: 2" in res.stdout
    assert "MEM: 512" in res.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not enforce RLIMIT_AS; Windows and Linux do")
def test_sandbox_memory_limit_stops_exceeding_child():
    config = SandboxConfig(max_memory_mb=64)
    sandbox = SubprocessSandbox(config=config)
    # Child process attempts to allocate 300MB, exceeding the 64MB limit
    code = textwrap.dedent("""
        a = bytearray(300 * 1024 * 1024)
        print("ALLOCATED_300MB")
    """)
    res = sandbox.run_code(code)
    # The child process must be stopped / fail due to memory limit
    assert not res.success
    assert "ALLOCATED_300MB" not in res.stdout
    assert res.returncode != 0 or res.timed_out
