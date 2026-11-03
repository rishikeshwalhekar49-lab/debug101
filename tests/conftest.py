"""Every test runs without real keys, without `.env`, and with its own memory folder."""

import pytest

KEY_VARIABLES = ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GROQ_BASE_URL", "CEREBRAS_BASE_URL", "GROQ_MODEL")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path_factory):
    import os

    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("SBA_MODEL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SBA_NO_DOTENV", "1")
    monkeypatch.setenv("SBA_HOME", str(tmp_path_factory.mktemp("sba_home")))
    yield
